import random
import numpy as np
import os
import json
from tqdm import tqdm 
import wandb
import logging
from collections import namedtuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Trainer:
    def __init__(self, config, model, train_dataset, dev_dataset=None, checkpoint=None):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.dev_dataset = train_dataset
        if dev_dataset is not None:
            self.dev_dataset = dev_dataset
            
        # Seed 
        self.set_seed()
        
        # Device
        self.device = train_dataset.device
        logger.info(f'---------- device: {self.device}')
        
        # Load checkpoint if available
        self.model.to(self.device)
        if checkpoint is not None:
            model.load_state_dict(torch.load(os.path.join(*checkpoint.split('\\'))))
            
        # Create dataloaders
        self.train_loader, self.dev_loader = self._get_dataloader(self.config)
        
        # Create optimizer and lr_scheduler
        self.optimizer, self.lr_scheduler = self._get_optimizer(self.config)
        
        # Mixed Precision Training
        self.use_amp = self.config['training'].getboolean('mixed_precision')
        if self.use_amp:
            self.grad_scaler = GradScaler()
            
        # Gradient accumulation
        self.grad_accumulation_steps = float(self.config['training']['grad_accumulation_steps'])
        
        # Namedtuple class util for prediction output
        # TODO: find graceful way to handle this
        self.StepOutput = namedtuple('StepOutput', ['loss', 'logits'])        
        
        logger.info(f' Training with Mixed Precision: {self.use_amp} and Gradient Accumulation: {self.grad_accumulation_steps}')
    
    def set_seed(self):
        self.seed = int(self.config['general']['seed'])
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
    def _get_dataloader(self, config):
        train_loader = DataLoader(self.train_dataset,
                                      batch_size=int(config['training']['train_bsz']),
                                      collate_fn=self.train_dataset.collate_fn,
                                      shuffle=True,
                                      drop_last=True,
                                    #   num_workers=int(config['general']['num_worker']),
                                    #   pin_memory=True
                                      )
        dev_loader = DataLoader(self.dev_dataset,
                                    batch_size=int(config['training']['val_bsz']),
                                    collate_fn=self.dev_dataset.collate_fn,
                                    shuffle=False,
                                    drop_last=False,
                                    # num_workers=int(config['general']['num_worker']),
                                    # pin_memory=True
                                    )
        
        return train_loader, dev_loader
    
    def _get_optimizer(self, config):
        total_steps = len(self.train_loader) * int(self.config['training']['n_epochs'])
        model_params = list(self.model.named_parameters())
        no_decay = ['bias']
        optimized_params = [
            {
                'params':[p for n, p in model_params if not any(nd in n for nd in no_decay)], 
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }   
        ]
        optimizer = AdamW(optimized_params, lr=float(config['training']['lr']))
        lr_scheduler = OneCycleLR(
            optimizer, 
            max_lr=float(self.config['training']['max_lr']), 
            total_steps=total_steps
            )
        
        return optimizer, lr_scheduler
    
    def run_train(self, run_name: str):
        best_loss = self.run_validation()
        
        for epoch in range(int(self.config['training']['n_epochs'])):
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            epoch_loss = 0
            batch_loss = 0
            self.model.train()
            self.model.zero_grad(set_to_none=True)
            
            for i, batch in pbar:
                batch_loss = self._training_step(batch)
                
                # step
                if (i+1) % self.grad_accumulation_steps == 0:
                    if not self.use_amp:
                        self.optimizer.step()
                    else:
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                    self.model.zero_grad(set_to_none=True)
                
                pbar.set_description(f'(Training) Epoch: {epoch} - Steps: {i}/{len(self.train_loader)} - Loss: {batch_loss}', refresh=True)
                epoch_loss += batch_loss
                batch_loss = 0
            
            # val_loss = self.run_validation()
    
    def run_validation(self):
        pbar = tqdm(enumerate(self.dev_loader), total=len(self.dev_loader))
        self.model.eval()
        epoch_loss = 0
        
        for i, batch in pbar:
            step_output = self._prediction_step(batch)
            pbar.set_description(f'(Validating) Steps: {i}/{len(self.dev_loader)} - Loss: {step_output.loss}', refresh=True)
            epoch_loss += step_output.loss
        
        logger.info(f' Validation loss: {epoch_loss}')
        # wandb.log({'epoch_val_loss: epoch_loss'})
        
        return epoch_loss
    
    def _training_step(self, batch):
        batch.to_device(self.device)
        output = self.model(batch)
        loss = output.loss
        
        # Divide loss for accumulation
        if self.grad_accumulation_steps > 1:
            loss = loss / self.grad_accumulation_steps
        
        # Switching between backward schemes
        if self.use_amp:    
            self.grad_scaler.scale(loss).backward()
        else:
            loss.backward()
            
        # log
        # wandb.log({
        #     'train_total_loss': loss.item(),
        #     'learning_rate': float(self.optimizer.param_groups[0]['lr'])
        #     })
        
        return loss.detach()
    
    @torch.no_grad()
    def _prediction_step(self, batch):
        batch.to_device(self.device)
        output = self.model(batch)
        loss = output.loss
        
        wandb.log({
            'val_loss': loss.detach()
            })  
        
        return self.StepOutput(loss.detach(), output.logits.detach())

def trainer_test(config):
    from transformers import GPT2TokenizerFast
    
    from src.dataset.base_dataset import MovieDataset
    from src.model.base_pretrained_decoder import PretrainedDecoderModel
    
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    dataset = MovieDataset(config, tokenizer, 'train_subset')
    model = PretrainedDecoderModel(config)
    
    trainer = Trainer(config, model , dataset)
    trainer.run_train('test_run')
    
    
if __name__ == '__main__':
    import configparser
    
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.cfg'))
    
    trainer_test(config)
    
