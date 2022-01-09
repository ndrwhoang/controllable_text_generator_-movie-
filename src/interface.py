import wandb
import torch
import logging

from transformers import GPT2TokenizerFast
from src.dataset.base_dataset import MovieDataset
from src.model.base_pretrained_decoder import PretrainedDecoderModel
from src.trainer import Trainer

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Interface:
    def __init__(self, config):
        self.config = config
        self.config_dict = {s:dict(config.items(s)) for s in config.sections()}
    
    def run_trial_training(self, run_name='subset_test_run'):
        logger.info(' Starts trial tranining run with train data subset')
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        dataset = MovieDataset(self.config, tokenizer, 'train_subset')
        model = PretrainedDecoderModel(self.config)
        trainer = Trainer(self.config, model, dataset)
        
        trainer.run_train(run_name)
        