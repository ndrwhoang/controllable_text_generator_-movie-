import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.output_util import ModelOutput

class DecoderModel(nn.Module):
    def __init__(self, config):
        super(DecoderModel, self).__init__()
        config = config['model']
        self.word_emb = nn.Embedding(int(config['vocab_size']), int(config['hidden_dim']))
        self.pos_emb = nn.Embedding(int(config['input_len']), int(config['hidden_dim']))
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=int(config['hidden_dim']), 
                                                        nhead=int(config['n_attn_heads']),
                                                        dropout=float(config['dropout']),
                                                        batch_first=True
                                                        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, int(config['n_layers']))
        # self.layer_norm = nn.LayerNorm(int(config['hidden_dim']))
        self.lm_head = nn.Linear(int(config['hidden_dim']), int(config['vocab_size']))
    
    def loss_fn(self, logits, input_ids):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        # loss = F.cross_entropy(shift_logits.permute(0, 2, 1), shift_labels)
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss
    
    def forward(self, batch):
        w_embedding = self.word_emb(batch.input_ids)
        p_embedding = self.pos_emb(batch.position_ids)
        hidden = w_embedding + p_embedding
        memory = torch.zeros(hidden.size())    
        memory_mask = torch.zeros(memory.size(1), memory.size(1))    
        hidden = self.decoder(tgt=hidden,
                              memory=memory,
                              tgt_mask=batch.tgt_mask,
                              memory_mask=memory_mask)
        lm_logits = self.lm_head(hidden)
        loss = self.loss_fn(lm_logits, batch.input_ids)
        
        return ModelOutput(loss, lm_logits)
        
def model_test(config):
    print('starts model output test')
    from torch.utils.data import DataLoader
    from src.dataset.base_dataset import MovieDataset
    from transformers import GPT2TokenizerFast
    from data.auxiliary import CONSTANT
    
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    # tokenizer.add_special_tokens([CONSTANT.sep_token], special_tokens=True)
    dataset = MovieDataset(config, tokenizer, 'train_subset')
    dataloader = DataLoader(dataset, 
                            batch_size=2, 
                            shuffle=False,
                            collate_fn=dataset.collate_fn)
    model = DecoderModel(config)
    model.to(dataset.device)
    print(model)
    
    for i, sample in enumerate(dataloader):
        sample.to_device(dataset.device)
        output = model(sample)
        print(output.loss)
        print(output.logits.size())


if __name__ == '__main__':
    import os
    import configparser
    
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.cfg'))
    
    model_test(config)