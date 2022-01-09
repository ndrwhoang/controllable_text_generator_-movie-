import os
import json
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from src.output_util import Batch
from data.auxiliary import CONSTANT

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
        
class MovieDataset(Dataset):
    def __init__(self, config, tokenizer, dataset):
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device('cuda' if 
                                   torch.cuda.is_available and 
                                   self.config['general'].getboolean('use_gpu') else
                                   'cpu')
        data_path = self.get_data_path(dataset)
        samples = self.make_samples(data_path)
        self.input_ids = self.convert_to_input(samples)
        
        self.n_samples = len(self.input_ids)
        logger.info(f' Finish processing data, n_samples : {self.n_samples}')
        
    def get_data_path(self, dataset):
        dataset_dict = {
            'train': self.config['data_path']['train'],
            'dev': self.config['data_path']['dev'], 
            'test': self.config['data_path']['test'],
            'train_subset': self.config['data_path']['train_subset']            
        }
        
        return dataset_dict[dataset]
    
    def make_samples(self, data_path):
        logger.info(f' Reading data rom {data_path}')
        data_path = os.path.join(*data_path.split('\\'))
        with open(data_path, 'r') as f:
            samples = json.load(f)
        f.close()
        logger.info(f' {len(samples)} samples contained in file')
        
        return samples
    
    def convert_to_input(self, samples):
        logger.info(' Start processing samples to input ids')
        input_ids = []
        
        for i_sample, sample in tqdm(enumerate(samples), total=len(samples)):
            prompt = ' '.join(sample['tags'].split(',')) + CONSTANT.sep_token + sample['title']
            synop = sample['plot_synopsis']
            input_text = prompt + ' ' + CONSTANT.sep_token + ' ' + synop
            input_id = self.tokenizer(input_text)['input_ids'][:1024]

            # input_ids.append(torch.tensor(input_id, device=self.device))
            input_ids.append(torch.tensor(input_id))
        
        return input_ids
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, item):
        return self.input_ids[item]
    
    def collate_fn(self, batch):
        input_ids = batch
        input_ids = pad_sequence(input_ids, batch_first=True)
        attention_mask = (input_ids != 0).long()
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(input_ids.size(-1))
        position_ids = torch.arange(0, input_ids.size(-1), dtype=torch.long)
        
        assert input_ids.size() == attention_mask.size()
        
        return Batch(input_ids, attention_mask, position_ids, tgt_mask)

def dataloader_test():
    print('make_samples test')
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.add_tokens([CONSTANT.sep_token], special_tokens=True)
    dataset = MovieDataset(config, tokenizer, 'train_subset')
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)
    
    for i, batch in enumerate(dataloader):
        if i == 5: break
        print('===========')
        print('*')
        print('*')
        print('*')
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        
        reverted_input = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        print(reverted_input)
        print(input_ids[0].tolist())
        print(attention_mask[0].tolist())
    
    
if __name__ == '__main__':
    import configparser
    from torch.utils.data import DataLoader
    from transformers import GPT2TokenizerFast
        
    config_path = os.path.join('configs', 'config.cfg')
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # dataloader_test()
    
    
    
    