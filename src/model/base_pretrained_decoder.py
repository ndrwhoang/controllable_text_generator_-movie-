import logging
from src.output_util import ModelOutput

import torch
import torch.nn as nn

from transformers import GPT2LMHeadModel
from transformers import logging as t_logging
t_logging.set_verbosity_error()

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PretrainedDecoderModel(nn.Module):
    def __init__(self, config):
        super(PretrainedDecoderModel, self).__init__()
        self.config = config['model']
        self.pretrained_gpt = GPT2LMHeadModel.from_pretrained('distilgpt2')
        self.frozen = self.config.getboolean('freeze')
        
        if self.frozen:
            logger.info(' Freezing attention layers')
            # The lm head weights are tied to the embedding layer,
            # so we need to enable grad again manually
            # TODO: research if doing this is appropriate, i.e. the 
            # weight tieing is significant
            for param in self.pretrained_gpt.base_model.parameters():
                param.requires_grad = False
            for param in self.pretrained_gpt.lm_head.parameters():
                param.requires_grad = True
        
    def forward(self, batch):
        outputs = self.pretrained_gpt(
            input_ids = batch.input_ids,
            attention_mask = batch.attention_mask,
            labels = batch.input_ids,
            return_dict=True
        )
        
        return ModelOutput(outputs['loss'], outputs['logits'])
        # return outputs

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
    model = PretrainedDecoderModel(config)
    model.to(dataset.device)
    print(model)
    
    # for i, sample in enumerate(dataloader):
    #     sample.to_device(dataset.device)
    #     output = model(sample)
    #     print(output.loss)
    #     print(output.logits.size())


if __name__ == '__main__':
    import os
    import configparser
    
    config = configparser.ConfigParser()
    config.read(os.path.join('configs', 'config.cfg'))
    
    model_test(config)