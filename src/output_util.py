import torch
from dataclasses import dataclass

@dataclass
class ModelOutput:
    # Output from model
    loss: torch.Tensor
    logits: torch.Tensor
    
@dataclass
class Batch:
    # Output from dataloader
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    tgt_mask: torch.Tensor
    
    def to_device(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.position_ids = self.position_ids.to(device)
        self.tgt_mask = self.tgt_mask.to(device)
