"""
轻量编码器模块
"""

import torch
import torch.nn as nn
from transformers import RobertaModel
from typing import Optional


class LightweightEncoder(nn.Module):
    def __init__(self, model_name: str = "roberta-base", hidden_size: int = 768, block_size: int = 32, pooling: str = "mean"):
        super().__init__()
        self.block_size = block_size
        self.pooling = pooling
        self.hidden_size = hidden_size
        self.encoder = RobertaModel.from_pretrained(model_name)
        self.config = self.encoder.config
        
    def pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            weights = attention_mask.unsqueeze(-1).float()
            return (hidden_states * weights).sum(1) / weights.sum(1)
        elif self.pooling == "max":
            weights = attention_mask.unsqueeze(-1).float()
            hidden_states = hidden_states.masked_fill(weights == 0, -float("inf"))
            return hidden_states.max(1)[0]
        elif self.pooling == "cls":
            return hidden_states[:, 0]
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, return_all: bool = False) -> torch.Tensor:
        if input_ids.dim() == 3:
            batch_size, num_blocks, block_size = input_ids.shape
            input_ids = input_ids.view(-1, block_size)
            if attention_mask is not None:
                attention_mask = attention_mask.view(-1, block_size)
        else:
            batch_size, num_blocks = input_ids.shape[0], 1
        
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=return_all,
            return_dict=True
        )
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        pooled = self.pool(outputs.last_hidden_state, attention_mask)
        
        if num_blocks > 1:
            pooled = pooled.view(batch_size, num_blocks, -1)
        
        if return_all:
            return pooled, outputs.hidden_states
        return pooled