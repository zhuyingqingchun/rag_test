"""
投影层模块
"""

import torch
import torch.nn as nn
from typing import Optional


class ProjectionLayer(nn.Module):
    def __init__(self, encoder_dim: int = 768, decoder_dim: int = 4096, hidden_dim: Optional[int] = None, 
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.hidden_dim = hidden_dim or decoder_dim
        
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "silu":
            act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, self.hidden_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, decoder_dim)
        )
        
        self.layer_norm = nn.LayerNorm(decoder_dim)
        self._init_weights()
    
    def _init_weights(self):
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, encoder_embeddings: torch.Tensor) -> torch.Tensor:
        projected = self.projection(encoder_embeddings)
        return self.layer_norm(projected)


class AdaptiveProjectionLayer(ProjectionLayer):
    def __init__(self, encoder_dim: int = 768, decoder_dim: int = 4096, hidden_dim: Optional[int] = None, 
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__(encoder_dim, decoder_dim, hidden_dim, dropout, activation)
        
        self.adaptive_scale = nn.Parameter(torch.ones(1))
        self.adaptive_bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, encoder_embeddings: torch.Tensor) -> torch.Tensor:
        projected = self.projection(encoder_embeddings)
        projected = projected * self.adaptive_scale + self.adaptive_bias
        return self.layer_norm(projected)