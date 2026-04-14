"""
RL策略模块
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from typing import Optional, Dict, Any, Tuple


class RLSelectiveExpansionPolicy(nn.Module):
    def __init__(self, encoder_dim: int = 768, hidden_dim: int = 512, num_layers: int = 2, 
                 num_heads: int = 8, expansion_ratio: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.hidden_dim = hidden_dim
        self.expansion_ratio = expansion_ratio
        
        self.input_projection = nn.Linear(encoder_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.expansion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, block_embeddings: torch.Tensor, query_embeddings: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        batch_size, num_blocks, hidden_size = block_embeddings.shape
        
        if query_embeddings is not None:
            query_emb = query_embeddings.unsqueeze(1).repeat(1, num_blocks, 1)
            combined = block_embeddings + query_emb
        else:
            combined = block_embeddings
        
        projected = self.input_projection(combined)
        contextualized = self.transformer(projected)
        
        expansion_logits = self.expansion_head(contextualized).squeeze(-1)
        values = self.value_head(contextualized).squeeze(-1)
        
        return {
            'expansion_logits': expansion_logits,
            'values': values,
            'hidden_states': contextualized
        }
    
    def select_and_expand(self, block_embeddings: torch.Tensor, query_tokens: torch.Tensor, 
                          model: Any, deterministic: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size, num_blocks, hidden_size = block_embeddings.shape
        
        # 获取查询嵌入并投影到与块嵌入相同的维度
        query_embeddings = model.decoder.model.get_input_embeddings()(query_tokens).mean(1)
        
        # 确保查询嵌入与块嵌入维度相同
        if query_embeddings.shape[-1] != hidden_size:
            # 创建临时投影层，使用与查询嵌入相同的数据类型
            projection = torch.nn.Linear(query_embeddings.shape[-1], hidden_size).to(query_embeddings.device).to(query_embeddings.dtype)
            query_embeddings = projection(query_embeddings)
        
        outputs = self.forward(block_embeddings, query_embeddings)
        expansion_logits = outputs['expansion_logits']
        
        if deterministic:
            num_to_expand = max(1, int(num_blocks * self.expansion_ratio))
            top_indices = torch.topk(expansion_logits, num_to_expand, dim=1)[1]
        else:
            probs = torch.softmax(expansion_logits, dim=1)
            num_to_expand = max(1, int(num_blocks * self.expansion_ratio))
            top_indices = torch.multinomial(probs, num_to_expand, replacement=False)
        
        # 只保留选中的块
        expanded_embeddings = torch.gather(
            block_embeddings, 1, top_indices.unsqueeze(-1).repeat(1, 1, hidden_size)
        )
        
        return expanded_embeddings, {
            'selected_indices': top_indices,
            'expansion_logits': expansion_logits,
            'num_expanded': num_to_expand
        }
    
    def compute_loss(self, block_embeddings: torch.Tensor, query_embeddings: torch.Tensor, 
                     rewards: torch.Tensor, model: Any) -> Dict[str, torch.Tensor]:
        outputs = self.forward(block_embeddings, query_embeddings)
        logits = outputs['expansion_logits']
        values = outputs['values']
        
        batch_size, num_blocks = logits.shape
        
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
        
        # 计算策略梯度损失
        action_indices = torch.multinomial(probs, 1, replacement=False).squeeze(1)
        selected_log_probs = log_probs[range(batch_size), action_indices]
        
        advantages = rewards - values.mean(1)
        policy_loss = - (selected_log_probs * advantages.detach()).mean()
        
        # 计算价值损失
        value_loss = nn.functional.mse_loss(values.mean(1), rewards)
        
        # 计算熵损失
        entropy = - (probs * log_probs).sum(1).mean()
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'total_loss': policy_loss + 0.5 * value_loss - 0.01 * entropy
        }