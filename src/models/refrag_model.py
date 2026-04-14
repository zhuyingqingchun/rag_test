"""
REFRAG主模型
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from typing import Optional, Dict, Any, List

from .encoder import LightweightEncoder
from .projection import ProjectionLayer, AdaptiveProjectionLayer
from .decoder import RefragDecoder
from .rl_policy import RLSelectiveExpansionPolicy


class RefragModel(nn.Module):
    def __init__(self, encoder_name: str = "roberta-base", decoder_name: str = "meta-llama/Llama-2-7b-hf", 
                 block_size: int = 32, use_rl_policy: bool = True, device: str = "cuda:0", **kwargs):
        super().__init__()
        self.block_size = block_size
        self.use_rl_policy = use_rl_policy
        self.device = device
        
        self.encoder = LightweightEncoder(
            model_name=encoder_name, block_size=block_size,** kwargs.get('encoder_config', {})
        ).to(device)
        
        # 先创建解码器以获取其隐藏维度
        self.decoder = RefragDecoder(
            model_name=decoder_name, block_size=block_size,** kwargs.get('decoder_config', {})
        )
        # 手动移动解码器到指定设备
        self.decoder.model = self.decoder.model.to(device)
        
        # 使用解码器的实际隐藏维度
        decoder_dim = self.decoder.hidden_size
        
        self.projection = ProjectionLayer(
            encoder_dim=self.encoder.hidden_size,
            decoder_dim=decoder_dim,
            **kwargs.get('projection_config', {})
        ).to(device)
        
        if use_rl_policy:
            self.rl_policy = RLSelectiveExpansionPolicy(
                encoder_dim=self.encoder.hidden_size, **kwargs.get('rl_config', {})
            ).to(device)
        else:
            self.rl_policy = None
        
        self.training_stage = "pretrain"  # pretrain, cpt, rl, sft
    
    def set_training_stage(self, stage: str):
        self.training_stage = stage
        
        # 根据训练阶段设置不同的参数冻结策略
        if stage == "pretrain":
            self.encoder.unfreeze()
            self.projection.unfreeze()
            self.decoder.freeze()
            if self.rl_policy:
                self.rl_policy.eval()
        elif stage == "cpt":
            self.encoder.freeze()
            self.projection.unfreeze()
            self.decoder.unfreeze()
            if self.rl_policy:
                self.rl_policy.eval()
        elif stage == "rl":
            self.encoder.freeze()
            self.projection.freeze()
            self.decoder.freeze()
            if self.rl_policy:
                self.rl_policy.train()
        elif stage == "sft":
            self.encoder.freeze()
            self.projection.freeze()
            self.decoder.unfreeze()
            if self.rl_policy:
                self.rl_policy.eval()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        if self.training_stage == "pretrain":
            return self._forward_pretrain(input_ids, attention_mask, labels, **kwargs)
        elif self.training_stage == "cpt":
            return self._forward_cpt(input_ids, attention_mask, labels, **kwargs)
        elif self.training_stage == "rl":
            return self._forward_rl(input_ids, attention_mask, labels, **kwargs)
        elif self.training_stage == "sft":
            return self._forward_sft(input_ids, attention_mask, labels, **kwargs)
        else:
            raise ValueError(f"Unknown training stage: {self.training_stage}")
    
    def _forward_pretrain(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                         labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        block_embeddings = self.encoder(input_ids, attention_mask)
        projected_embeddings = self.projection(block_embeddings)
        
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            block_embeddings=projected_embeddings,
            labels=labels
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
            'block_embeddings': block_embeddings
        }
    
    def _forward_cpt(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                    labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        block_embeddings = self.encoder(input_ids, attention_mask)
        projected_embeddings = self.projection(block_embeddings)
        
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            block_embeddings=projected_embeddings,
            labels=labels
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
            'block_embeddings': block_embeddings
        }
    
    def _forward_rl(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                   labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        context_ids = kwargs.get('context_ids')
        context_mask = kwargs.get('context_mask')
        query_ids = kwargs.get('query_ids')
        query_mask = kwargs.get('query_mask')
        rewards = kwargs.get('rewards')
        
        block_embeddings = self.encoder(context_ids, context_mask)
        projected_embeddings = self.projection(block_embeddings)
        
        query_embeddings = self.decoder.model.get_input_embeddings()(query_ids).mean(1)
        
        rl_outputs = self.rl_policy.compute_loss(
            block_embeddings, query_embeddings, rewards, self
        )
        
        return rl_outputs
    
    def _forward_sft(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                    labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, Any]:
        context_ids = kwargs.get('context_ids')
        context_mask = kwargs.get('context_mask')
        
        block_embeddings = self.encoder(context_ids, context_mask)
        projected_embeddings = self.projection(block_embeddings)
        
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            block_embeddings=projected_embeddings,
            labels=labels
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
            'block_embeddings': block_embeddings
        }
    
    def generate(self, query_ids: torch.Tensor, context_ids: torch.Tensor, 
                 query_mask: Optional[torch.Tensor] = None, 
                 context_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        block_embeddings = self.encoder(context_ids, context_mask)
        projected_embeddings = self.projection(block_embeddings)
        
        if self.use_rl_policy and self.rl_policy is not None:
            projected_embeddings, rl_info = self.rl_policy.select_and_expand(
                projected_embeddings, query_ids, self, deterministic=True
            )
        
        generated_ids = self.decoder.generate(
            query_tokens=query_ids,
            block_embeddings=projected_embeddings,
            query_attention_mask=query_mask,
            **kwargs
        )
        
        return generated_ids
    
    def save_pretrained(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型配置
        config = {
            'encoder_name': self.encoder.encoder.config._name_or_path,
            'decoder_name': self.decoder.tokenizer.name_or_path,
            'block_size': self.block_size,
            'use_rl_policy': self.use_rl_policy
        }
        
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config, f)
        
        # 保存权重
        torch.save(self.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))
        
        # 保存tokenizer
        self.decoder.tokenizer.save_pretrained(save_path)
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        with open(os.path.join(model_path, 'config.json'), 'r') as f:
            config = json.load(f)
        
        model = cls(
            encoder_name=config['encoder_name'],
            decoder_name=config['decoder_name'],
            block_size=config['block_size'],
            use_rl_policy=config['use_rl_policy'],
            **kwargs
        )
        
        model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin')))
        return model


# 添加缺失的导入
import os
import json