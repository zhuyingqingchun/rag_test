"""
解码器模块
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List, Dict, Any


class RefragDecoder(nn.Module):
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf", block_size: int = 32, 
                 use_flash_attention: bool = True, max_length: int = 4096, trust_remote_code: bool = False):
        super().__init__()
        self.block_size = block_size
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float32,  # 使用Float32以避免数据类型不匹配
            device_map=None  # 不自动分配设备，稍后手动移动
        )
        self.config = self.model.config
        self.hidden_size = self.config.hidden_size
        self.decoder_frozen = False
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.decoder_frozen = True
    
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
        self.decoder_frozen = False
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                block_embeddings: Optional[torch.Tensor] = None, 
                block_attention_mask: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        if block_embeddings is not None:
            if block_embeddings.dim() == 3:
                batch_size, num_blocks, hidden_size = block_embeddings.shape
                block_embeddings = block_embeddings.view(batch_size, num_blocks, hidden_size)
                
                if block_attention_mask is None:
                    block_attention_mask = torch.ones(
                        batch_size, num_blocks, dtype=torch.long, device=block_embeddings.device
                    )
                
                inputs_embeds = torch.cat([block_embeddings, inputs_embeds], dim=1)
                
                if attention_mask is not None:
                    attention_mask = torch.cat([block_attention_mask, attention_mask], dim=1)
                else:
                    attention_mask = torch.cat([
                        block_attention_mask,
                        torch.ones_like(input_ids, device=input_ids.device)
                    ], dim=1)
        
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs
    
    def generate(self, query_tokens: torch.Tensor, block_embeddings: torch.Tensor, 
                 query_attention_mask: Optional[torch.Tensor] = None, 
                 max_new_tokens: int = 256, temperature: float = 0.7, 
                 top_p: float = 0.9, top_k: int = 50, num_beams: int = 1, 
                 do_sample: bool = True) -> torch.Tensor:
        inputs_embeds = self.model.get_input_embeddings()(query_tokens)
        
        if block_embeddings.dim() == 3:
            inputs_embeds = torch.cat([block_embeddings, inputs_embeds], dim=1)
            
            if query_attention_mask is not None:
                batch_size, num_blocks = block_embeddings.shape[:2]
                block_attention_mask = torch.ones(
                    batch_size, num_blocks, dtype=torch.long, device=block_embeddings.device
                )
                attention_mask = torch.cat([block_attention_mask, query_attention_mask], dim=1)
            else:
                attention_mask = None
        else:
            attention_mask = query_attention_mask
        
        generated_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        return generated_ids