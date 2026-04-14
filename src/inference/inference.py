"""
推理模块
"""

import torch
import time
from typing import Optional, Dict, Any, Tuple

from models import RefragModel


class InferenceConfig:
    def __init__(self, 
                 max_new_tokens: int = 256,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 top_k: int = 50,
                 num_beams: int = 1,
                 do_sample: bool = True,
                 use_cache: bool = True,
                 block_embedding_cache: bool = True,
                 selective_expansion: bool = True,
                 expansion_ratio: float = 0.1):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_beams = num_beams
        self.do_sample = do_sample
        self.use_cache = use_cache
        self.block_embedding_cache = block_embedding_cache
        self.selective_expansion = selective_expansion
        self.expansion_ratio = expansion_ratio


class InferenceMetrics:
    def __init__(self, ttft: float, throughput: float, total_time: float, 
                 num_tokens: int, cache_hit_rate: float):
        self.ttft = ttft  # Time To First Token (ms)
        self.throughput = throughput  # Tokens per second
        self.total_time = total_time  # Total inference time (ms)
        self.num_tokens = num_tokens  # Number of generated tokens
        self.cache_hit_rate = cache_hit_rate  # Block embedding cache hit rate


class RefragInference:
    def __init__(self, model: RefragModel, config: InferenceConfig = None, 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.config = config or InferenceConfig()
        self.device = device
        
        self.block_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def encode_context(self, context_tokens: torch.Tensor, 
                      attention_mask: Optional[torch.Tensor] = None, 
                      use_cache: bool = True) -> torch.Tensor:
        cache_key = None
        
        if use_cache:
            # 生成缓存键
            cache_key = hash(context_tokens.cpu().numpy().tobytes())
            if cache_key in self.block_cache:
                self.cache_hits += 1
                return self.block_cache[cache_key]
            else:
                self.cache_misses += 1
        
        # 编码上下文
        block_embeddings = self.model.encoder(context_tokens, attention_mask)
        projected_embeddings = self.model.projection(block_embeddings)
        
        if use_cache and cache_key is not None:
            self.block_cache[cache_key] = projected_embeddings
        
        return projected_embeddings
    
    def generate(self, query_tokens: torch.Tensor, context_tokens: torch.Tensor, 
                 query_attention_mask: Optional[torch.Tensor] = None, 
                 context_attention_mask: Optional[torch.Tensor] = None,** kwargs) -> Tuple[torch.Tensor, InferenceMetrics]:
        start_time = time.time()
        
        encode_start = time.time()
        block_embeddings = self.encode_context(
            context_tokens, context_attention_mask, use_cache=self.config.block_embedding_cache
        )
        encode_time = time.time() - encode_start
        
        if self.config.selective_expansion and self.model.rl_policy is not None:
            block_embeddings, rl_info = self.model.rl_policy.select_and_expand(
                block_embeddings, query_tokens, self.model, deterministic=True
            )
        
        generate_start = time.time()
        with torch.no_grad():
            generated_ids = self.model.decoder.generate(
                query_tokens=query_tokens,
                block_embeddings=block_embeddings,
                query_attention_mask=query_attention_mask,
                max_new_tokens=kwargs.get('max_new_tokens', self.config.max_new_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                top_p=kwargs.get('top_p', self.config.top_p),
                top_k=kwargs.get('top_k', self.config.top_k),
                num_beams=kwargs.get('num_beams', self.config.num_beams),
                do_sample=kwargs.get('do_sample', self.config.do_sample)
            )
        generate_time = time.time() - generate_start
        
        total_time = time.time() - start_time
        num_tokens = generated_ids.shape[1]
        ttft = (encode_time + generate_start - start_time) * 1000  # ms
        throughput = num_tokens / generate_time if generate_time > 0 else 0
        
        cache_stats = self.get_cache_stats()
        
        metrics = InferenceMetrics(
            ttft=ttft, throughput=throughput, total_time=total_time * 1000,
            num_tokens=num_tokens, cache_hit_rate=cache_stats["hit_rate"]
        )
        
        return generated_ids, metrics
    
    def get_cache_stats(self) -> Dict[str, float]:
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0
        
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total": total,
            "hit_rate": hit_rate
        }
    
    def clear_cache(self):
        self.block_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def process_query(self, query: str, context: str) -> Tuple[str, InferenceMetrics]:
        # 预处理输入
        tokenizer = self.model.decoder.tokenizer
        
        query_tokens = tokenizer(
            query,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt"
        )
        
        # 确保所有张量都在同一设备上
        query_input_ids = query_tokens["input_ids"].to(self.device)
        query_attention_mask = query_tokens["attention_mask"].to(self.device)
        
        # 分块处理上下文
        max_block_size = self.model.block_size * 10
        context_tokens = tokenizer(
            context,
            truncation=True,
            max_length=max_block_size * 10,  # 限制上下文长度
            return_tensors="pt"
        )["input_ids"].squeeze(0)
        
        blocks = []
        for i in range(0, len(context_tokens), max_block_size):
            block = context_tokens[i:i+max_block_size]
            padded_block = torch.nn.functional.pad(
                block, (0, max_block_size - len(block)),
                value=tokenizer.pad_token_id
            )
            blocks.append(padded_block)
        
        if blocks:
            context_tokens = torch.stack(blocks).unsqueeze(0).to(self.device)
        else:
            context_tokens = torch.tensor([[]], device=self.device)
        
        # 生成响应
        generated_ids, metrics = self.generate(
            query_tokens=query_input_ids,
            context_tokens=context_tokens,
            query_attention_mask=query_attention_mask
        )
        
        # 解码响应
        response = tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return response, metrics