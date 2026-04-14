#!/usr/bin/env python3
"""
使用预训练模型的推理脚本
"""

import argparse
import yaml
import torch
from transformers import AutoTokenizer
import os

import sys
sys.path.append('/mnt/PRO6000_disk/swd/servo_0/refrag/src')

from models import RefragModel
from inference import RefragInference, InferenceConfig


def main():
    parser = argparse.ArgumentParser(description="Use Pre-trained REFRAG Model")
    parser.add_argument("--config", type=str, default="configs/pretrained_config.yaml")
    parser.add_argument("--encoder_path", type=str, default=None, 
                       help="Path to pre-trained encoder model")
    parser.add_argument("--decoder_path", type=str, default=None, 
                       help="Path to pre-trained decoder model")
    parser.add_argument("--query", type=str, required=True, 
                       help="Query text")
    parser.add_argument("--context", type=str, required=True, 
                       help="Context text")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 加载tokenizer
    decoder_model = args.decoder_path or config['model']['decoder_name']
    tokenizer = AutoTokenizer.from_pretrained(decoder_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建模型
    encoder_model = args.encoder_path or config['model']['encoder_name']
    
    model = RefragModel(
        encoder_name=encoder_model,
        decoder_name=decoder_model,
        block_size=config['model']['block_size'],
        use_rl_policy=config['model']['use_rl_policy'],
        encoder_config=config['model']['encoder_config'],
        projection_config=config['model']['projection_config'],
        decoder_config=config['model']['decoder_config'],
        rl_config=config['model']['rl_config']
    )
    
    # 移动到设备
    model.to(device)
    model.eval()
    
    # 创建推理配置
    inference_config = InferenceConfig(
        max_new_tokens=config['inference']['max_new_tokens'],
        temperature=config['inference']['temperature'],
        top_p=config['inference']['top_p'],
        top_k=config['inference']['top_k'],
        num_beams=config['inference']['num_beams'],
        do_sample=config['inference']['do_sample'],
        use_cache=config['inference']['use_cache'],
        block_embedding_cache=config['inference']['block_embedding_cache'],
        selective_expansion=config['inference']['selective_expansion'],
        expansion_ratio=config['inference']['expansion_ratio']
    )
    
    # 创建推理器
    inference = RefragInference(
        model=model,
        config=inference_config,
        device=device
    )
    
    # 处理输入
    query_tokens = tokenizer(
        args.query,
        padding='max_length',
        max_length=config['data']['rag']['max_query_length'],
        return_tensors='pt'
    ).to(device)
    
    # 分块处理上下文
    context_chunks = []
    max_chunk_size = config['model']['block_size'] * 10  # 每块10个block
    
    context_tokens = tokenizer(
        args.context,
        truncation=False,
        return_tensors='pt'
    )['input_ids'].squeeze(0)
    
    for i in range(0, len(context_tokens), max_chunk_size):
        chunk = context_tokens[i:i+max_chunk_size]
        padded_chunk = torch.nn.functional.pad(
            chunk, 
            (0, max_chunk_size - len(chunk)),
            value=tokenizer.pad_token_id
        )
        context_chunks.append(padded_chunk)
    
    if context_chunks:
        context_tokens = torch.stack(context_chunks).to(device)
    else:
        context_tokens = torch.tensor([[]], device=device)
    
    # 生成响应
    print("\nGenerating response...")
    generated_ids, metrics = inference.generate(
        query_tokens=query_tokens['input_ids'],
        context_tokens=context_tokens,
        query_attention_mask=query_tokens['attention_mask'],
        max_new_tokens=config['inference']['max_new_tokens']
    )
    
    # 解码响应
    response = tokenizer.decode(
        generated_ids[0], 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )
    
    # 打印结果
    print("\n" + "="*80)
    print("Query:")
    print(args.query)
    print("\nContext:")
    print(args.context[:200] + "..." if len(args.context) > 200 else args.context)
    print("\nResponse:")
    print(response)
    print("\nMetrics:")
    print(f"  TTFT: {metrics.ttft:.2f} ms")
    print(f"  Throughput: {metrics.throughput:.2f} tokens/s")
    print(f"  Total time: {metrics.total_time:.2f} ms")
    print(f"  Cache hit rate: {metrics.cache_hit_rate:.2f}")
    print("="*80)


if __name__ == "__main__":
    main()
