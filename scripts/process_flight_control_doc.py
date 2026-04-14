#!/usr/bin/env python3
"""
使用RAG系统处理飞控文档的脚本
"""

import argparse
import torch
import sys
import os
# 确保src目录在Python路径中
sys.path.insert(0, os.path.abspath('/mnt/PRO6000_disk/swd/servo_0/refrag/src'))
from models import RefragModel
from inference import RefragInference, InferenceConfig


def load_document(doc_path):
    """加载文档内容"""
    with open(doc_path, 'r', encoding='utf-8') as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description="Process Flight Control Document with RAG")
    parser.add_argument("--doc_path", type=str, default="data/flight_control_doc.md")
    parser.add_argument("--encoder_model", type=str, default="distilroberta-base")
    parser.add_argument("--decoder_model", type=str, default="gpt2")
    parser.add_argument("--block_size", type=int, default=32)
    args = parser.parse_args()
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 加载文档
    print(f"Loading document: {args.doc_path}")
    document = load_document(args.doc_path)
    print(f"Document length: {len(document)} characters")
    
    # 创建模型
    print("Creating RAG model...")
    model = RefragModel(
        encoder_name=args.encoder_model,
        decoder_name=args.decoder_model,
        block_size=args.block_size,
        use_rl_policy=False,  # 禁用RL策略以避免兼容性问题
        device=device
    )
    model.eval()
    
    # 创建推理器
    inference_config = InferenceConfig(
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        selective_expansion=True,
        expansion_ratio=0.2
    )
    
    inference = RefragInference(
        model=model,
        config=inference_config,
        device=device
    )
    
    # 测试查询
    test_queries = [
        "飞控系统由哪些部分组成？",
        "飞控系统的工作原理是什么？",
        "飞控系统有哪些飞行模式？",
        "飞控系统的关键技术有哪些？",
        "飞控系统在无人机中有哪些应用？"
    ]
    
    print("\n" + "="*80)
    print("Testing RAG system with flight control document")
    print("="*80)
    
    for i, query in enumerate(test_queries):
        print(f"\n{'='*60}")
        print(f"Query {i+1}: {query}")
        print(f"{'='*60}")
        
        response, metrics = inference.process_query(query, document)
        
        print("Response:")
        print(response)
        print("\nMetrics:")
        print(f"  TTFT: {metrics.ttft:.2f} ms")
        print(f"  Throughput: {metrics.throughput:.2f} tokens/s")
        print(f"  Total time: {metrics.total_time:.2f} ms")
        print(f"  Cache hit rate: {metrics.cache_hit_rate:.2f}")
    
    # 交互式查询
    print("\n" + "="*80)
    print("Interactive mode. Type 'exit' to quit.")
    print("="*80)
    
    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            break
        
        response, metrics = inference.process_query(query, document)
        
        print("\nResponse:")
        print(response)
        print("\nMetrics:")
        print(f"  TTFT: {metrics.ttft:.2f} ms")
        print(f"  Throughput: {metrics.throughput:.2f} tokens/s")
        print(f"  Total time: {metrics.total_time:.2f} ms")
        print(f"  Cache hit rate: {metrics.cache_hit_rate:.2f}")


if __name__ == "__main__":
    main()
