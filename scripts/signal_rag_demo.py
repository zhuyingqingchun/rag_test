#!/usr/bin/env python3
"""
信号数据RAG演示脚本
展示如何使用训练好的信号编码器进行信号检索和问答
"""

import argparse
import numpy as np
import torch
import json
import os
import sys
sys.path.append('/mnt/PRO6000_disk/swd/servo_0/refrag/src')

from signal_rag.signal_encoder import SignalEncoder, SignalPreprocessor, SignalRAG


def load_trained_encoder(checkpoint_path: str, **model_kwargs) -> SignalEncoder:
    """加载训练好的编码器"""
    model = SignalEncoder(**model_kwargs)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def generate_signal_database(num_signals: int = 100) -> tuple:
    """生成示例信号数据库"""
    sample_rate = 1000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    signals = []
    metadata = []
    
    # 定义不同的信号类型
    signal_types = [
        {'name': 'normal', 'description': '正常运行信号', 'freq': 10},
        {'name': 'fault_bearing', 'description': '轴承故障信号', 'freq': 15, 'modulation': True},
        {'name': 'fault_gear', 'description': '齿轮故障信号', 'freq': 20, 'harmonics': True},
        {'name': 'fault_imbalance', 'description': '不平衡故障信号', 'freq': 5, 'amplitude_var': True},
        {'name': 'fault_misalignment', 'description': '不对中故障信号', 'freq': 25, 'phase_shift': True}
    ]
    
    for i in range(num_signals):
        signal_type = signal_types[i % len(signal_types)]
        
        # 生成基础信号
        freq = signal_type['freq']
        signal = np.sin(2 * np.pi * freq * t)
        
        # 添加故障特征
        if 'modulation' in signal_type and signal_type['modulation']:
            # 调制特征
            mod_freq = 2
            signal *= (1 + 0.5 * np.sin(2 * np.pi * mod_freq * t))
        
        if 'harmonics' in signal_type and signal_type['harmonics']:
            # 谐波特征
            signal += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
            signal += 0.2 * np.sin(2 * np.pi * freq * 3 * t)
        
        if 'amplitude_var' in signal_type and signal_type['amplitude_var']:
            # 幅度变化
            signal *= (1 + 0.3 * np.sin(2 * np.pi * 0.5 * t))
        
        if 'phase_shift' in signal_type and signal_type['phase_shift']:
            # 相位偏移
            signal = np.sin(2 * np.pi * freq * t + np.pi/4)
        
        # 添加噪声
        signal += 0.1 * np.random.randn(len(t))
        
        signals.append(signal)
        metadata.append({
            'id': i,
            'type': signal_type['name'],
            'description': signal_type['description'],
            'frequency': freq,
            'timestamp': f"2024-01-{i+1:02d} 10:00:00"
        })
    
    return signals, metadata


def evaluate_signal_rag(rag: SignalRAG, test_queries: list, ground_truth: list) -> dict:
    """评估信号RAG系统"""
    correct = 0
    total = len(test_queries)
    
    retrieval_accuracies = []
    
    for query_signal, expected_type in zip(test_queries, ground_truth):
        results = rag.retrieve(query_signal)
        
        if results:
            # 检查top-1结果是否正确
            top_result = results[0]
            retrieved_type = top_result[3]['type']
            
            if retrieved_type == expected_type:
                correct += 1
                retrieval_accuracies.append(1.0)
            else:
                retrieval_accuracies.append(0.0)
    
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'retrieval_accuracy': accuracy,
        'num_correct': correct,
        'num_total': total
    }


def main():
    parser = argparse.ArgumentParser(description="Signal RAG Demo")
    parser.add_argument("--encoder_path", type=str, default=None, help="Path to trained encoder checkpoint")
    parser.add_argument("--num_signals", type=int, default=100, help="Number of signals in database")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to retrieve")
    parser.add_argument("--save_db", type=str, default=None, help="Path to save signal database")
    parser.add_argument("--load_db", type=str, default=None, help="Path to load signal database")
    args = parser.parse_args()
    
    print("="*80)
    print("Signal Data RAG System Demo")
    print("="*80)
    
    # 创建或加载编码器
    if args.encoder_path and os.path.exists(args.encoder_path):
        print(f"Loading trained encoder from {args.encoder_path}...")
        encoder = load_trained_encoder(
            args.encoder_path,
            input_dim=1,
            hidden_dim=256,
            output_dim=768,
            num_layers=4
        )
    else:
        print("Using untrained encoder (random initialization)...")
        encoder = SignalEncoder(
            input_dim=1,
            hidden_dim=256,
            output_dim=768,
            num_layers=4
        )
    
    # 创建预处理器
    preprocessor = SignalPreprocessor(window_size=1000, hop_size=500)
    
    # 创建RAG系统
    rag = SignalRAG(encoder, preprocessor, top_k=args.top_k)
    
    # 加载或创建信号数据库
    if args.load_db and os.path.exists(args.load_db):
        print(f"Loading signal database from {args.load_db}...")
        rag.load_database(args.load_db)
        print(f"Loaded {len(rag.signal_database)} signals")
    else:
        print(f"Generating {args.num_signals} synthetic signals...")
        signals, metadata = generate_signal_database(args.num_signals)
        
        print("Adding signals to RAG database...")
        rag.add_signals(signals, metadata)
        print(f"Added {len(signals)} signals to database")
        
        # 保存数据库
        if args.save_db:
            print(f"Saving database to {args.save_db}...")
            os.makedirs(os.path.dirname(args.save_db), exist_ok=True)
            rag.save_database(args.save_db)
    
    # 生成测试查询
    print("\nGenerating test queries...")
    sample_rate = 1000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    test_queries = []
    ground_truth = []
    
    # 查询1: 轴承故障
    query1 = np.sin(2 * np.pi * 15 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
    query1 += 0.1 * np.random.randn(len(t))
    test_queries.append(query1)
    ground_truth.append('fault_bearing')
    
    # 查询2: 齿轮故障
    query2 = np.sin(2 * np.pi * 20 * t)
    query2 += 0.3 * np.sin(2 * np.pi * 40 * t)
    query2 += 0.2 * np.sin(2 * np.pi * 60 * t)
    query2 += 0.1 * np.random.randn(len(t))
    test_queries.append(query2)
    ground_truth.append('fault_gear')
    
    # 查询3: 不平衡
    query3 = np.sin(2 * np.pi * 5 * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.5 * t))
    query3 += 0.1 * np.random.randn(len(t))
    test_queries.append(query3)
    ground_truth.append('fault_imbalance')
    
    # 查询4: 正常运行
    query4 = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
    test_queries.append(query4)
    ground_truth.append('normal')
    
    # 评估RAG系统
    print("\nEvaluating RAG system...")
    metrics = evaluate_signal_rag(rag, test_queries, ground_truth)
    
    print(f"\nRetrieval Accuracy: {metrics['retrieval_accuracy']:.2%}")
    print(f"Correct: {metrics['num_correct']}/{metrics['num_total']}")
    
    # 展示检索结果
    print("\n" + "="*80)
    print("Retrieval Results")
    print("="*80)
    
    query_names = ['Bearing Fault', 'Gear Fault', 'Imbalance', 'Normal']
    
    for query_signal, expected_type, query_name in zip(test_queries, ground_truth, query_names):
        print(f"\nQuery: {query_name} (Expected: {expected_type})")
        print("-" * 60)
        
        results = rag.retrieve(query_signal)
        
        for rank, (idx, sim, signal, meta) in enumerate(results, 1):
            match = "✓" if meta['type'] == expected_type else "✗"
            print(f"  {rank}. [{match}] Index: {idx}, Similarity: {sim:.4f}")
            print(f"      Type: {meta['type']}, Description: {meta['description']}")
    
    # 交互式查询
    print("\n" + "="*80)
    print("Interactive Mode")
    print("Commands:")
    print("  query <type>  - Query by signal type (normal, bearing, gear, imbalance, misalignment)")
    print("  stats         - Show database statistics")
    print("  exit          - Exit the program")
    print("="*80)
    
    while True:
        command = input("\n> ").strip().lower()
        
        if command == 'exit':
            break
        
        elif command == 'stats':
            print(f"\nDatabase Statistics:")
            print(f"  Total signals: {len(rag.signal_database)}")
            
            # 统计各类信号数量
            type_counts = {}
            for meta in rag.metadata:
                signal_type = meta['type']
                type_counts[signal_type] = type_counts.get(signal_type, 0) + 1
            
            print("  Signal types:")
            for signal_type, count in sorted(type_counts.items()):
                print(f"    {signal_type}: {count}")
        
        elif command.startswith('query '):
            query_type = command.split()[1]
            
            # 生成查询信号
            if query_type == 'normal':
                query_signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
            elif query_type == 'bearing':
                query_signal = np.sin(2 * np.pi * 15 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
                query_signal += 0.1 * np.random.randn(len(t))
            elif query_type == 'gear':
                query_signal = np.sin(2 * np.pi * 20 * t)
                query_signal += 0.3 * np.sin(2 * np.pi * 40 * t)
                query_signal += 0.1 * np.random.randn(len(t))
            elif query_type == 'imbalance':
                query_signal = np.sin(2 * np.pi * 5 * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.5 * t))
                query_signal += 0.1 * np.random.randn(len(t))
            elif query_type == 'misalignment':
                query_signal = np.sin(2 * np.pi * 25 * t + np.pi/4)
                query_signal += 0.1 * np.random.randn(len(t))
            else:
                print(f"Unknown signal type: {query_type}")
                continue
            
            # 检索
            results = rag.retrieve(query_signal)
            
            print(f"\nQuery: {query_type}")
            print("-" * 60)
            
            for rank, (idx, sim, signal, meta) in enumerate(results, 1):
                print(f"  {rank}. Index: {idx}, Similarity: {sim:.4f}")
                print(f"      Type: {meta['type']}, Description: {meta['description']}")
        
        else:
            print("Unknown command. Type 'exit' to quit.")
    
    print("\nGoodbye!")


if __name__ == "__main__":
    main()
