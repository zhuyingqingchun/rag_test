#!/usr/bin/env python3
"""
信号数据RAG非交互测试脚本
"""

import argparse
import numpy as np
import torch
import os
import sys
sys.path.append('/mnt/PRO6000_disk/swd/servo_0/refrag/src')

from signal_rag.signal_encoder import SignalEncoder, SignalPreprocessor, SignalRAG


def generate_signal_database(num_signals: int = 100) -> tuple:
    """生成示例信号数据库"""
    sample_rate = 1000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    signals = []
    metadata = []
    
    signal_types = [
        {'name': 'normal', 'description': '正常运行信号', 'freq': 10},
        {'name': 'fault_bearing', 'description': '轴承故障信号', 'freq': 15, 'modulation': True},
        {'name': 'fault_gear', 'description': '齿轮故障信号', 'freq': 20, 'harmonics': True},
        {'name': 'fault_imbalance', 'description': '不平衡故障信号', 'freq': 5, 'amplitude_var': True},
        {'name': 'fault_misalignment', 'description': '不对中故障信号', 'freq': 25, 'phase_shift': True}
    ]
    
    for i in range(num_signals):
        signal_type = signal_types[i % len(signal_types)]
        
        freq = signal_type['freq']
        signal = np.sin(2 * np.pi * freq * t)
        
        if 'modulation' in signal_type and signal_type['modulation']:
            mod_freq = 2
            signal *= (1 + 0.5 * np.sin(2 * np.pi * mod_freq * t))
        
        if 'harmonics' in signal_type and signal_type['harmonics']:
            signal += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
            signal += 0.2 * np.sin(2 * np.pi * freq * 3 * t)
        
        if 'amplitude_var' in signal_type and signal_type['amplitude_var']:
            signal *= (1 + 0.3 * np.sin(2 * np.pi * 0.5 * t))
        
        if 'phase_shift' in signal_type and signal_type['phase_shift']:
            signal = np.sin(2 * np.pi * freq * t + np.pi/4)
        
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
    parser = argparse.ArgumentParser(description="Signal RAG Non-Interactive Test")
    parser.add_argument("--num_signals", type=int, default=100, help="Number of signals in database")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top results to retrieve")
    parser.add_argument("--save_db", type=str, default=None, help="Path to save signal database")
    args = parser.parse_args()
    
    print("="*80)
    print("Signal Data RAG System Non-Interactive Test")
    print("="*80)
    
    encoder = SignalEncoder(
        input_dim=1,
        hidden_dim=256,
        output_dim=768,
        num_layers=4
    )
    
    preprocessor = SignalPreprocessor(window_size=1000, hop_size=500)
    rag = SignalRAG(encoder, preprocessor, top_k=args.top_k)
    
    print(f"Generating {args.num_signals} synthetic signals...")
    signals, metadata = generate_signal_database(args.num_signals)
    
    print("Adding signals to RAG database...")
    rag.add_signals(signals, metadata)
    print(f"Added {len(signals)} signals to database")
    
    if args.save_db:
        print(f"Saving database to {args.save_db}...")
        os.makedirs(os.path.dirname(args.save_db) if os.path.dirname(args.save_db) else '.', exist_ok=True)
        rag.save_database(args.save_db)
    
    print("\nGenerating test queries...")
    sample_rate = 1000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    test_queries = []
    ground_truth = []
    
    query1 = np.sin(2 * np.pi * 15 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
    query1 += 0.1 * np.random.randn(len(t))
    test_queries.append(query1)
    ground_truth.append('fault_bearing')
    
    query2 = np.sin(2 * np.pi * 20 * t)
    query2 += 0.3 * np.sin(2 * np.pi * 40 * t)
    query2 += 0.2 * np.sin(2 * np.pi * 60 * t)
    query2 += 0.1 * np.random.randn(len(t))
    test_queries.append(query2)
    ground_truth.append('fault_gear')
    
    query3 = np.sin(2 * np.pi * 5 * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.5 * t))
    query3 += 0.1 * np.random.randn(len(t))
    test_queries.append(query3)
    ground_truth.append('fault_imbalance')
    
    query4 = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
    test_queries.append(query4)
    ground_truth.append('normal')
    
    print("\nEvaluating RAG system...")
    metrics = evaluate_signal_rag(rag, test_queries, ground_truth)
    
    print(f"\nRetrieval Accuracy: {metrics['retrieval_accuracy']:.2%}")
    print(f"Correct: {metrics['num_correct']}/{metrics['num_total']}")
    
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
    
    print("\n" + "="*80)
    print("Database Statistics")
    print("="*80)
    print(f"Total signals: {len(rag.signal_database)}")
    
    type_counts = {}
    for meta in rag.metadata:
        signal_type = meta['type']
        type_counts[signal_type] = type_counts.get(signal_type, 0) + 1
    
    print("Signal types:")
    for signal_type, count in sorted(type_counts.items()):
        print(f"  {signal_type}: {count}")
    
    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
