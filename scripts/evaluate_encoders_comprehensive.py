#!/usr/bin/env python3
"""
信号编码器全面评估脚本
测试不同编码器在不同数据复杂度下的性能
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List, Optional
import os
import sys
import time
sys.path.append('/mnt/PRO6000_disk/swd/servo_0/refrag/src')

from signal_rag.multiple_encoders import (
    SignalEncoder, AutoencoderSignalEncoder, 
    CNNSignalEncoder, TransformerSignalEncoder
)
from signal_rag.signal_encoder import SignalPreprocessor, SignalRAG


class SignalDataset(Dataset):
    """信号数据集"""
    
    def __init__(self, signals: np.ndarray, labels: np.ndarray = None, 
                 preprocessor: SignalPreprocessor = None):
        self.signals = signals
        self.labels = labels
        self.preprocessor = preprocessor
    
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        
        # 预处理
        if self.preprocessor:
            segments = self.preprocessor.segment_signal(signal)
            if len(segments) > 0:
                signal = segments[0]  # 使用第一个段
        
        # 转换为tensor
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        
        signal_tensor = torch.FloatTensor(signal)
        
        if self.labels is not None:
            label = torch.LongTensor([self.labels[idx]])[0]
            return signal_tensor, label
        
        return signal_tensor


def generate_synthetic_data(
    num_samples: int = 1000, 
    num_classes: int = 10, 
    complexity: str = 'low'  # low, medium, high
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成不同复杂度的合成信号数据
    """
    signals = []
    labels = []
    
    sample_rate = 1000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    for i in range(num_samples):
        class_id = i % num_classes
        
        # 根据复杂度生成不同的信号
        if complexity == 'low':
            # 简单正弦波
            freq = 5 + class_id * 2
            signal = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
        
        elif complexity == 'medium':
            # 带谐波的信号
            freq = 5 + class_id * 2
            signal = np.sin(2 * np.pi * freq * t)
            signal += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
            signal += 0.2 * np.sin(2 * np.pi * freq * 3 * t)
            signal += 0.1 * np.random.randn(len(t))
        
        else:  # high
            # 复杂调制信号
            freq = 5 + class_id * 2
            mod_freq = 2 + class_id
            signal = np.sin(2 * np.pi * freq * t) * (1 + 0.5 * np.sin(2 * np.pi * mod_freq * t))
            signal += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
            signal += 0.2 * np.sin(2 * np.pi * freq * 3 * t)
            signal += 0.15 * np.random.randn(len(t))
        
        signals.append(signal)
        labels.append(class_id)
    
    return np.array(signals), np.array(labels)


def evaluate_encoder(
    encoder, 
    test_signals, 
    test_labels, 
    top_k=5, 
    device='cuda'
) -> Dict[str, float]:
    """
    评估编码器的检索性能
    """
    preprocessor = SignalPreprocessor(window_size=1000, hop_size=500)
    rag = SignalRAG(encoder, preprocessor, top_k=top_k)
    
    # 添加信号到数据库
    metadata = [{'type': f'class_{label}'} for label in test_labels]
    
    # 记录添加时间
    start_time = time.time()
    rag.add_signals(test_signals, metadata)
    add_time = time.time() - start_time
    
    # 评估检索性能
    correct = 0
    total = len(test_signals)
    retrieval_times = []
    
    for i, query_signal in enumerate(test_signals):
        expected_label = f'class_{test_labels[i]}'
        
        # 记录检索时间
        start_time = time.time()
        results = rag.retrieve(query_signal)
        retrieval_time = time.time() - start_time
        retrieval_times.append(retrieval_time)
        
        if results:
            top_result = results[0]
            retrieved_label = top_result[3]['type']
            if retrieved_label == expected_label:
                correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    avg_retrieval_time = np.mean(retrieval_times) if retrieval_times else 0.0
    
    return {
        'accuracy': accuracy,
        'add_time': add_time,
        'avg_retrieval_time': avg_retrieval_time
    }


def load_encoder(encoder_name: str, checkpoint_path: str, device: str) -> nn.Module:
    """
    加载编码器模型
    """
    if encoder_name == 'lstm':
        model = SignalEncoder(
            input_dim=1,
            hidden_dim=256,
            output_dim=768
        )
    elif encoder_name == 'autoencoder':
        model = AutoencoderSignalEncoder(
            input_dim=1,
            hidden_dim=256,
            output_dim=768,
            seq_len=1000
        )
    elif encoder_name == 'cnn':
        model = CNNSignalEncoder(
            input_dim=1,
            hidden_dim=256,
            output_dim=768
        )
    elif encoder_name == 'transformer':
        model = TransformerSignalEncoder(
            input_dim=1,
            hidden_dim=256,
            output_dim=768
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_name}")
    
    # 加载权重
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Signal Encoder Evaluation")
    parser.add_argument("--output_dir", type=str, default="outputs/encoder_evaluation", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of samples per complexity level")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k retrieval")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 定义编码器和检查点路径
    encoders = {
        'cnn': 'outputs/cnn_encoder/best_model.pt',
        'lstm': None,
        'autoencoder': None,
        'transformer': None
    }
    
    # 定义复杂度级别
    complexities = ['low', 'medium', 'high']
    
    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 评估结果
    results = {}
    
    for encoder_name, checkpoint_path in encoders.items():
        print(f"\n{'='*80}")
        print(f"Evaluating {encoder_name} encoder...")
        print(f"{'='*80}")
        
        # 加载编码器
        encoder = load_encoder(encoder_name, checkpoint_path, device)
        
        # 评估不同复杂度的数据
        encoder_results = {}
        
        for complexity in complexities:
            print(f"\nEvaluating on {complexity} complexity data...")
            
            # 生成数据
            signals, labels = generate_synthetic_data(
                num_samples=args.num_samples,
                num_classes=args.num_classes,
                complexity=complexity
            )
            
            # 评估
            eval_result = evaluate_encoder(
                encoder, 
                signals, 
                labels, 
                top_k=args.top_k,
                device=device
            )
            
            encoder_results[complexity] = eval_result
            print(f"  Accuracy: {eval_result['accuracy']:.2%}")
            print(f"  Add time: {eval_result['add_time']:.4f}s")
            print(f"  Avg retrieval time: {eval_result['avg_retrieval_time']:.4f}s")
        
        results[encoder_name] = encoder_results
    
    # 保存评估结果
    with open(os.path.join(args.output_dir, "evaluation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 打印总结
    print(f"\n{'='*80}")
    print("Evaluation Summary")
    print(f"{'='*80}")
    
    for encoder_name, encoder_results in results.items():
        print(f"\n{encoder_name}:")
        for complexity, result in encoder_results.items():
            print(f"  {complexity}: Accuracy = {result['accuracy']:.2%}, Time = {result['avg_retrieval_time']:.4f}s")
    
    # 找出最佳编码器
    best_encoder = None
    best_accuracy = 0
    
    for encoder_name, encoder_results in results.items():
        avg_accuracy = np.mean([r['accuracy'] for r in encoder_results.values()])
        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_encoder = encoder_name
    
    print(f"\nBest encoder: {best_encoder} with average accuracy {best_accuracy:.2%}")
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
