#!/usr/bin/env python3
"""
设备处理测试脚本
"""

import numpy as np
import torch
import sys
sys.path.append('/mnt/PRO6000_disk/swd/servo_0/refrag/src')

from signal_rag.multiple_encoders import SignalEncoder

# 测试 LSTM 编码器的设备处理
def test_lstm_encoder():
    print("Testing LSTM encoder device handling...")
    
    # 创建模型
    model = SignalEncoder(
        input_dim=1,
        hidden_dim=256,
        output_dim=768
    )
    
    # 移动到 CUDA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    print(f"Model device: {next(model.parameters()).device}")
    
    # 创建测试信号
    sample_rate = 1000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
    
    # 测试 encode_signal 方法
    print("\nTesting encode_signal...")
    embedding = model.encode_signal(signal)
    print(f"Embedding shape: {embedding.shape}")
    print("encode_signal test passed!")
    
    # 测试 encode_batch 方法
    print("\nTesting encode_batch...")
    signals = [signal, signal]
    embeddings = model.encode_batch(signals)
    print(f"Batch embeddings shape: {embeddings.shape}")
    print("encode_batch test passed!")

if __name__ == "__main__":
    test_lstm_encoder()
