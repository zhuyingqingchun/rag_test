#!/usr/bin/env python3
"""
不同信号编码器实现
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional


class SignalEncoder(nn.Module):
    """
    原始信号数据编码器
    将时序信号转换为固定维度的向量表示
    """
    
    def __init__(
        self,
        input_dim: int = 1,           # 输入信号维度
        hidden_dim: int = 256,        # 隐藏层维度
        output_dim: int = 768,        # 输出嵌入维度
        num_layers: int = 4,          # LSTM层数
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_attention = use_attention
        
        # LSTM编码器
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 注意力机制
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim * 2,  # 双向LSTM
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, signal: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            signal: 输入信号 [batch_size, seq_len, input_dim]
            mask: 掩码 [batch_size, seq_len]
        
        Returns:
            信号嵌入 [batch_size, output_dim]
        """
        # LSTM编码
        lstm_out, (hidden, cell) = self.lstm(signal)
        # lstm_out: [batch_size, seq_len, hidden_dim * 2]
        
        if self.use_attention:
            # 应用注意力机制
            if mask is not None:
                # 转换mask为注意力mask
                attn_mask = ~mask.bool()
                attn_output, _ = self.attention(
                    lstm_out, lstm_out, lstm_out,
                    key_padding_mask=attn_mask
                )
            else:
                attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
            
            # 平均池化
            pooled = attn_output.mean(dim=1)
        else:
            # 使用最后一个时间步
            pooled = lstm_out[:, -1, :]
        
        # 投影到输出维度
        output = self.projection(pooled)
        output = self.layer_norm(output)
        
        return output
    
    def encode_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        编码单个信号
        
        Args:
            signal: 信号数组 [seq_len] 或 [seq_len, input_dim]
        
        Returns:
            信号嵌入 [output_dim]
        """
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        
        # 转换为tensor
        signal_tensor = torch.FloatTensor(signal).unsqueeze(0)
        
        # 移动到模型所在设备
        device = next(self.parameters()).device
        signal_tensor = signal_tensor.to(device)
        
        with torch.no_grad():
            embedding = self.forward(signal_tensor)
        
        return embedding.squeeze(0).cpu().numpy()
    
    def encode_batch(self, signals: List[np.ndarray]) -> np.ndarray:
        """
        批量编码信号
        
        Args:
            signals: 信号列表
        
        Returns:
            信号嵌入矩阵 [num_signals, output_dim]
        """
        embeddings = []
        for signal in signals:
            emb = self.encode_signal(signal)
            embeddings.append(emb)
        
        return np.array(embeddings)


class AutoencoderSignalEncoder(nn.Module):
    """
    自编码器信号编码器
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 256,
        output_dim: int = 768,
        seq_len: int = 1000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(seq_len * input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim)
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, seq_len * input_dim)
        )
    
    def forward(self, signal: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            signal: 输入信号 [batch_size, seq_len, input_dim]
            mask: 掩码 [batch_size, seq_len]
        
        Returns:
            嵌入和重构信号
        """
        batch_size, seq_len, input_dim = signal.shape
        
        # 展平信号
        x = signal.reshape(batch_size, -1)
        
        # 编码
        embedding = self.encoder(x)
        
        # 解码
        reconstructed = self.decoder(embedding)
        reconstructed = reconstructed.reshape(batch_size, seq_len, input_dim)
        
        return embedding, reconstructed
    
    def encode_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        编码单个信号
        """
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        
        signal_tensor = torch.FloatTensor(signal).unsqueeze(0)
        
        # 移动到模型所在设备
        device = next(self.parameters()).device
        signal_tensor = signal_tensor.to(device)
        
        with torch.no_grad():
            embedding, _ = self.forward(signal_tensor)
        
        return embedding.squeeze(0).cpu().numpy()
    
    def encode_batch(self, signals: List[np.ndarray]) -> np.ndarray:
        """
        批量编码信号
        """
        embeddings = []
        for signal in signals:
            emb = self.encode_signal(signal)
            embeddings.append(emb)
        
        return np.array(embeddings)


class CNNSignalEncoder(nn.Module):
    """
    CNN信号编码器
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 256,
        output_dim: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # CNN特征提取
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, signal: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            signal: 输入信号 [batch_size, seq_len, input_dim]
            mask: 掩码 [batch_size, seq_len]
        
        Returns:
            信号嵌入 [batch_size, output_dim]
        """
        # 转换为 [batch_size, input_dim, seq_len]
        x = signal.transpose(1, 2)
        
        # CNN特征提取
        features = self.cnn(x)
        features = features.squeeze(2)
        
        # 投影到输出维度
        output = self.projection(features)
        output = self.layer_norm(output)
        
        return output
    
    def encode_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        编码单个信号
        """
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        
        signal_tensor = torch.FloatTensor(signal).unsqueeze(0)
        
        # 移动到模型所在设备
        device = next(self.parameters()).device
        signal_tensor = signal_tensor.to(device)
        
        with torch.no_grad():
            embedding = self.forward(signal_tensor)
        
        return embedding.squeeze(0).cpu().numpy()
    
    def encode_batch(self, signals: List[np.ndarray]) -> np.ndarray:
        """
        批量编码信号
        """
        embeddings = []
        for signal in signals:
            emb = self.encode_signal(signal)
            embeddings.append(emb)
        
        return np.array(embeddings)


class TransformerSignalEncoder(nn.Module):
    """
    Transformer信号编码器
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 256,
        output_dim: int = 768,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 输入嵌入
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码
        self.position_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, signal: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            signal: 输入信号 [batch_size, seq_len, input_dim]
            mask: 掩码 [batch_size, seq_len]
        
        Returns:
            信号嵌入 [batch_size, output_dim]
        """
        batch_size, seq_len, _ = signal.shape
        
        # 输入嵌入
        x = self.input_embedding(signal)
        
        # 添加位置编码
        if seq_len <= 1000:
            x = x + self.position_encoding[:, :seq_len, :]
        else:
            # 对于长序列，重复位置编码
            pos_enc = self.position_encoding.repeat(1, (seq_len // 1000) + 1, 1)
            x = x + pos_enc[:, :seq_len, :]
        
        # Transformer编码
        if mask is not None:
            # 创建注意力掩码
            attn_mask = torch.zeros((seq_len, seq_len), device=signal.device)
            attn_mask[mask.bool()] = -float('inf')
            output = self.transformer(x, mask=attn_mask)
        else:
            output = self.transformer(x)
        
        # 平均池化
        output = output.mean(dim=1)
        
        # 投影到输出维度
        output = self.projection(output)
        output = self.layer_norm(output)
        
        return output
    
    def encode_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        编码单个信号
        """
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        
        signal_tensor = torch.FloatTensor(signal).unsqueeze(0)
        
        # 移动到模型所在设备
        device = next(self.parameters()).device
        signal_tensor = signal_tensor.to(device)
        
        with torch.no_grad():
            embedding = self.forward(signal_tensor)
        
        return embedding.squeeze(0).cpu().numpy()
    
    def encode_batch(self, signals: List[np.ndarray]) -> np.ndarray:
        """
        批量编码信号
        """
        embeddings = []
        for signal in signals:
            emb = self.encode_signal(signal)
            embeddings.append(emb)
        
        return np.array(embeddings)
