"""
信号数据编码器
用于将信号数据编码为向量表示
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
import sys
sys.path.append('/mnt/PRO6000_disk/swd/servo_0')


class SignalEncoder(nn.Module):
    """
    信号数据编码器
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


class SignalPreprocessor:
    """信号预处理器"""
    
    def __init__(
        self,
        sample_rate: int = 1000,
        window_size: int = 1024,
        hop_size: int = 512,
        normalize: bool = True
    ):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.normalize = normalize
    
    def normalize_signal(self, signal: np.ndarray) -> np.ndarray:
        """标准化信号"""
        mean = np.mean(signal)
        std = np.std(signal)
        if std > 0:
            return (signal - mean) / std
        return signal
    
    def segment_signal(self, signal: np.ndarray) -> List[np.ndarray]:
        """
        将信号分割为多个窗口
        
        Args:
            signal: 输入信号 [seq_len]
        
        Returns:
            窗口列表
        """
        if self.normalize:
            signal = self.normalize_signal(signal)
        
        segments = []
        for start in range(0, len(signal) - self.window_size + 1, self.hop_size):
            segment = signal[start:start + self.window_size]
            segments.append(segment)
        
        return segments
    
    def extract_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        提取信号特征
        
        Args:
            signal: 输入信号
        
        Returns:
            特征字典
        """
        features = {
            'mean': float(np.mean(signal)),
            'std': float(np.std(signal)),
            'max': float(np.max(signal)),
            'min': float(np.min(signal)),
            'rms': float(np.sqrt(np.mean(signal ** 2))),
            'peak_to_peak': float(np.max(signal) - np.min(signal)),
            'zero_crossing_rate': float(np.sum(np.diff(np.sign(signal)) != 0) / len(signal))
        }
        
        # 频域特征
        fft = np.fft.fft(signal)
        magnitude = np.abs(fft)
        features['dominant_freq'] = float(np.argmax(magnitude[:len(magnitude)//2]))
        features['spectral_energy'] = float(np.sum(magnitude ** 2))
        
        return features


class SignalRAG:
    """信号数据RAG系统"""
    
    def __init__(
        self,
        encoder: SignalEncoder,
        preprocessor: SignalPreprocessor,
        top_k: int = 5
    ):
        self.encoder = encoder
        self.encoder.eval()
        self.preprocessor = preprocessor
        self.top_k = top_k
        
        # 存储信号数据库
        self.signal_database = []
        self.embeddings = None
        self.metadata = []
    
    def add_signals(self, signals: List[np.ndarray], metadata: List[Dict] = None):
        """
        添加信号到数据库
        
        Args:
            signals: 信号列表
            metadata: 元数据列表
        """
        # 编码所有信号
        embeddings = self.encoder.encode_batch(signals)
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        self.signal_database.extend(signals)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in signals])
    
    def retrieve(self, query_signal: np.ndarray) -> List[Tuple[int, float, np.ndarray, Dict]]:
        """
        检索相似信号
        
        Args:
            query_signal: 查询信号
        
        Returns:
            检索结果列表 [(index, similarity, signal, metadata), ...]
        """
        if self.embeddings is None or len(self.signal_database) == 0:
            return []
        
        # 编码查询信号
        query_embedding = self.encoder.encode_signal(query_signal)
        
        # 计算相似度
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # 获取top-k
        top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((
                int(idx),
                float(similarities[idx]),
                self.signal_database[idx],
                self.metadata[idx]
            ))
        
        return results
    
    def save_database(self, path: str):
        """保存信号数据库"""
        data = {
            'embeddings': self.embeddings,
            'signals': self.signal_database,
            'metadata': self.metadata
        }
        np.savez(path, **data)
    
    def load_database(self, path: str):
        """加载信号数据库"""
        data = np.load(path, allow_pickle=True)
        self.embeddings = data['embeddings']
        self.signal_database = list(data['signals'])
        self.metadata = list(data['metadata'])


if __name__ == "__main__":
    # 测试信号编码器
    print("Testing Signal Encoder...")
    
    # 创建示例信号
    sample_rate = 1000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 生成测试信号
    signal1 = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(len(t))
    signal2 = np.cos(2 * np.pi * 20 * t) + 0.3 * np.random.randn(len(t))
    signal3 = np.sin(2 * np.pi * 5 * t) * np.exp(-t) + 0.2 * np.random.randn(len(t))
    
    # 创建编码器
    encoder = SignalEncoder(input_dim=1, output_dim=768)
    
    # 编码信号
    emb1 = encoder.encode_signal(signal1)
    emb2 = encoder.encode_signal(signal2)
    emb3 = encoder.encode_signal(signal3)
    
    print(f"Signal 1 embedding shape: {emb1.shape}")
    print(f"Signal 2 embedding shape: {emb2.shape}")
    print(f"Signal 3 embedding shape: {emb3.shape}")
    
    # 计算相似度
    sim12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    sim13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
    
    print(f"\nSimilarity between signal 1 and 2: {sim12:.4f}")
    print(f"Similarity between signal 1 and 3: {sim13:.4f}")
    
    # 测试RAG系统
    print("\nTesting Signal RAG...")
    preprocessor = SignalPreprocessor(window_size=1000, hop_size=500)
    rag = SignalRAG(encoder, preprocessor, top_k=3)
    
    # 添加信号到数据库
    rag.add_signals([signal1, signal2, signal3], [
        {'label': 'sine_10hz', 'description': '10Hz sine wave'},
        {'label': 'cosine_20hz', 'description': '20Hz cosine wave'},
        {'label': 'damped_sine', 'description': 'Damped sine wave'}
    ])
    
    # 检索相似信号
    query_signal = np.sin(2 * np.pi * 10 * t) + 0.4 * np.random.randn(len(t))
    results = rag.retrieve(query_signal)
    
    print("\nRetrieval results:")
    for idx, sim, signal, meta in results:
        print(f"  Index: {idx}, Similarity: {sim:.4f}, Label: {meta.get('label', 'unknown')}")
    
    print("\nSignal RAG test completed!")
