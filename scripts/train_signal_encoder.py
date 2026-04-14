#!/usr/bin/env python3
"""
信号编码器训练脚本
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import sys
sys.path.append('/mnt/PRO6000_disk/swd/servo_0/refrag/src')

from signal_rag.signal_encoder import SignalEncoder, SignalPreprocessor


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


class ContrastiveLoss(nn.Module):
    """对比损失函数"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算对比损失
        
        Args:
            embeddings: 嵌入向量 [batch_size, embedding_dim]
            labels: 标签 [batch_size]
        
        Returns:
            损失值
        """
        batch_size = embeddings.shape[0]
        
        # 归一化嵌入
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # 创建标签掩码
        labels = labels.unsqueeze(0)
        mask = (labels == labels.T).float()
        
        # 移除对角线（自身相似度）
        mask = mask - torch.eye(batch_size, device=mask.device)
        
        # 计算对比损失
        exp_sim = torch.exp(similarity_matrix)
        
        # 正样本对
        pos_sim = (exp_sim * mask).sum(dim=1)
        
        # 负样本对
        neg_sim = exp_sim.sum(dim=1) - torch.diag(exp_sim)
        
        # 损失
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8))
        loss = loss.mean()
        
        return loss


class TripletLoss(nn.Module):
    """三元组损失函数"""
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negative: torch.Tensor) -> torch.Tensor:
        """
        计算三元组损失
        
        Args:
            anchor: 锚点样本 [batch_size, embedding_dim]
            positive: 正样本 [batch_size, embedding_dim]
            negative: 负样本 [batch_size, embedding_dim]
        
        Returns:
            损失值
        """
        # 计算距离
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        
        # 三元组损失
        loss = torch.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()


class SignalEncoderTrainer:
    """信号编码器训练器"""
    
    def __init__(
        self,
        model: SignalEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader),
            eta_min=learning_rate * 0.1
        )
        
        self.contrastive_loss = ContrastiveLoss()
        self.triplet_loss = TripletLoss()
        
        self.train_logs = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (signals, labels) in enumerate(self.train_loader):
            signals = signals.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            embeddings = self.model(signals)
            
            # 计算损失
            loss = self.contrastive_loss(embeddings, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """验证"""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for signals, labels in self.val_loader:
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                
                embeddings = self.model(signals)
                loss = self.contrastive_loss(embeddings, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def train(self, num_epochs: int, output_dir: str):
        """训练模型"""
        os.makedirs(output_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # 训练
            train_metrics = self.train_epoch()
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            
            # 验证
            val_metrics = self.validate()
            if val_metrics:
                print(f"Val Loss: {val_metrics['val_loss']:.4f}")
                
                # 保存最佳模型
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_checkpoint(
                        os.path.join(output_dir, "best_model.pt"),
                        epoch,
                        val_metrics
                    )
                    print("Saved best model!")
            
            # 定期保存
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(
                    os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pt"),
                    epoch,
                    {**train_metrics, **val_metrics}
                )
            
            # 记录日志
            self.train_logs.append({
                'epoch': epoch + 1,
                **train_metrics,
                **val_metrics
            })
        
        # 保存训练日志
        with open(os.path.join(output_dir, "train_logs.json"), 'w') as f:
            json.dump(self.train_logs, f, indent=2)
        
        print("\nTraining completed!")
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        torch.save(checkpoint, path)


def generate_synthetic_data(num_samples: int = 1000, num_classes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成合成信号数据用于测试
    
    Args:
        num_samples: 样本数量
        num_classes: 类别数量
    
    Returns:
        signals, labels
    """
    signals = []
    labels = []
    
    sample_rate = 1000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    for i in range(num_samples):
        class_id = i % num_classes
        
        # 根据类别生成不同频率的信号
        freq = 5 + class_id * 2
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(0.5, 1.5)
        noise_level = np.random.uniform(0.1, 0.3)
        
        signal = amplitude * np.sin(2 * np.pi * freq * t + phase) + noise_level * np.random.randn(len(t))
        
        signals.append(signal)
        labels.append(class_id)
    
    return np.array(signals), np.array(labels)


def main():
    parser = argparse.ArgumentParser(description="Train Signal Encoder")
    parser.add_argument("--data_path", type=str, default=None, help="Path to signal data (.npz file)")
    parser.add_argument("--output_dir", type=str, default="outputs/signal_encoder", help="Output directory")
    parser.add_argument("--input_dim", type=int, default=1, help="Input dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--output_dim", type=int, default=768, help="Output embedding dimension")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of LSTM layers")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--use_synthetic", action="store_true", help="Use synthetic data for testing")
    args = parser.parse_args()
    
    # 加载数据
    if args.use_synthetic or args.data_path is None:
        print("Generating synthetic data...")
        signals, labels = generate_synthetic_data(num_samples=1000, num_classes=10)
    else:
        print(f"Loading data from {args.data_path}...")
        data = np.load(args.data_path)
        signals = data['signals']
        labels = data['labels']
    
    print(f"Loaded {len(signals)} signals")
    
    # 划分训练集和验证集
    num_train = int(0.8 * len(signals))
    train_signals = signals[:num_train]
    train_labels = labels[:num_train]
    val_signals = signals[num_train:]
    val_labels = labels[num_train:]
    
    # 创建预处理器
    preprocessor = SignalPreprocessor(window_size=1000, hop_size=500)
    
    # 创建数据集
    train_dataset = SignalDataset(train_signals, train_labels, preprocessor)
    val_dataset = SignalDataset(val_signals, val_labels, preprocessor)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 创建模型
    model = SignalEncoder(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers
    )
    
    # 创建训练器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    trainer = SignalEncoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        device=device
    )
    
    # 训练
    trainer.train(num_epochs=args.num_epochs, output_dir=args.output_dir)
    
    print(f"\nModel saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
