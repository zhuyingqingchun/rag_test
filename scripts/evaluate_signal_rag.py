#!/usr/bin/env python3
"""
信号数据RAG评估脚本
"""

import argparse
import numpy as np
import torch
import json
import os
import sys
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass
sys.path.append('/mnt/PRO6000_disk/swd/servo_0/refrag/src')

from signal_rag.signal_encoder import SignalEncoder, SignalRAG


@dataclass
class SignalRAGMetrics:
    """信号RAG评估指标"""
    retrieval_accuracy: float
    retrieval_precision: float
    retrieval_recall: float
    mrr: float
    avg_retrieval_time: float
    embedding_quality: float


class SignalRAGEvaluator:
    """信号RAG评估器"""
    
    def __init__(self, rag: SignalRAG):
        self.rag = rag
    
    def evaluate_retrieval(
        self,
        query_signals: List[np.ndarray],
        ground_truth_indices: List[int],
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        评估检索性能
        
        Args:
            query_signals: 查询信号列表
            ground_truth_indices: 真实相关信号索引列表
            top_k: 检索数量
        
        Returns:
            检索指标字典
        """
        correct = 0
        total_precision = 0.0
        total_recall = 0.0
        total_mrr = 0.0
        retrieval_times = []
        
        for query_signal, gt_idx in zip(query_signals, ground_truth_indices):
            # 记录检索时间
            start_time = time.time()
            results = self.rag.retrieve(query_signal)
            retrieval_time = (time.time() - start_time) * 1000  # ms
            retrieval_times.append(retrieval_time)
            
            if not results:
                continue
            
            # 获取检索到的索引
            retrieved_indices = [r[0] for r in results]
            
            # 检查是否正确检索到
            if gt_idx in retrieved_indices:
                correct += 1
                
                # 计算MRR
                rank = retrieved_indices.index(gt_idx) + 1
                total_mrr += 1.0 / rank
            
            # 计算精确率和召回率（假设只有一个相关文档）
            precision = 1.0 / len(retrieved_indices) if gt_idx in retrieved_indices else 0.0
            recall = 1.0 if gt_idx in retrieved_indices else 0.0
            
            total_precision += precision
            total_recall += recall
        
        num_queries = len(query_signals)
        
        return {
            'retrieval_accuracy': correct / num_queries if num_queries > 0 else 0.0,
            'retrieval_precision': total_precision / num_queries if num_queries > 0 else 0.0,
            'retrieval_recall': total_recall / num_queries if num_queries > 0 else 0.0,
            'mrr': total_mrr / num_queries if num_queries > 0 else 0.0,
            'avg_retrieval_time': np.mean(retrieval_times) if retrieval_times else 0.0
        }
    
    def evaluate_embedding_quality(
        self,
        signals: List[np.ndarray],
        labels: List[int]
    ) -> float:
        """
        评估嵌入质量（使用类内距离和类间距离）
        
        Args:
            signals: 信号列表
            labels: 标签列表
        
        Returns:
            嵌入质量分数
        """
        # 编码所有信号
        embeddings = self.rag.encoder.encode_batch(signals)
        
        # 归一化
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # 计算类内距离和类间距离
        unique_labels = np.unique(labels)
        
        intra_class_distances = []
        inter_class_distances = []
        
        for label in unique_labels:
            # 获取同类样本
            mask = np.array(labels) == label
            class_embeddings = embeddings[mask]
            
            if len(class_embeddings) < 2:
                continue
            
            # 计算类内平均距离
            intra_dist = 0.0
            count = 0
            for i in range(len(class_embeddings)):
                for j in range(i + 1, len(class_embeddings)):
                    dist = np.linalg.norm(class_embeddings[i] - class_embeddings[j])
                    intra_dist += dist
                    count += 1
            
            if count > 0:
                intra_class_distances.append(intra_dist / count)
            
            # 计算类间平均距离
            other_mask = ~mask
            other_embeddings = embeddings[other_mask]
            
            if len(other_embeddings) > 0:
                inter_dist = 0.0
                count = 0
                for emb1 in class_embeddings:
                    for emb2 in other_embeddings:
                        dist = np.linalg.norm(emb1 - emb2)
                        inter_dist += dist
                        count += 1
                
                if count > 0:
                    inter_class_distances.append(inter_dist / count)
        
        # 嵌入质量 = 类间距离 / (类内距离 + epsilon)
        avg_intra = np.mean(intra_class_distances) if intra_class_distances else 1.0
        avg_inter = np.mean(inter_class_distances) if inter_class_distances else 1.0
        
        quality = avg_inter / (avg_intra + 1e-8)
        
        return quality
    
    def evaluate(
        self,
        test_data: Dict
    ) -> SignalRAGMetrics:
        """
        完整评估
        
        Args:
            test_data: 测试数据字典，包含:
                - query_signals: 查询信号列表
                - ground_truth_indices: 真实索引列表
                - test_signals: 测试信号列表（用于评估嵌入质量）
                - test_labels: 测试标签列表
        
        Returns:
            评估指标
        """
        print("Evaluating retrieval performance...")
        retrieval_metrics = self.evaluate_retrieval(
            test_data['query_signals'],
            test_data['ground_truth_indices']
        )
        
        print("Evaluating embedding quality...")
        embedding_quality = self.evaluate_embedding_quality(
            test_data['test_signals'],
            test_data['test_labels']
        )
        
        return SignalRAGMetrics(
            retrieval_accuracy=retrieval_metrics['retrieval_accuracy'],
            retrieval_precision=retrieval_metrics['retrieval_precision'],
            retrieval_recall=retrieval_metrics['retrieval_recall'],
            mrr=retrieval_metrics['mrr'],
            avg_retrieval_time=retrieval_metrics['avg_retrieval_time'],
            embedding_quality=embedding_quality
        )
    
    def generate_report(self, metrics: SignalRAGMetrics, output_path: str):
        """生成评估报告"""
        report = {
            'retrieval_metrics': {
                'accuracy': round(metrics.retrieval_accuracy, 4),
                'precision': round(metrics.retrieval_precision, 4),
                'recall': round(metrics.retrieval_recall, 4),
                'mrr': round(metrics.mrr, 4)
            },
            'efficiency_metrics': {
                'avg_retrieval_time_ms': round(metrics.avg_retrieval_time, 2)
            },
            'embedding_quality': round(metrics.embedding_quality, 4),
            'overall_score': round(
                (metrics.retrieval_accuracy * 0.3 +
                 metrics.retrieval_precision * 0.2 +
                 metrics.retrieval_recall * 0.2 +
                 metrics.mrr * 0.15 +
                 min(metrics.embedding_quality / 10, 1.0) * 0.15),
                4
            )
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\nEvaluation report saved to: {output_path}")
        print("\n" + "="*60)
        print("Signal RAG Evaluation Results")
        print("="*60)
        print(f"Retrieval Accuracy:     {metrics.retrieval_accuracy:.4f}")
        print(f"Retrieval Precision:    {metrics.retrieval_precision:.4f}")
        print(f"Retrieval Recall:       {metrics.retrieval_recall:.4f}")
        print(f"MRR:                    {metrics.mrr:.4f}")
        print(f"Avg Retrieval Time:     {metrics.avg_retrieval_time:.2f} ms")
        print(f"Embedding Quality:      {metrics.embedding_quality:.4f}")
        print(f"Overall Score:          {report['overall_score']:.4f}")
        print("="*60)


def generate_test_data(
    num_database_signals: int = 100,
    num_query_signals: int = 20,
    num_classes: int = 5
) -> Dict:
    """生成测试数据"""
    sample_rate = 1000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # 生成数据库信号
    database_signals = []
    database_labels = []
    
    for i in range(num_database_signals):
        label = i % num_classes
        freq = 5 + label * 5
        signal = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
        database_signals.append(signal)
        database_labels.append(label)
    
    # 生成查询信号（从数据库中采样并添加噪声）
    query_indices = np.random.choice(num_database_signals, num_query_signals, replace=False)
    query_signals = []
    ground_truth_indices = []
    
    for idx in query_indices:
        base_signal = database_signals[idx].copy()
        # 添加噪声
        noisy_signal = base_signal + 0.2 * np.random.randn(len(base_signal))
        query_signals.append(noisy_signal)
        ground_truth_indices.append(int(idx))
    
    return {
        'database_signals': database_signals,
        'database_labels': database_labels,
        'query_signals': query_signals,
        'ground_truth_indices': ground_truth_indices,
        'test_signals': database_signals,
        'test_labels': database_labels
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Signal RAG")
    parser.add_argument("--encoder_path", type=str, required=True, help="Path to trained encoder")
    parser.add_argument("--database_path", type=str, default=None, help="Path to signal database")
    parser.add_argument("--output", type=str, default="signal_rag_evaluation.json", help="Output report path")
    parser.add_argument("--num_database_signals", type=int, default=100, help="Number of database signals")
    parser.add_argument("--num_query_signals", type=int, default=20, help="Number of query signals")
    args = parser.parse_args()
    
    print("="*60)
    print("Signal RAG Evaluation")
    print("="*60)
    
    # 加载编码器
    print(f"Loading encoder from {args.encoder_path}...")
    encoder = SignalEncoder(input_dim=1, hidden_dim=256, output_dim=768, num_layers=4)
    checkpoint = torch.load(args.encoder_path, map_location='cpu')
    encoder.load_state_dict(checkpoint['model_state_dict'])
    encoder.eval()
    
    # 创建RAG系统
    from signal_rag.signal_encoder import SignalPreprocessor
    preprocessor = SignalPreprocessor(window_size=1000, hop_size=500)
    rag = SignalRAG(encoder, preprocessor, top_k=5)
    
    # 加载或生成数据
    if args.database_path and os.path.exists(args.database_path):
        print(f"Loading database from {args.database_path}...")
        rag.load_database(args.database_path)
        
        # 生成测试查询
        test_data = generate_test_data(
            num_database_signals=len(rag.signal_database),
            num_query_signals=args.num_query_signals,
            num_classes=5
        )
        test_data['database_signals'] = rag.signal_database
        test_data['database_labels'] = [meta.get('label', 0) for meta in rag.metadata]
        test_data['test_signals'] = rag.signal_database
        test_data['test_labels'] = [meta.get('label', 0) for meta in rag.metadata]
    else:
        print("Generating synthetic test data...")
        test_data = generate_test_data(
            num_database_signals=args.num_database_signals,
            num_query_signals=args.num_query_signals,
            num_classes=5
        )
        
        # 添加到RAG数据库
        print("Adding signals to RAG database...")
        metadata = [{'label': label} for label in test_data['database_labels']]
        rag.add_signals(test_data['database_signals'], metadata)
    
    print(f"Database size: {len(rag.signal_database)} signals")
    print(f"Number of queries: {len(test_data['query_signals'])}")
    
    # 创建评估器
    evaluator = SignalRAGEvaluator(rag)
    
    # 执行评估
    metrics = evaluator.evaluate(test_data)
    
    # 生成报告
    evaluator.generate_report(metrics, args.output)


if __name__ == "__main__":
    main()
