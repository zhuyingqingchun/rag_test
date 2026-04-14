"""
REFRAG评估指标模块
支持RAG、多轮对话、长文档摘要等任务的评估
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from collections import defaultdict
import re


class MetricsCalculator:
    """指标计算器基类"""
    
    @staticmethod
    def accuracy(predictions: List[Any], references: List[Any]) -> float:
        """计算准确率"""
        correct = sum(p == r for p, r in zip(predictions, references))
        return correct / len(predictions) if predictions else 0.0
    
    @staticmethod
    def f1_score(predictions: List[Any], references: List[Any]) -> float:
        """计算F1分数"""
        tp = sum(p == r == 1 for p, r in zip(predictions, references))
        fp = sum(p == 1 and r == 0 for p, r in zip(predictions, references))
        fn = sum(p == 0 and r == 1 for p, r in zip(predictions, references))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


class RAGMetrics:
    """RAG任务评估指标"""
    
    @staticmethod
    def exact_match(predictions: List[str], references: List[str]) -> float:
        """精确匹配率"""
        matches = [p.strip().lower() == r.strip().lower() 
                  for p, r in zip(predictions, references)]
        return sum(matches) / len(matches) if matches else 0.0
    
    @staticmethod
    def f1_score(predictions: List[str], references: List[str]) -> float:
        """Token-level F1分数"""
        f1_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            
            if len(pred_tokens) == 0 and len(ref_tokens) == 0:
                f1_scores.append(1.0)
                continue
            
            common = pred_tokens & ref_tokens
            precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
            recall = len(common) / len(ref_tokens) if ref_tokens else 0.0
            
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * precision * recall / (precision + recall))
        
        return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    @staticmethod
    def retrieval_accuracy(
        retrieved_passages: List[List[str]],
        relevant_passages: List[List[str]]
    ) -> Dict[str, float]:
        """检索准确率指标"""
        recall_at_k = defaultdict(list)
        precision_at_k = defaultdict(list)
        
        k_values = [1, 5, 10, 20]
        
        for retrieved, relevant in zip(retrieved_passages, relevant_passages):
            relevant_set = set(relevant)
            
            for k in k_values:
                if k > len(retrieved):
                    continue
                
                retrieved_k = set(retrieved[:k])
                hits = len(retrieved_k & relevant_set)
                
                recall = hits / len(relevant_set) if relevant_set else 0.0
                precision = hits / k
                
                recall_at_k[f"R@{k}"].append(recall)
                precision_at_k[f"P@{k}"].append(precision)
        
        metrics = {}
        for k in k_values:
            key = f"R@{k}"
            if key in recall_at_k:
                metrics[key] = sum(recall_at_k[key]) / len(recall_at_k[key])
            
            key = f"P@{k}"
            if key in precision_at_k:
                metrics[key] = sum(precision_at_k[key]) / len(precision_at_k[key])
        
        return metrics


class DialogueMetrics:
    """多轮对话评估指标"""
    
    @staticmethod
    def turn_accuracy(
        predictions: List[List[str]],
        references: List[List[str]]
    ) -> Dict[str, float]:
        """每轮对话准确率"""
        turn_accuracies = defaultdict(list)
        
        for pred_dialogue, ref_dialogue in zip(predictions, references):
            for turn_idx, (pred, ref) in enumerate(zip(pred_dialogue, ref_dialogue)):
                correct = pred.strip().lower() == ref.strip().lower()
                turn_accuracies[f"turn_{turn_idx}"].append(float(correct))
        
        metrics = {}
        for turn, accs in turn_accuracies.items():
            metrics[turn] = sum(accs) / len(accs)
        
        # 整体准确率
        all_accuracies = []
        for accs in turn_accuracies.values():
            all_accuracies.extend(accs)
        metrics["overall"] = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0
        
        return metrics
    
    @staticmethod
    def response_relevance(
        responses: List[str],
        contexts: List[str]
    ) -> float:
        """响应相关性（简化版，基于关键词重叠）"""
        relevance_scores = []
        
        for response, context in zip(responses, contexts):
            response_words = set(response.lower().split())
            context_words = set(context.lower().split())
            
            if not response_words:
                relevance_scores.append(0.0)
                continue
            
            overlap = len(response_words & context_words)
            relevance = overlap / len(response_words)
            relevance_scores.append(relevance)
        
        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0


class SummarizationMetrics:
    """长文档摘要评估指标"""
    
    @staticmethod
    def rouge_score(
        predictions: List[str],
        references: List[str],
        max_n: int = 2
    ) -> Dict[str, float]:
        """
        计算ROUGE分数（简化实现）
        
        Args:
            predictions: 预测摘要列表
            references: 参考摘要列表
            max_n: 最大n-gram大小
        
        Returns:
            rouge_scores: ROUGE分数字典
        """
        def get_ngrams(text: str, n: int) -> set:
            words = text.lower().split()
            return set(tuple(words[i:i+n]) for i in range(len(words)-n+1))
        
        def compute_rouge_n(pred: str, ref: str, n: int) -> float:
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)
            
            if not ref_ngrams:
                return 0.0
            
            overlap = len(pred_ngrams & ref_ngrams)
            return overlap / len(ref_ngrams)
        
        rouge_scores = defaultdict(list)
        
        for pred, ref in zip(predictions, references):
            for n in range(1, max_n + 1):
                score = compute_rouge_n(pred, ref, n)
                rouge_scores[f"rouge-{n}"].append(score)
        
        # 计算平均值
        results = {}
        for metric, scores in rouge_scores.items():
            results[metric] = sum(scores) / len(scores) if scores else 0.0
        
        return results
    
    @staticmethod
    def coverage_score(
        summaries: List[str],
        documents: List[str]
    ) -> float:
        """摘要覆盖率（关键信息覆盖程度）"""
        coverage_scores = []
        
        for summary, document in zip(summaries, documents):
            # 提取文档中的关键句子（简化：按长度选择）
            doc_sentences = re.split(r'[.!?]+', document)
            key_sentences = sorted(doc_sentences, key=len, reverse=True)[:5]
            
            # 检查关键信息是否在摘要中
            covered = 0
            for sentence in key_sentences:
                sentence_words = set(sentence.lower().split())
                summary_words = set(summary.lower().split())
                
                if sentence_words and len(sentence_words & summary_words) / len(sentence_words) > 0.5:
                    covered += 1
            
            coverage = covered / len(key_sentences) if key_sentences else 0.0
            coverage_scores.append(coverage)
        
        return sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0
    
    @staticmethod
    def compression_ratio(
        summaries: List[str],
        documents: List[str]
    ) -> float:
        """压缩率"""
        ratios = []
        
        for summary, document in zip(summaries, documents):
            doc_len = len(document.split())
            summary_len = len(summary.split())
            
            if doc_len > 0:
                ratio = summary_len / doc_len
                ratios.append(ratio)
        
        return sum(ratios) / len(ratios) if ratios else 0.0


class EfficiencyMetrics:
    """效率指标"""
    
    @staticmethod
    def latency_metrics(latencies: List[float]) -> Dict[str, float]:
        """延迟指标"""
        if not latencies:
            return {}
        
        return {
            "mean_latency_ms": np.mean(latencies),
            "median_latency_ms": np.median(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies)
        }
    
    @staticmethod
    def throughput_metrics(
        num_tokens: List[int],
        times: List[float]
    ) -> Dict[str, float]:
        """吞吐量指标"""
        if not num_tokens or not times:
            return {}
        
        throughputs = [n / t for n, t in zip(num_tokens, times)]
        
        return {
            "mean_throughput_tokens_per_sec": np.mean(throughputs),
            "total_tokens": sum(num_tokens),
            "total_time_sec": sum(times)
        }
    
    @staticmethod
    def memory_metrics(memory_usage_mb: List[float]) -> Dict[str, float]:
        """内存使用指标"""
        if not memory_usage_mb:
            return {}
        
        return {
            "mean_memory_mb": np.mean(memory_usage_mb),
            "peak_memory_mb": np.max(memory_usage_mb),
            "memory_efficiency": np.mean(memory_usage_mb) / np.max(memory_usage_mb) if np.max(memory_usage_mb) > 0 else 0.0
        }


class ComprehensiveEvaluator:
    """综合评估器"""
    
    def __init__(self):
        self.rag_metrics = RAGMetrics()
        self.dialogue_metrics = DialogueMetrics()
        self.summarization_metrics = SummarizationMetrics()
        self.efficiency_metrics = EfficiencyMetrics()
    
    def evaluate_rag(
        self,
        predictions: List[str],
        references: List[str],
        retrieved_passages: Optional[List[List[str]]] = None,
        relevant_passages: Optional[List[List[str]]] = None
    ) -> Dict[str, float]:
        """评估RAG任务"""
        metrics = {
            "exact_match": self.rag_metrics.exact_match(predictions, references),
            "f1_score": self.rag_metrics.f1_score(predictions, references)
        }
        
        if retrieved_passages and relevant_passages:
            retrieval_metrics = self.rag_metrics.retrieval_accuracy(
                retrieved_passages, relevant_passages
            )
            metrics.update(retrieval_metrics)
        
        return metrics
    
    def evaluate_dialogue(
        self,
        predictions: List[List[str]],
        references: List[List[str]],
        contexts: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """评估多轮对话任务"""
        metrics = self.dialogue_metrics.turn_accuracy(predictions, references)
        
        if contexts:
            # 展平预测和上下文
            flat_predictions = [p for dialogue in predictions for p in dialogue]
            flat_contexts = []
            for ctx, dialogue in zip(contexts, predictions):
                flat_contexts.extend([ctx] * len(dialogue))
            
            metrics["response_relevance"] = self.dialogue_metrics.response_relevance(
                flat_predictions, flat_contexts
            )
        
        return metrics
    
    def evaluate_summarization(
        self,
        predictions: List[str],
        references: List[str],
        documents: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """评估摘要任务"""
        metrics = self.summarization_metrics.rouge_score(predictions, references)
        
        if documents:
            metrics["coverage"] = self.summarization_metrics.coverage_score(
                predictions, documents
            )
            metrics["compression_ratio"] = self.summarization_metrics.compression_ratio(
                predictions, documents
            )
        
        return metrics
    
    def evaluate_efficiency(
        self,
        latencies: List[float],
        num_tokens: Optional[List[int]] = None,
        times: Optional[List[float]] = None,
        memory_usage_mb: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """评估效率指标"""
        metrics = self.efficiency_metrics.latency_metrics(latencies)
        
        if num_tokens and times:
            throughput_metrics = self.efficiency_metrics.throughput_metrics(
                num_tokens, times
            )
            metrics.update(throughput_metrics)
        
        if memory_usage_mb:
            memory_metrics = self.efficiency_metrics.memory_metrics(memory_usage_mb)
            metrics.update(memory_metrics)
        
        return metrics
    
    def generate_report(self, all_metrics: Dict[str, Dict[str, float]]) -> str:
        """生成评估报告"""
        report_lines = ["# REFRAG Evaluation Report", ""]
        
        for task, metrics in all_metrics.items():
            report_lines.append(f"## {task.upper()} Metrics")
            report_lines.append("")
            report_lines.append("| Metric | Value |")
            report_lines.append("|--------|-------|")
            
            for metric, value in metrics.items():
                if isinstance(value, float):
                    report_lines.append(f"| {metric} | {value:.4f} |")
                else:
                    report_lines.append(f"| {metric} | {value} |")
            
            report_lines.append("")
        
        return "\n".join(report_lines)
