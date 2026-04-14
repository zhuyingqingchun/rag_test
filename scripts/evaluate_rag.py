#!/usr/bin/env python3
"""
RAG效果评估脚本
"""

import argparse
import json
import numpy as np
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModel
import time
import os


@dataclass
class RAGEvaluationMetrics:
    """RAG评估指标"""
    # 检索指标
    retrieval_accuracy: float  # 检索准确率
    retrieval_recall: float    # 检索召回率
    retrieval_precision: float # 检索精确率
    mrr: float                 # 平均倒数排名
    
    # 生成指标
    bleu_score: float          # BLEU分数
    rouge_l: float             # ROUGE-L分数
    answer_relevance: float    # 回答相关性
    answer_faithfulness: float # 回答忠实度
    
    # 效率指标
    avg_retrieval_time: float  # 平均检索时间(ms)
    avg_generation_time: float # 平均生成时间(ms)
    total_latency: float       # 总延迟(ms)
    
    # 综合指标
    overall_score: float       # 综合评分


class RAGEvaluator:
    """RAG系统评估器"""
    
    def __init__(self, encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        self.encoder = AutoModel.from_pretrained(encoder_model).to(self.device)
        self.encoder.eval()
    
    def encode_text(self, text: str) -> torch.Tensor:
        """编码文本为向量"""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()
    
    def compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    def evaluate_retrieval(
        self,
        queries: List[str],
        retrieved_docs: List[List[str]],
        ground_truth_docs: List[List[str]]
    ) -> Dict[str, float]:
        """
        评估检索性能
        
        Args:
            queries: 查询列表
            retrieved_docs: 每个查询检索到的文档列表
            ground_truth_docs: 每个查询的真实相关文档列表
        
        Returns:
            检索指标字典
        """
        metrics = {
            'accuracy': [],
            'recall': [],
            'precision': [],
            'mrr': []
        }
        
        for query, retrieved, ground_truth in zip(queries, retrieved_docs, ground_truth_docs):
            # 计算准确率 (至少有一个相关文档被检索到)
            relevant_retrieved = set(retrieved) & set(ground_truth)
            accuracy = 1.0 if len(relevant_retrieved) > 0 else 0.0
            metrics['accuracy'].append(accuracy)
            
            # 计算召回率
            recall = len(relevant_retrieved) / len(ground_truth) if ground_truth else 0.0
            metrics['recall'].append(recall)
            
            # 计算精确率
            precision = len(relevant_retrieved) / len(retrieved) if retrieved else 0.0
            metrics['precision'].append(precision)
            
            # 计算MRR (Mean Reciprocal Rank)
            mrr = 0.0
            for i, doc in enumerate(retrieved):
                if doc in ground_truth:
                    mrr = 1.0 / (i + 1)
                    break
            metrics['mrr'].append(mrr)
        
        return {
            'retrieval_accuracy': np.mean(metrics['accuracy']),
            'retrieval_recall': np.mean(metrics['recall']),
            'retrieval_precision': np.mean(metrics['precision']),
            'mrr': np.mean(metrics['mrr'])
        }
    
    def evaluate_generation(
        self,
        generated_answers: List[str],
        reference_answers: List[str],
        queries: List[str]
    ) -> Dict[str, float]:
        """
        评估生成性能
        
        Args:
            generated_answers: 生成的回答列表
            reference_answers: 参考答案列表
            queries: 查询列表
        
        Returns:
            生成指标字典
        """
        from rouge_score import rouge_scorer
        from nltk.translate.bleu_score import sentence_bleu
        
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
        rouge_scores = []
        bleu_scores = []
        answer_relevances = []
        answer_faithfulnesses = []
        
        for gen_answer, ref_answer, query in zip(generated_answers, reference_answers, queries):
            # 计算ROUGE-L分数
            scores = scorer.score(ref_answer, gen_answer)
            rouge_l = scores['rougeL'].fmeasure
            rouge_scores.append(rouge_l)
            
            # 计算BLEU分数
            try:
                # 简单的BLEU计算
                reference = [ref_answer.split()]
                candidate = gen_answer.split()
                bleu = sentence_bleu(reference, candidate)
                bleu_scores.append(bleu)
            except:
                bleu_scores.append(0.0)
            
            # 计算回答相关性 (生成回答与查询的相似度)
            query_emb = self.encode_text(query)
            answer_emb = self.encode_text(gen_answer)
            relevance = self.compute_cosine_similarity(query_emb[0], answer_emb[0])
            answer_relevances.append(relevance)
            
            # 计算回答忠实度 (生成回答与参考答案的相似度)
            ref_emb = self.encode_text(ref_answer)
            faithfulness = self.compute_cosine_similarity(answer_emb[0], ref_emb[0])
            answer_faithfulnesses.append(faithfulness)
        
        return {
            'bleu_score': np.mean(bleu_scores),
            'rouge_l': np.mean(rouge_scores),
            'answer_relevance': np.mean(answer_relevances),
            'answer_faithfulness': np.mean(answer_faithfulnesses)
        }
    
    def evaluate_efficiency(
        self,
        retrieval_times: List[float],
        generation_times: List[float]
    ) -> Dict[str, float]:
        """
        评估效率指标
        
        Args:
            retrieval_times: 检索时间列表(ms)
            generation_times: 生成时间列表(ms)
        
        Returns:
            效率指标字典
        """
        return {
            'avg_retrieval_time': np.mean(retrieval_times),
            'avg_generation_time': np.mean(generation_times),
            'total_latency': np.mean(retrieval_times) + np.mean(generation_times)
        }
    
    def compute_overall_score(self, metrics: Dict[str, float]) -> float:
        """计算综合评分"""
        # 权重配置
        weights = {
            'retrieval_accuracy': 0.15,
            'retrieval_recall': 0.15,
            'retrieval_precision': 0.10,
            'mrr': 0.10,
            'rouge_l': 0.20,
            'answer_relevance': 0.15,
            'answer_faithfulness': 0.15
        }
        
        overall_score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                overall_score += metrics[metric] * weight
        
        return overall_score
    
    def evaluate(
        self,
        test_data: List[Dict[str, Any]],
        rag_system: Any
    ) -> RAGEvaluationMetrics:
        """
        完整评估RAG系统
        
        Args:
            test_data: 测试数据列表，每项包含query, ground_truth_docs, reference_answer
            rag_system: RAG系统实例
        
        Returns:
            评估指标
        """
        queries = []
        retrieved_docs_list = []
        ground_truth_docs_list = []
        generated_answers = []
        reference_answers = []
        retrieval_times = []
        generation_times = []
        
        print("Evaluating RAG system...")
        
        for i, item in enumerate(test_data):
            query = item['query']
            queries.append(query)
            ground_truth_docs_list.append(item['ground_truth_docs'])
            reference_answers.append(item['reference_answer'])
            
            # 执行RAG并记录时间
            
            # 检索
            start_time = time.time()
            retrieved_docs = rag_system.retrieve(query)
            retrieval_time = (time.time() - start_time) * 1000
            retrieval_times.append(retrieval_time)
            retrieved_docs_list.append(retrieved_docs)
            
            # 生成
            start_time = time.time()
            answer = rag_system.generate(query, retrieved_docs)
            generation_time = (time.time() - start_time) * 1000
            generation_times.append(generation_time)
            generated_answers.append(answer)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(test_data)} queries")
        
        # 计算各项指标
        retrieval_metrics = self.evaluate_retrieval(
            queries, retrieved_docs_list, ground_truth_docs_list
        )
        
        generation_metrics = self.evaluate_generation(
            generated_answers, reference_answers, queries
        )
        
        efficiency_metrics = self.evaluate_efficiency(
            retrieval_times, generation_times
        )
        
        # 合并所有指标
        all_metrics = {**retrieval_metrics, **generation_metrics, **efficiency_metrics}
        overall_score = self.compute_overall_score(all_metrics)
        
        return RAGEvaluationMetrics(
            **all_metrics,
            overall_score=overall_score
        )
    
    def generate_report(self, metrics: RAGEvaluationMetrics, output_path: str):
        """生成评估报告"""
        report = {
            'retrieval_metrics': {
                'accuracy': float(round(metrics.retrieval_accuracy, 4)),
                'recall': float(round(metrics.retrieval_recall, 4)),
                'precision': float(round(metrics.retrieval_precision, 4)),
                'mrr': float(round(metrics.mrr, 4))
            },
            'generation_metrics': {
                'bleu_score': float(round(metrics.bleu_score, 4)),
                'rouge_l': float(round(metrics.rouge_l, 4)),
                'answer_relevance': float(round(metrics.answer_relevance, 4)),
                'answer_faithfulness': float(round(metrics.answer_faithfulness, 4))
            },
            'efficiency_metrics': {
                'avg_retrieval_time_ms': float(round(metrics.avg_retrieval_time, 2)),
                'avg_generation_time_ms': float(round(metrics.avg_generation_time, 2)),
                'total_latency_ms': float(round(metrics.total_latency, 2))
            },
            'overall_score': float(round(metrics.overall_score, 4))
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\nEvaluation report saved to: {output_path}")
        print("\n" + "="*60)
        print("RAG Evaluation Results")
        print("="*60)
        print(f"Retrieval Accuracy:     {metrics.retrieval_accuracy:.4f}")
        print(f"Retrieval Recall:       {metrics.retrieval_recall:.4f}")
        print(f"Retrieval Precision:    {metrics.retrieval_precision:.4f}")
        print(f"MRR:                    {metrics.mrr:.4f}")
        print(f"ROUGE-L:                {metrics.rouge_l:.4f}")
        print(f"Answer Relevance:       {metrics.answer_relevance:.4f}")
        print(f"Answer Faithfulness:    {metrics.answer_faithfulness:.4f}")
        print(f"Avg Retrieval Time:     {metrics.avg_retrieval_time:.2f} ms")
        print(f"Avg Generation Time:    {metrics.avg_generation_time:.2f} ms")
        print(f"Total Latency:          {metrics.total_latency:.2f} ms")
        print(f"Overall Score:          {metrics.overall_score:.4f}")
        print("="*60)


# 示例RAG系统接口
class ExampleRAGSystem:
    """示例RAG系统，用于演示评估"""
    
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.encoder.eval()
        
        # 预编码所有文档
        self.doc_embeddings = self._encode_documents()
    
    def _encode_documents(self):
        """编码所有文档"""
        embeddings = []
        for doc in self.documents:
            inputs = self.tokenizer(doc, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.encoder(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).numpy()
            embeddings.append(emb[0])
        return np.array(embeddings)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """检索相关文档"""
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            query_emb = outputs.last_hidden_state.mean(dim=1).numpy()[0]
        
        # 计算相似度
        similarities = np.dot(self.doc_embeddings, query_emb) / (
            np.linalg.norm(self.doc_embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        
        # 获取top-k文档
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]
    
    def generate(self, query: str, retrieved_docs: List[str]) -> str:
        """生成回答（简化版）"""
        # 这里应该调用实际的生成模型
        # 简化处理：返回检索到的文档的拼接
        return " ".join(retrieved_docs[:2])


def generate_test_data(num_queries: int = 20) -> List[Dict[str, Any]]:
    """生成测试数据"""
    test_data = []
    
    # 示例文档
    documents = [
        "飞控系统（Flight Control System，FCS）是飞行器的核心控制系统，负责稳定飞行器姿态、控制飞行轨迹和执行导航任务。",
        "飞控系统主要由传感器系统、飞控计算机、执行机构和通信系统组成。",
        "传感器系统包括陀螺仪、加速度计、气压计、磁力计等，用于采集飞行器的姿态、位置和速度信息。",
        "飞控计算机是核心处理单元，运行控制算法，处理传感器数据并生成控制指令。",
        "执行机构包括舵机、电机等，执行飞控计算机的控制指令，调整飞行器的姿态和动力。",
        "通信系统用于与地面站、其他飞行器或导航卫星进行通信。",
        "现代飞控系统通常采用电传飞控（Fly-by-Wire）技术，通过传感器采集飞行状态数据，经过计算机处理后发送指令给执行机构。",
        "飞控系统的工作原理是通过姿态传感器实时监测飞行器的姿态角，并与期望姿态进行比较，计算出控制指令。",
        "飞控系统通常支持多种飞行模式，如稳定模式、定高模式、定点模式、返航模式和自主导航模式。",
        "飞控系统的关键技术包括传感器融合技术、控制算法、故障检测与容错等。"
    ]
    
    # 示例查询和参考答案
    query_answer_pairs = [
        {
            "query": "飞控系统的组成部分有哪些？",
            "ground_truth_docs": [documents[1], documents[2], documents[3], documents[4], documents[5]],
            "reference_answer": "飞控系统主要由传感器系统、飞控计算机、执行机构和通信系统组成。传感器系统包括陀螺仪、加速度计、气压计、磁力计等；飞控计算机是核心处理单元，运行控制算法；执行机构包括舵机、电机等；通信系统用于与地面站、其他飞行器或导航卫星进行通信。"
        },
        {
            "query": "飞控系统的工作原理是什么？",
            "ground_truth_docs": [documents[0], documents[7]],
            "reference_answer": "飞控系统的工作原理是通过姿态传感器实时监测飞行器的姿态角（俯仰角、横滚角、偏航角），并与期望姿态进行比较，计算出控制指令，通过执行机构调整飞行器姿态。现代飞控系统通常采用电传飞控技术，通过传感器采集飞行状态数据，经过计算机处理后发送指令给执行机构。"
        },
        {
            "query": "飞控系统有哪些飞行模式？",
            "ground_truth_docs": [documents[8]],
            "reference_answer": "飞控系统通常支持多种飞行模式，如稳定模式、定高模式、定点模式、返航模式和自主导航模式。"
        },
        {
            "query": "飞控系统的关键技术有哪些？",
            "ground_truth_docs": [documents[9]],
            "reference_answer": "飞控系统的关键技术包括传感器融合技术、控制算法、故障检测与容错等。"
        },
        {
            "query": "什么是电传飞控技术？",
            "ground_truth_docs": [documents[6]],
            "reference_answer": "电传飞控（Fly-by-Wire）技术是现代飞控系统采用的技术，通过传感器采集飞行状态数据，经过计算机处理后发送指令给执行机构，实现对飞行器的精确控制。"
        }
    ]
    
    # 生成测试数据
    for i in range(num_queries):
        pair = query_answer_pairs[i % len(query_answer_pairs)]
        test_data.append({
            "query": pair["query"],
            "ground_truth_docs": pair["ground_truth_docs"],
            "reference_answer": pair["reference_answer"]
        })
    
    return test_data, documents


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG System")
    parser.add_argument("--output", type=str, default="rag_evaluation_report.json", help="Output report path")
    parser.add_argument("--num_queries", type=int, default=20, help="Number of test queries")
    args = parser.parse_args()
    
    # 生成测试数据
    test_data, documents = generate_test_data(args.num_queries)
    
    # 创建RAG系统
    rag_system = ExampleRAGSystem(documents)
    
    # 创建评估器
    evaluator = RAGEvaluator()
    
    # 执行评估
    metrics = evaluator.evaluate(test_data, rag_system)
    
    # 生成报告
    evaluator.generate_report(metrics, args.output)


if __name__ == "__main__":
    main()
