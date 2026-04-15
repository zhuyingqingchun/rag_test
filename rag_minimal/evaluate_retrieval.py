#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检索评估模块。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from text_utils import unique_tokens


def load_retrieved_context(context_path: str) -> Dict[str, Any]:
    with open(context_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_recall_at_k(results: List[Dict[str, Any]], relevant_docs: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k_sources = [r['source'] for r in results[:k]]
    total_relevant = len(relevant_docs)
    if total_relevant == 0:
        return 1.0
    relevant_count = sum(1 for doc in relevant_docs if doc in top_k_sources)
    return relevant_count / total_relevant


def calculate_precision_at_k(results: List[Dict[str, Any]], relevant_docs: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k_sources = [r['source'] for r in results[:k]]
    relevant_count = sum(1 for doc in top_k_sources if doc in relevant_docs)
    return relevant_count / k


def check_hit_rate(results: List[Dict[str, Any]], relevant_docs: List[str], k: int) -> bool:
    top_k_sources = [r['source'] for r in results[:k]]
    return any(doc in top_k_sources for doc in relevant_docs)


def calculate_relevance_score(result: Dict[str, Any], query: str, relevant_docs: List[str]) -> int:
    source = result['source']
    text_tokens = unique_tokens(result['text'])
    query_tokens = unique_tokens(query)
    overlap = query_tokens & text_tokens
    if source in relevant_docs:
        return 2 if overlap else 1
    return 1 if overlap else 0


def evaluate_retrieval(query: str, context_path: str, relevant_docs: Optional[List[str]] = None, k: int = 5) -> Dict[str, Any]:
    context = load_retrieved_context(context_path)
    results = context.get('results', [])
    if not results:
        return {
            'status': 'warning',
            'message': '没有检索结果',
            'query': query,
            'total_chunks': context.get('total_chunks', 0),
            'results_count': 0,
            'metrics': {'recall@k': 0.0, 'precision@k': 0.0, 'hit_rate': False, 'avg_relevance': 0.0},
            'top_k_results': [],
        }
    if relevant_docs is None:
        relevant_docs = list({r['source'] for r in results})
    recall = calculate_recall_at_k(results, relevant_docs, k)
    precision = calculate_precision_at_k(results, relevant_docs, k)
    hit = check_hit_rate(results, relevant_docs, k)
    relevance_scores = [calculate_relevance_score(r, query, relevant_docs) for r in results[:k]]
    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0
    return {
        'status': 'success',
        'query': query,
        'total_chunks': context.get('total_chunks', 0),
        'results_count': len(results),
        'metrics': {'recall@k': round(recall, 3), 'precision@k': round(precision, 3), 'hit_rate': hit, 'avg_relevance': round(avg_relevance, 2)},
        'relevance_scores': relevance_scores,
        'top_k_results': [{'rank': i + 1, 'source': r['source'], 'score': r.get('score', 0), 'relevance': relevance_scores[i] if i < len(relevance_scores) else 0, 'preview': r['text'][:100]} for i, r in enumerate(results[:k])],
    }


def print_evaluation(result: Dict[str, Any]) -> None:
    print('=' * 60)
    print('检索评估结果')
    print('=' * 60)
    print(f"查询: {result['query']}")
    print(f"总块数: {result['total_chunks']}")
    print(f"召回数: {result['results_count']}")
    metrics = result['metrics']
    print(f"  Recall@5:    {metrics['recall@k']:.3f}")
    print(f"  Precision@5: {metrics['precision@k']:.3f}")
    print(f"  Hit@5:       {'✓' if metrics['hit_rate'] else '✗'}")
    print(f"  平均相关性:  {metrics['avg_relevance']:.2f}/2")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description='检索评估模块')
    parser.add_argument('--query', '-q', required=True, help='用户查询')
    parser.add_argument('--context', '-c', required=True, help='检索结果 JSON 文件')
    parser.add_argument('--relevant', '-r', nargs='+', default=None, help='相关文档列表')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='评估的 top-k')
    parser.add_argument('--output', '-o', default=None, help='输出评估结果文件')
    args = parser.parse_args()
    result = evaluate_retrieval(args.query, args.context, args.relevant, args.top_k)
    print_evaluation(result)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
