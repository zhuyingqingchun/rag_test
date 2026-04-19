#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线评测高级检索：rewrite / rerank / abstain 视角下评估 advanced_retrieve。"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from advanced_retrieve import advanced_retrieve, parse_abstain_thresholds_arg
from evaluate_retrieval import (
    _extract_query,
    _gold_size,
    _is_relevant,
    build_gold_spec,
    load_eval_cases,
)


def evaluate_single_case(case: dict[str, Any],
                         chunks_path: str,
                         method: str,
                         top_k: int,
                         index_dir: str | None,
                         embedding_model: str,
                         hybrid_weights: dict[str, float] | None,
                         rewrite: bool,
                         rewrite_max_queries: int,
                         rerank: bool,
                         reranker_model: str,
                         rerank_top_n: int,
                         rrf_k: int,
                         abstain_thresholds: dict[str, float] | None) -> dict[str, Any]:
    query = _extract_query(case)
    gold = build_gold_spec(case)
    payload = advanced_retrieve(
        query=query,
        chunks_path=chunks_path,
        method=method,
        top_k=top_k,
        index_dir=index_dir,
        embedding_model=embedding_model,
        hybrid_weights=hybrid_weights,
        rewrite=rewrite,
        rewrite_max_queries=rewrite_max_queries,
        rerank=rerank,
        reranker_model=reranker_model,
        rerank_top_n=rerank_top_n,
        rrf_k=rrf_k,
        abstain_thresholds=abstain_thresholds,
    )

    matched_ranks: list[int] = []
    matched_items: list[dict[str, Any]] = []
    for item in payload['results']:
        if _is_relevant(item, gold):
            matched_ranks.append(int(item['rank']))
            matched_items.append(
                {
                    'rank': int(item['rank']),
                    'score': float(item.get('score', 0.0)),
                    'rrf_score': float(item.get('rrf_score', item.get('score', 0.0))),
                    'doc_id': item['chunk'].get('doc_id'),
                    'chunk_id': item['chunk'].get('chunk_id'),
                    'source': item['chunk'].get('source'),
                    'text_preview': str(item['chunk'].get('text', ''))[:120],
                }
            )

    relevant_hits = len(matched_ranks)
    precision_at_k = relevant_hits / max(len(payload['results']), 1)
    recall_at_k = min(relevant_hits, _gold_size(gold)) / max(_gold_size(gold), 1)
    first_relevant_rank = matched_ranks[0] if matched_ranks else None
    mrr_at_k = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
    hit_at_k = 1.0 if matched_ranks else 0.0
    should_abstain = bool(payload['advanced']['should_abstain'])
    answered = 0.0 if should_abstain else 1.0
    useful_answer = 1.0 if (not should_abstain and matched_ranks) else 0.0

    return {
        'case_id': case.get('id') or case.get('case_id') or query,
        'query': query,
        'method': method,
        'top_k': top_k,
        'hit_at_k': round(hit_at_k, 6),
        'precision_at_k': round(precision_at_k, 6),
        'recall_at_k': round(recall_at_k, 6),
        'mrr_at_k': round(mrr_at_k, 6),
        'answered': answered,
        'should_abstain': should_abstain,
        'abstain_reason': payload['advanced'].get('abstain_reason', ''),
        'useful_answer': useful_answer,
        'first_relevant_rank': first_relevant_rank,
        'matched_ranks': matched_ranks,
        'matched_items': matched_items,
        'rewrites': payload.get('rewrites', []),
        'confidence': payload['advanced'].get('confidence'),
        'confidence_detail': payload['advanced'].get('confidence_detail'),
        'results': [
            {
                'rank': int(item['rank']),
                'score': float(item.get('score', 0.0)),
                'rrf_score': float(item.get('rrf_score', item.get('score', 0.0))),
                'rrf_rank': int(item.get('rrf_rank', 0)),
                'doc_id': item['chunk'].get('doc_id'),
                'chunk_id': item['chunk'].get('chunk_id'),
                'source': item['chunk'].get('source'),
                'text_preview': str(item['chunk'].get('text', ''))[:120],
            }
            for item in payload['results']
        ],
    }


def summarize(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(case_results)
    if count == 0:
        raise ValueError('没有评测结果可汇总')

    def _mean(key: str) -> float:
        return round(sum(float(item[key]) for item in case_results) / count, 6)

    ranks = [item['first_relevant_rank'] for item in case_results if item['first_relevant_rank']]
    answered_cases = [item for item in case_results if not item['should_abstain']]
    useful_answers = sum(float(item['useful_answer']) for item in case_results)

    answered_hit_rate = (
        round(sum(float(item['hit_at_k']) for item in answered_cases) / len(answered_cases), 6)
        if answered_cases else 0.0
    )

    return {
        'queries': count,
        'hit_at_k': _mean('hit_at_k'),
        'precision_at_k': _mean('precision_at_k'),
        'recall_at_k': _mean('recall_at_k'),
        'mrr_at_k': _mean('mrr_at_k'),
        'abstain_rate': round(sum(1.0 for item in case_results if item['should_abstain']) / count, 6),
        'answered_rate': round(sum(float(item['answered']) for item in case_results) / count, 6),
        'answered_hit_rate': answered_hit_rate,
        'useful_answer_rate': round(useful_answers / count, 6),
        'avg_first_relevant_rank': round(sum(ranks) / len(ranks), 6) if ranks else None,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description='离线评测高级检索 advanced_retrieve')
    parser.add_argument('--eval-file', required=True, help='评测样本 JSON / JSONL 文件')
    parser.add_argument('--chunks', required=True, help='chunk 数据文件路径')
    parser.add_argument('--method', default='hybrid', choices=['bm25', 'tfidf', 'keyword', 'vector', 'hybrid'], help='基础检索方法')
    parser.add_argument('--top-k', type=int, default=5, help='最终返回 top-k')
    parser.add_argument('--index-dir', default='', help='向量索引目录')
    parser.add_argument('--embedding-model', default='BAAI/bge-m3', help='embedding 模型')
    parser.add_argument('--hybrid-weights', default='', help='hybrid 权重 JSON 字符串或 JSON 文件路径')
    parser.add_argument('--rewrite', action='store_true', help='启用多查询改写')
    parser.add_argument('--rewrite-max-queries', type=int, default=4, help='最大改写数')
    parser.add_argument('--rerank', action='store_true', help='启用 rerank')
    parser.add_argument('--reranker-model', default='', help='CrossEncoder / BGE reranker 模型名或本地路径')
    parser.add_argument('--rerank-top-n', type=int, default=10, help='参与 rerank 的候选数')
    parser.add_argument('--rrf-k', type=int, default=60, help='RRF 常数')
    parser.add_argument('--abstain-thresholds', default='', help='abstain 阈值 JSON 字符串或 JSON 文件路径')
    parser.add_argument('--output', default='./data/processed/advanced_retrieval_eval_report.json', help='输出报告路径')
    args = parser.parse_args()

    cases = load_eval_cases(args.eval_file)
    hybrid_weights = None
    if args.hybrid_weights:
        from retrieve import parse_hybrid_weights_arg
        hybrid_weights = parse_hybrid_weights_arg(args.hybrid_weights)

    abstain_thresholds = parse_abstain_thresholds_arg(args.abstain_thresholds)

    case_results = [
        evaluate_single_case(
            case=case,
            chunks_path=args.chunks,
            method=args.method,
            top_k=args.top_k,
            index_dir=args.index_dir or None,
            embedding_model=args.embedding_model,
            hybrid_weights=hybrid_weights,
            rewrite=args.rewrite,
            rewrite_max_queries=args.rewrite_max_queries,
            rerank=args.rerank,
            reranker_model=args.reranker_model,
            rerank_top_n=args.rerank_top_n,
            rrf_k=args.rrf_k,
            abstain_thresholds=abstain_thresholds,
        )
        for case in cases
    ]

    summary = summarize(case_results)
    output = {
        'timestamp': datetime.now().isoformat(),
        'eval_file': os.path.abspath(args.eval_file),
        'chunks': os.path.abspath(args.chunks),
        'index_dir': os.path.abspath(args.index_dir) if args.index_dir else '',
        'embedding_model': args.embedding_model,
        'method': args.method,
        'top_k': args.top_k,
        'rewrite': args.rewrite,
        'rewrite_max_queries': args.rewrite_max_queries,
        'rerank': args.rerank,
        'reranker_model': args.reranker_model,
        'rerank_top_n': args.rerank_top_n,
        'rrf_k': args.rrf_k,
        'abstain_thresholds': abstain_thresholds,
        'summary': summary,
        'cases': case_results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'评测样本数: {len(cases)}')
    print(f'输出报告: {output_path}')
    print(
        f"[{args.method}] hit@{args.top_k}={summary['hit_at_k']:.4f} "
        f"precision@{args.top_k}={summary['precision_at_k']:.4f} "
        f"recall@{args.top_k}={summary['recall_at_k']:.4f} "
        f"mrr@{args.top_k}={summary['mrr_at_k']:.4f} "
        f"abstain_rate={summary['abstain_rate']:.4f} "
        f"useful_answer_rate={summary['useful_answer_rate']:.4f}"
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
