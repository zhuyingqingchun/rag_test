#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""高级检索：多查询改写 + RRF 融合 + 可选 rerank + abstain 建议。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from query_rewrite import generate_rewrites
from rerank import rerank_results
from retrieve import parse_hybrid_weights_arg, retrieve

DEFAULT_RRF_K = 60


def _chunk_key(item: Dict[str, Any]) -> Tuple[str, str]:
    chunk = item['chunk']
    return (str(chunk.get('doc_id', '')), str(chunk.get('chunk_id', '')))


def fuse_with_rrf(payloads: List[Dict[str, Any]], top_k: int, rrf_k: int = DEFAULT_RRF_K) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
    total_queries = len(payloads)

    for payload in payloads:
        q = payload['query']
        for item in payload['results']:
            key = _chunk_key(item)
            rank = int(item['rank'])
            contribution = 1.0 / (rrf_k + rank)
            bucket = merged.setdefault(
                key,
                {
                    'chunk': item['chunk'],
                    'score': 0.0,
                    'rank': 0,
                    'query_hits': 0,
                    'raw_best_score': 0.0,
                    'per_query': [],
                },
            )
            bucket['score'] += contribution
            bucket['query_hits'] += 1
            bucket['raw_best_score'] = max(bucket['raw_best_score'], float(item.get('score', 0.0)))
            bucket['per_query'].append(
                {
                    'query': q,
                    'rank': rank,
                    'raw_score': float(item.get('score', 0.0)),
                }
            )

    ranked = sorted(
        merged.values(),
        key=lambda item: (float(item['score']), int(item['query_hits']), float(item['raw_best_score'])),
        reverse=True,
    )[:top_k]

    for idx, item in enumerate(ranked, start=1):
        item['score'] = round(float(item['score']), 6)
        item['rank'] = idx
        item['query_coverage'] = round(item['query_hits'] / max(total_queries, 1), 6)
    return ranked


def _compute_confidence(results: List[Dict[str, Any]], total_queries: int, rrf_k: int) -> Dict[str, Any]:
    if not results:
        return {
            'normalized_top1': 0.0,
            'normalized_margin': 0.0,
            'top1_query_coverage': 0.0,
            'mean_top3_score': 0.0,
        }

    max_possible = total_queries * (1.0 / (rrf_k + 1))
    top1 = float(results[0].get('rrf_score', results[0]['score']))
    top2 = float(results[1].get('rrf_score', results[1]['score'])) if len(results) > 1 else 0.0
    mean_top3 = sum(float(item.get('rrf_score', item['score'])) for item in results[:3]) / max(min(3, len(results)), 1)

    normalized_top1 = top1 / max_possible if max_possible > 0 else 0.0
    normalized_margin = (top1 - top2) / max_possible if max_possible > 0 else 0.0
    top1_query_coverage = float(results[0].get('query_coverage', 0.0))

    return {
        'normalized_top1': round(normalized_top1, 6),
        'normalized_margin': round(normalized_margin, 6),
        'top1_query_coverage': round(top1_query_coverage, 6),
        'mean_top3_score': round(mean_top3, 6),
    }


def _abstain_decision(confidence: Dict[str, Any]) -> Tuple[bool, str]:
    if confidence['normalized_top1'] < 0.22:
        return True, 'top1 证据强度不足'
    if confidence['top1_query_coverage'] < 0.34 and confidence['normalized_margin'] < 0.03:
        return True, '多查询之间缺少稳定共识'
    return False, ''


def advanced_retrieve(query: str,
                      chunks_path: str,
                      method: str = 'hybrid',
                      top_k: int = 5,
                      index_dir: str | None = None,
                      embedding_model: str = 'BAAI/bge-m3',
                      hybrid_weights: Dict[str, float] | None = None,
                      rewrite: bool = True,
                      rewrite_max_queries: int = 4,
                      rerank: bool = False,
                      reranker_model: str = '',
                      rerank_top_n: int = 10,
                      rrf_k: int = DEFAULT_RRF_K) -> Dict[str, Any]:
    rewrites = generate_rewrites(query, max_queries=rewrite_max_queries) if rewrite else [query]
    rewrites = rewrites or [query]

    payloads = [
        retrieve(
            query=q,
            chunks_path=chunks_path,
            method=method,
            top_k=max(top_k, rerank_top_n if rerank else top_k),
            index_dir=index_dir,
            embedding_model=embedding_model,
            hybrid_weights=hybrid_weights,
        )
        for q in rewrites
    ]

    fused_results = fuse_with_rrf(payloads, top_k=max(top_k, rerank_top_n if rerank else top_k), rrf_k=rrf_k)

    rrf_scores = {
        _chunk_key(item): float(item['score'])
        for item in fused_results
    }

    if rerank:
        fused_results = rerank_results(query, fused_results, model_name=reranker_model, top_n=rerank_top_n)

    final_results = fused_results[:top_k]
    for item in final_results:
        key = _chunk_key(item)
        item['rrf_score'] = rrf_scores.get(key, float(item.get('score', 0.0)))

    confidence = _compute_confidence(final_results, total_queries=len(rewrites), rrf_k=rrf_k)
    should_abstain, abstain_reason = _abstain_decision(confidence)

    return {
        'query': query,
        'method': method,
        'rewrites': rewrites,
        'rrf_k': rrf_k,
        'total_chunks': payloads[0]['total_chunks'] if payloads else 0,
        'results_count': len(final_results),
        'results': final_results,
        'index_dir': payloads[0].get('index_dir') if payloads else index_dir,
        'embedding_model': payloads[0].get('embedding_model') if payloads else embedding_model,
        'hybrid_weights': payloads[0].get('hybrid_weights') if payloads and 'hybrid_weights' in payloads[0] else hybrid_weights,
        'advanced': {
            'rewrite_enabled': rewrite,
            'rerank_enabled': rerank,
            'reranker_model': reranker_model,
            'confidence': confidence,
            'should_abstain': should_abstain,
            'abstain_reason': abstain_reason,
        },
        'per_query_runs': [
            {
                'query': payload['query'],
                'method': payload['method'],
                'results_count': payload['results_count'],
            }
            for payload in payloads
        ],
    }


def save_results(payload: Dict[str, Any], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(payload)
    serializable['timestamp'] = datetime.now().isoformat()
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description='高级检索：rewrite + RRF + rerank + abstain')
    parser.add_argument('--query', '-q', required=True, help='查询语句')
    parser.add_argument('--chunks', '-c', required=True, help='chunk 数据文件路径')
    parser.add_argument('--method', '-m', default='hybrid', choices=['bm25', 'tfidf', 'keyword', 'vector', 'hybrid'], help='基础检索方法')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='最终返回 top-k')
    parser.add_argument('--index-dir', '-i', default='', help='向量索引目录')
    parser.add_argument('--embedding-model', '-e', default='BAAI/bge-m3', help='embedding 模型')
    parser.add_argument('--hybrid-weights', default='', help='hybrid 权重 JSON 字符串或 JSON 文件路径')
    parser.add_argument('--rewrite', action='store_true', help='启用多查询改写')
    parser.add_argument('--rewrite-max-queries', type=int, default=4, help='最大改写数')
    parser.add_argument('--rerank', action='store_true', help='启用 rerank')
    parser.add_argument('--reranker-model', default='', help='CrossEncoder / BGE reranker 模型名或本地路径；留空则使用启发式 rerank')
    parser.add_argument('--rerank-top-n', type=int, default=10, help='参与 rerank 的候选数')
    parser.add_argument('--rrf-k', type=int, default=DEFAULT_RRF_K, help='RRF 常数')
    parser.add_argument('--output', '-o', default='./data/processed/advanced_retrieved_context.json', help='输出文件')
    args = parser.parse_args()

    payload = advanced_retrieve(
        query=args.query,
        chunks_path=args.chunks,
        method=args.method,
        top_k=args.top_k,
        index_dir=args.index_dir or None,
        embedding_model=args.embedding_model,
        hybrid_weights=parse_hybrid_weights_arg(args.hybrid_weights),
        rewrite=args.rewrite,
        rewrite_max_queries=args.rewrite_max_queries,
        rerank=args.rerank,
        reranker_model=args.reranker_model,
        rerank_top_n=args.rerank_top_n,
        rrf_k=args.rrf_k,
    )
    save_results(payload, args.output)

    print(f"查询: {payload['query']}")
    print(f"基础方法: {payload['method']}")
    print(f"改写数: {len(payload['rewrites'])}")
    print(f"结果数: {payload['results_count']}")
    print(f"abstain: {payload['advanced']['should_abstain']} {payload['advanced']['abstain_reason']}")
    print(f"输出文件: {args.output}")
    for item in payload['results']:
        print(f"[{item['rank']}] score={item['score']:.4f} coverage={item.get('query_coverage', 0.0):.2f}")
        print(f"    {str(item['chunk'].get('text', ''))[:120]}...")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
