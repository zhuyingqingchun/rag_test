#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫描 hybrid 权重，比较效果并观察对默认权重的稳定性。"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple

from evaluate_retrieval import evaluate_method, load_eval_cases
from retrieve import DEFAULT_HYBRID_WEIGHTS, normalize_hybrid_weights, retrieve


PRESET_WEIGHT_CANDIDATES: list[dict[str, Any]] = [
    {
        'name': 'default',
        'weights': dict(DEFAULT_HYBRID_WEIGHTS),
    },
    {
        'name': 'vector_heavy',
        'weights': {'bm25': 0.25, 'tfidf': 0.10, 'keyword': 0.05, 'vector': 0.60},
    },
    {
        'name': 'bm25_vector_balanced',
        'weights': {'bm25': 0.50, 'tfidf': 0.00, 'keyword': 0.00, 'vector': 0.50},
    },
    {
        'name': 'sparse_friendly',
        'weights': {'bm25': 0.45, 'tfidf': 0.20, 'keyword': 0.10, 'vector': 0.25},
    },
    {
        'name': 'vector_plus_bm25',
        'weights': {'bm25': 0.40, 'tfidf': 0.05, 'keyword': 0.05, 'vector': 0.50},
    },
]


def _load_extra_candidates(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError('额外权重文件必须是 JSON 数组或单个 JSON 对象')

    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError('额外权重候选必须是对象')
        name = str(item.get('name') or f'custom_{idx + 1}')
        weights = item.get('weights', item)
        if not isinstance(weights, dict):
            raise ValueError(f'候选 {name} 的 weights 必须是对象')
        normalized.append(
            {
                'name': name,
                'weights': normalize_hybrid_weights({str(k): float(v) for k, v in weights.items()}),
            }
        )
    return normalized


def _pairs_from_results(results: list[dict[str, Any]]) -> set[Tuple[str, str]]:
    return {
        (str(item['chunk'].get('doc_id', '')), str(item['chunk'].get('chunk_id', '')))
        for item in results
    }


def _jaccard(a: set[Tuple[str, str]], b: set[Tuple[str, str]]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _stability_against_default(cases: list[dict[str, Any]],
                               chunks_path: str,
                               top_k: int,
                               index_dir: str | None,
                               embedding_model: str,
                               candidate_weights: dict[str, float]) -> float:
    default_weights = dict(DEFAULT_HYBRID_WEIGHTS)
    scores: list[float] = []
    for case in cases:
        query = str(case.get('query') or case.get('question') or case.get('instruction') or '').strip()
        if not query:
            continue
        default_results = retrieve(
            query=query,
            chunks_path=chunks_path,
            method='hybrid',
            top_k=top_k,
            index_dir=index_dir,
            embedding_model=embedding_model,
            hybrid_weights=default_weights,
        )['results']
        candidate_results = retrieve(
            query=query,
            chunks_path=chunks_path,
            method='hybrid',
            top_k=top_k,
            index_dir=index_dir,
            embedding_model=embedding_model,
            hybrid_weights=candidate_weights,
        )['results']
        scores.append(_jaccard(_pairs_from_results(default_results), _pairs_from_results(candidate_results)))
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 6)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description='扫描 hybrid 权重，并评估效果与稳定性')
    parser.add_argument('--eval-file', required=True, help='评测样本 JSON / JSONL 文件')
    parser.add_argument('--chunks', required=True, help='chunk 数据文件路径')
    parser.add_argument('--top-k', type=int, default=5, help='评测 top-k')
    parser.add_argument('--index-dir', default='', help='向量索引目录')
    parser.add_argument('--embedding-model', default='BAAI/bge-m3', help='embedding 模型')
    parser.add_argument('--extra-candidates', default='', help='额外候选权重 JSON 文件')
    parser.add_argument('--output', default='./data/processed/hybrid_weight_scan.json', help='扫描结果输出路径')
    args = parser.parse_args()

    cases = load_eval_cases(args.eval_file)
    candidates = list(PRESET_WEIGHT_CANDIDATES)
    candidates.extend(_load_extra_candidates(args.extra_candidates or None))

    leaderboard: list[dict[str, Any]] = []
    for candidate in candidates:
        name = candidate['name']
        weights = normalize_hybrid_weights(candidate['weights'])
        report = evaluate_method(
            cases=cases,
            chunks_path=args.chunks,
            method='hybrid',
            top_k=args.top_k,
            index_dir=args.index_dir or None,
            embedding_model=args.embedding_model,
            hybrid_weights=weights,
        )
        summary = report['summary']
        stability = _stability_against_default(
            cases=cases,
            chunks_path=args.chunks,
            top_k=args.top_k,
            index_dir=args.index_dir or None,
            embedding_model=args.embedding_model,
            candidate_weights=weights,
        )
        score = round(
            summary['recall_at_k'] * 0.5 +
            summary['mrr_at_k'] * 0.3 +
            summary['precision_at_k'] * 0.2,
            6,
        )
        leaderboard.append(
            {
                'name': name,
                'weights': weights,
                'summary': summary,
                'score': score,
                'overlap_vs_default_at_k': stability,
            }
        )

    leaderboard.sort(key=lambda item: (item['score'], item['summary']['recall_at_k'], item['summary']['mrr_at_k']), reverse=True)

    output = {
        'timestamp': datetime.now().isoformat(),
        'eval_file': os.path.abspath(args.eval_file),
        'chunks': os.path.abspath(args.chunks),
        'index_dir': os.path.abspath(args.index_dir) if args.index_dir else '',
        'embedding_model': args.embedding_model,
        'top_k': args.top_k,
        'leaderboard': leaderboard,
        'best': leaderboard[0] if leaderboard else None,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'评测样本数: {len(cases)}')
    print(f'输出文件: {output_path}')
    if leaderboard:
        best = leaderboard[0]
        print(
            f"最优候选: {best['name']} "
            f"score={best['score']:.4f} "
            f"recall@{args.top_k}={best['summary']['recall_at_k']:.4f} "
            f"mrr@{args.top_k}={best['summary']['mrr_at_k']:.4f} "
            f"overlap_vs_default={best['overlap_vs_default_at_k']:.4f}"
        )
        print('候选排名:')
        for item in leaderboard:
            print(
                f"  - {item['name']}: score={item['score']:.4f}, "
                f"recall={item['summary']['recall_at_k']:.4f}, "
                f"mrr={item['summary']['mrr_at_k']:.4f}, "
                f"overlap={item['overlap_vs_default_at_k']:.4f}"
            )
    return 0


if __name__ == '__main__':
    sys.exit(main())
