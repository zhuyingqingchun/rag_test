#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线检索评测：比较 bm25 / vector / hybrid 的效果。"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from retrieve import DEFAULT_HYBRID_WEIGHTS, parse_hybrid_weights_arg, retrieve


@dataclass
class GoldSpec:
    pairs: set[Tuple[str, str]]
    doc_ids: set[str]
    chunk_ids: set[str]
    substrings: list[str]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def load_eval_cases(path: str) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f'评测文件不存在: {path}')

    raw = source.read_text(encoding='utf-8').strip()
    if not raw:
        raise ValueError(f'评测文件为空: {path}')

    if source.suffix.lower() == '.jsonl':
        cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        payload = json.loads(raw)
        if isinstance(payload, list):
            cases = payload
        elif isinstance(payload, dict):
            for key in ('cases', 'examples', 'data', 'items'):
                if isinstance(payload.get(key), list):
                    cases = payload[key]
                    break
            else:
                raise ValueError('JSON 评测文件需要是数组，或包含 cases/examples/data/items 字段')
        else:
            raise ValueError('无法解析评测文件格式')

    if not cases:
        raise ValueError('评测样本数为 0')
    return cases


def _extract_query(case: dict[str, Any]) -> str:
    for key in ('query', 'question', 'instruction'):
        value = str(case.get(key, '')).strip()
        if value:
            return value
    raise ValueError(f'评测样本缺少 query/question 字段: {case}')


def _normalize_pairs(values: Iterable[Any]) -> set[Tuple[str, str]]:
    pairs: set[Tuple[str, str]] = set()
    for item in values:
        if isinstance(item, str):
            normalized = item.replace('#', ':').replace('/', ':')
            if ':' in normalized:
                left, right = normalized.split(':', 1)
                pairs.add((left.strip(), right.strip()))
        elif isinstance(item, dict):
            doc_id = str(item.get('doc_id', '')).strip()
            chunk_id = str(item.get('chunk_id', '')).strip()
            if doc_id and chunk_id:
                pairs.add((doc_id, chunk_id))
    return pairs


def build_gold_spec(case: dict[str, Any]) -> GoldSpec:
    pair_values: list[Any] = []
    for key in ('relevant_pairs', 'positive_pairs', 'gold_pairs'):
        pair_values.extend(_as_list(case.get(key)))

    doc_values: list[Any] = []
    for key in ('relevant_doc_ids', 'positive_doc_ids', 'doc_ids'):
        doc_values.extend(_as_list(case.get(key)))

    chunk_values: list[Any] = []
    for key in ('relevant_chunk_ids', 'positive_chunk_ids', 'chunk_ids'):
        chunk_values.extend(_as_list(case.get(key)))

    substrings: list[str] = []
    for key in ('expected_substrings', 'expected_keywords', 'keywords'):
        substrings.extend([str(x).strip() for x in _as_list(case.get(key)) if str(x).strip()])

    spec = GoldSpec(
        pairs=_normalize_pairs(pair_values),
        doc_ids={str(x).strip() for x in doc_values if str(x).strip()},
        chunk_ids={str(x).strip() for x in chunk_values if str(x).strip()},
        substrings=substrings,
    )
    if not any((spec.pairs, spec.doc_ids, spec.chunk_ids, spec.substrings)):
        raise ValueError(
            '评测样本缺少 gold 标注；至少提供 relevant_pairs / relevant_doc_ids / '
            'relevant_chunk_ids / expected_substrings 之一'
        )
    return spec


def _is_relevant(result: dict[str, Any], gold: GoldSpec) -> bool:
    chunk = result['chunk']
    doc_id = str(chunk.get('doc_id', '')).strip()
    chunk_id = str(chunk.get('chunk_id', '')).strip()
    text = str(chunk.get('text', ''))

    if gold.pairs:
        return (doc_id, chunk_id) in gold.pairs
    if gold.chunk_ids:
        return chunk_id in gold.chunk_ids
    if gold.doc_ids:
        return doc_id in gold.doc_ids
    return any(fragment in text for fragment in gold.substrings)


def _gold_size(gold: GoldSpec) -> int:
    if gold.pairs:
        return len(gold.pairs)
    if gold.chunk_ids:
        return len(gold.chunk_ids)
    if gold.doc_ids:
        return len(gold.doc_ids)
    return 1


def evaluate_single_case(case: dict[str, Any],
                         chunks_path: str,
                         method: str,
                         top_k: int,
                         index_dir: str | None,
                         embedding_model: str,
                         hybrid_weights: dict[str, float] | None = None) -> dict[str, Any]:
    query = _extract_query(case)
    gold = build_gold_spec(case)
    payload = retrieve(
        query=query,
        chunks_path=chunks_path,
        method=method,
        top_k=top_k,
        index_dir=index_dir,
        embedding_model=embedding_model,
        hybrid_weights=hybrid_weights,
    )

    matched_ranks: list[int] = []
    matched_items: list[dict[str, Any]] = []
    for item in payload['results']:
        if _is_relevant(item, gold):
            matched_ranks.append(int(item['rank']))
            matched_items.append(
                {
                    'rank': int(item['rank']),
                    'score': float(item['score']),
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

    return {
        'case_id': case.get('id') or case.get('case_id') or query,
        'query': query,
        'method': method,
        'top_k': top_k,
        'hit_at_k': round(hit_at_k, 6),
        'precision_at_k': round(precision_at_k, 6),
        'recall_at_k': round(recall_at_k, 6),
        'mrr_at_k': round(mrr_at_k, 6),
        'first_relevant_rank': first_relevant_rank,
        'matched_ranks': matched_ranks,
        'matched_items': matched_items,
        'results': [
            {
                'rank': int(item['rank']),
                'score': float(item['score']),
                'doc_id': item['chunk'].get('doc_id'),
                'chunk_id': item['chunk'].get('chunk_id'),
                'source': item['chunk'].get('source'),
                'text_preview': str(item['chunk'].get('text', ''))[:120],
            }
            for item in payload['results']
        ],
        'hybrid_weights': payload.get('hybrid_weights'),
    }


def summarize_case_results(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(case_results)
    if count == 0:
        raise ValueError('没有评测结果可汇总')

    def _mean(key: str) -> float:
        return round(sum(float(item[key]) for item in case_results) / count, 6)

    ranks = [item['first_relevant_rank'] for item in case_results if item['first_relevant_rank']]
    return {
        'queries': count,
        'hit_at_k': _mean('hit_at_k'),
        'precision_at_k': _mean('precision_at_k'),
        'recall_at_k': _mean('recall_at_k'),
        'mrr_at_k': _mean('mrr_at_k'),
        'avg_first_relevant_rank': round(sum(ranks) / len(ranks), 6) if ranks else None,
    }


def evaluate_method(cases: list[dict[str, Any]],
                    chunks_path: str,
                    method: str,
                    top_k: int,
                    index_dir: str | None,
                    embedding_model: str,
                    hybrid_weights: dict[str, float] | None = None) -> dict[str, Any]:
    case_results = [
        evaluate_single_case(
            case=case,
            chunks_path=chunks_path,
            method=method,
            top_k=top_k,
            index_dir=index_dir,
            embedding_model=embedding_model,
            hybrid_weights=hybrid_weights,
        )
        for case in cases
    ]
    return {
        'method': method,
        'top_k': top_k,
        'summary': summarize_case_results(case_results),
        'hybrid_weights': hybrid_weights,
        'cases': case_results,
    }


def evaluate_methods(cases: list[dict[str, Any]],
                     chunks_path: str,
                     methods: list[str],
                     top_k: int,
                     index_dir: str | None,
                     embedding_model: str,
                     hybrid_weights: dict[str, float] | None = None) -> dict[str, Any]:
    reports = {}
    for method in methods:
        report = evaluate_method(
            cases=cases,
            chunks_path=chunks_path,
            method=method,
            top_k=top_k,
            index_dir=index_dir,
            embedding_model=embedding_model,
            hybrid_weights=hybrid_weights if method == 'hybrid' else None,
        )
        reports[method] = report
    return reports


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description='离线检索评测：比较 bm25 / vector / hybrid')
    parser.add_argument('--eval-file', required=True, help='评测样本 JSON / JSONL 文件')
    parser.add_argument('--chunks', required=True, help='chunk 数据文件路径')
    parser.add_argument('--methods', default='bm25,vector,hybrid', help='逗号分隔的方法列表')
    parser.add_argument('--top-k', type=int, default=5, help='使用 top-k 结果计算指标')
    parser.add_argument('--index-dir', default='', help='向量索引目录')
    parser.add_argument('--embedding-model', default='BAAI/bge-m3', help='embedding 模型')
    parser.add_argument('--hybrid-weights', default='', help='hybrid 权重 JSON 字符串或 JSON 文件路径')
    parser.add_argument('--output', default='./data/processed/retrieval_eval_report.json', help='输出报告路径')
    args = parser.parse_args()

    cases = load_eval_cases(args.eval_file)
    methods = [item.strip() for item in args.methods.split(',') if item.strip()]
    hybrid_weights = parse_hybrid_weights_arg(args.hybrid_weights) or dict(DEFAULT_HYBRID_WEIGHTS)

    reports = evaluate_methods(
        cases=cases,
        chunks_path=args.chunks,
        methods=methods,
        top_k=args.top_k,
        index_dir=args.index_dir or None,
        embedding_model=args.embedding_model,
        hybrid_weights=hybrid_weights,
    )

    output = {
        'timestamp': datetime.now().isoformat(),
        'eval_file': os.path.abspath(args.eval_file),
        'chunks': os.path.abspath(args.chunks),
        'index_dir': os.path.abspath(args.index_dir) if args.index_dir else '',
        'embedding_model': args.embedding_model,
        'top_k': args.top_k,
        'methods': methods,
        'reports': reports,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'评测样本数: {len(cases)}')
    print(f'输出报告: {output_path}')
    for method in methods:
        summary = reports[method]['summary']
        print(
            f"[{method}] hit@{args.top_k}={summary['hit_at_k']:.4f} "
            f"precision@{args.top_k}={summary['precision_at_k']:.4f} "
            f"recall@{args.top_k}={summary['recall_at_k']:.4f} "
            f"mrr@{args.top_k}={summary['mrr_at_k']:.4f}"
        )
    return 0


if __name__ == '__main__':
    sys.exit(main())
