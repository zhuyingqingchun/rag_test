#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""结果重排序：优先支持 CrossEncoder，本地不可用时回退到启发式 rerank。"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from text_utils import unique_tokens

_CROSS_ENCODER_CACHE: Dict[str, Any] = {}


def _keyword_overlap_ratio(query: str, text: str) -> float:
    q = unique_tokens(query)
    if not q:
        return 0.0
    d = unique_tokens(text)
    return len(q & d) / len(q)


def _normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi == lo:
        return [1.0 if value > 0 else 0.0 for value in values]
    return [(value - lo) / (hi - lo) for value in values]


def _load_cross_encoder(model_name: str):
    if not model_name:
        return None
    if model_name not in _CROSS_ENCODER_CACHE:
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                'CrossEncoder 不可用，请安装 sentence-transformers 或留空 reranker model 使用启发式 rerank'
            ) from exc
        _CROSS_ENCODER_CACHE[model_name] = CrossEncoder(model_name)
    return _CROSS_ENCODER_CACHE[model_name]


def heuristic_rerank(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    base_scores = [float(item.get('score', 0.0)) for item in results]
    base_norm = _normalize(base_scores)
    overlap_scores = [
        _keyword_overlap_ratio(query, str(item['chunk'].get('text', '')))
        for item in results
    ]
    combined: List[Tuple[float, Dict[str, Any]]] = []
    for idx, item in enumerate(results):
        rerank_score = 0.6 * base_norm[idx] + 0.4 * overlap_scores[idx]
        enriched = dict(item)
        enriched['base_score'] = float(item.get('score', 0.0))
        enriched['rerank_score'] = round(rerank_score, 6)
        enriched['score'] = round(rerank_score, 6)
        combined.append((rerank_score, enriched))
    combined.sort(key=lambda x: x[0], reverse=True)
    output = []
    for rank, (_, item) in enumerate(combined, start=1):
        item['rank'] = rank
        output.append(item)
    return output


def cross_encoder_rerank(query: str, results: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
    model = _load_cross_encoder(model_name)
    pairs = [(query, str(item['chunk'].get('text', ''))) for item in results]
    scores = model.predict(pairs)
    score_list = [float(score) for score in scores]
    score_norm = _normalize(score_list)
    combined: List[Tuple[float, Dict[str, Any]]] = []
    for idx, item in enumerate(results):
        enriched = dict(item)
        enriched['base_score'] = float(item.get('score', 0.0))
        enriched['rerank_score'] = round(score_norm[idx], 6)
        enriched['score'] = round(score_norm[idx], 6)
        combined.append((score_norm[idx], enriched))
    combined.sort(key=lambda x: x[0], reverse=True)
    output = []
    for rank, (_, item) in enumerate(combined, start=1):
        item['rank'] = rank
        output.append(item)
    return output


def rerank_results(query: str,
                   results: List[Dict[str, Any]],
                   model_name: str = '',
                   top_n: int = 10) -> List[Dict[str, Any]]:
    candidates = list(results)[:max(1, top_n)]
    if not candidates:
        return []

    if model_name:
        try:
            return cross_encoder_rerank(query, candidates, model_name)
        except Exception:
            pass
    return heuristic_rerank(query, candidates)
