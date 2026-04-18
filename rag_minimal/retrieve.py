#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基础检索模块。"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

from text_utils import tokenize, unique_tokens


DEFAULT_HYBRID_WEIGHTS = {
    'bm25': 0.5,
    'tfidf': 0.3,
    'keyword': 0.2,
}


def load_chunks(chunks_path: str) -> List[Dict[str, Any]]:
    chunks = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def _minmax_normalize(scores: List[float]) -> List[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 if score > 0 else 0.0 for score in scores]
    return [(score - min_score) / (max_score - min_score) for score in scores]


def bm25_score_list(query: str, chunks: List[Dict[str, Any]]) -> List[float]:
    if not BM25_AVAILABLE:
        raise ImportError("rank-bm25 未安装，请运行: pip install rank-bm25")
    tokenized_corpus = [tokenize(chunk['text']) for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = tokenize(query)
    return [float(score) for score in bm25.get_scores(query_tokens)]


def tfidf_score_list(query: str, chunks: List[Dict[str, Any]]) -> List[float]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    corpus = [' '.join(tokenize(chunk['text'])) for chunk in chunks]
    vectorizer = TfidfVectorizer(token_pattern=r'[^ ]+')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vec = vectorizer.transform([' '.join(tokenize(query))])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    return [float(score) for score in similarities]


def keyword_score_list(query: str, chunks: List[Dict[str, Any]]) -> List[float]:
    query_terms = unique_tokens(query)
    scores: List[float] = []
    for chunk in chunks:
        chunk_terms = unique_tokens(chunk['text'])
        overlap = query_terms & chunk_terms
        score = len(overlap) / len(query_terms) if query_terms else 0.0
        scores.append(float(score))
    return scores


def _topk_results_from_scores(chunks: List[Dict[str, Any]],
                              scores: List[float],
                              top_k: int = 5,
                              score_breakdowns: Dict[int, Dict[str, float]] | None = None) -> List[Dict[str, Any]]:
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for idx in ranked_indices:
        if scores[idx] > 0:
            item = {
                "chunk": chunks[idx],
                "score": float(scores[idx]),
                "rank": len(results) + 1,
            }
            if score_breakdowns is not None and idx in score_breakdowns:
                item["score_breakdown"] = score_breakdowns[idx]
            results.append(item)
    return results


def bm25_search(query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    scores = bm25_score_list(query, chunks)
    return _topk_results_from_scores(chunks, scores, top_k)


def tfidf_search(query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    scores = tfidf_score_list(query, chunks)
    return _topk_results_from_scores(chunks, scores, top_k)


def keyword_search(query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    scores = keyword_score_list(query, chunks)
    return _topk_results_from_scores(chunks, scores, top_k)


def hybrid_search(query: str,
                  chunks: List[Dict[str, Any]],
                  top_k: int = 5,
                  weights: Dict[str, float] | None = None) -> List[Dict[str, Any]]:
    if weights is None:
        weights = dict(DEFAULT_HYBRID_WEIGHTS)

    active_methods: Dict[str, List[float]] = {}
    if BM25_AVAILABLE:
        active_methods['bm25'] = bm25_score_list(query, chunks)
    else:
        active_methods['keyword'] = keyword_score_list(query, chunks)

    active_methods['tfidf'] = tfidf_score_list(query, chunks)
    active_methods['keyword'] = keyword_score_list(query, chunks)

    normalized_scores = {
        name: _minmax_normalize(score_list)
        for name, score_list in active_methods.items()
    }

    total_weight = sum(weights.get(name, 0.0) for name in normalized_scores.keys())
    if total_weight <= 0:
        total_weight = float(len(normalized_scores))

    fused_scores: List[float] = []
    score_breakdowns: Dict[int, Dict[str, float]] = {}
    for idx in range(len(chunks)):
        breakdown = {}
        fused_score = 0.0
        for name, score_list in normalized_scores.items():
            method_weight = weights.get(name, 0.0)
            if method_weight <= 0:
                continue
            contribution = (method_weight / total_weight) * score_list[idx]
            breakdown[name] = round(contribution, 6)
            fused_score += contribution
        fused_scores.append(float(fused_score))
        score_breakdowns[idx] = breakdown

    return _topk_results_from_scores(chunks, fused_scores, top_k, score_breakdowns)


def retrieve(query: str, chunks_path: str, method: str = 'bm25', top_k: int = 5) -> Dict[str, Any]:
    chunks = load_chunks(chunks_path)
    if method == 'bm25':
        results = bm25_search(query, chunks, top_k) if BM25_AVAILABLE else keyword_search(query, chunks, top_k)
    elif method == 'tfidf':
        results = tfidf_search(query, chunks, top_k)
    elif method == 'keyword':
        results = keyword_search(query, chunks, top_k)
    elif method == 'hybrid':
        results = hybrid_search(query, chunks, top_k)
    else:
        raise ValueError(f"不支持的检索方法: {method}")

    output = {
        "query": query,
        "method": method,
        "total_chunks": len(chunks),
        "results_count": len(results),
        "results": results,
    }
    if method == 'hybrid':
        output['hybrid_weights'] = dict(DEFAULT_HYBRID_WEIGHTS)
    return output


def save_results(results: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = {
        "query": results['query'],
        "method": results['method'],
        "total_chunks": results['total_chunks'],
        "results_count": results['results_count'],
        "timestamp": datetime.now().isoformat(),
        "results": [
            {
                "doc_id": r['chunk']['doc_id'],
                "chunk_id": r['chunk']['chunk_id'],
                "source": r['chunk']['source'],
                "text": r['chunk']['text'],
                "score": r['score'],
                "rank": r['rank'],
                **({"score_breakdown": r['score_breakdown']} if 'score_breakdown' in r else {}),
            }
            for r in results['results']
        ],
    }
    if 'hybrid_weights' in results:
        output['hybrid_weights'] = results['hybrid_weights']
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description='基础检索模块')
    parser.add_argument('--query', '-q', required=True, help='查询语句')
    parser.add_argument('--chunks', '-c', default='./data/processed/docs_chunks.jsonl', help='chunk 数据文件路径')
    parser.add_argument('--method', '-m', default='bm25', choices=['bm25', 'tfidf', 'keyword', 'hybrid'], help='检索方法')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='返回 top-k 个结果')
    parser.add_argument('--output', '-o', default='./data/processed/retrieved_context.json', help='输出文件路径')
    args = parser.parse_args()

    results = retrieve(args.query, args.chunks, args.method, args.top_k)
    print(f"查询: {results['query']}")
    print(f"检索方法: {results['method']}")
    print(f"总块数: {results['total_chunks']}")
    print(f"召回块数: {results['results_count']}")
    if results['results']:
        print("\n检索结果:")
        for r in results['results']:
            print(f"  [{r['rank']}] (score: {r['score']:.4f})")
            print(f"      {r['chunk']['text'][:100]}...")
    save_results(results, args.output)
    print(f"\n结果已保存到: {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
