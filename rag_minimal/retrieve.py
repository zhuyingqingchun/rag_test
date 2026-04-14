#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础检索模块
使用 BM25 或 TF-IDF 在 chunk 中检索相关片段
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


def load_chunks(chunks_path: str) -> List[Dict[str, Any]]:
    """加载 chunk 数据"""
    chunks = []
    
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    
    return chunks


def tokenize(text: str) -> List[str]:
    """简单分词（按空格）"""
    return text.split()


def bm25_search(query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """使用 BM25 检索"""
    if not BM25_AVAILABLE:
        raise ImportError("rank-bm25 未安装，请运行: pip install rank-bm25")
    
    # 准备语料库
    corpus = [chunk['text'] for chunk in chunks]
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    
    # 初始化 BM25
    bm25 = BM25Okapi(tokenized_corpus)
    
    # 检索
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    
    # 排序并返回 top-k
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    results = []
    for idx in ranked_indices:
        if scores[idx] > 0:
            results.append({
                "chunk": chunks[idx],
                "score": scores[idx],
                "rank": len(results) + 1
            })
    
    return results


def tfidf_search(query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """使用 TF-IDF 检索（简单实现）"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # 准备语料库
    corpus = [chunk['text'] for chunk in chunks]
    
    # 初始化 TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # 转换查询
    query_vec = vectorizer.transform([query])
    
    # 计算相似度
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # 排序并返回 top-k
    ranked_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
    
    results = []
    for idx in ranked_indices:
        if similarities[idx] > 0:
            results.append({
                "chunk": chunks[idx],
                "score": float(similarities[idx]),
                "rank": len(results) + 1
            })
    
    return results


def keyword_search(query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """简单关键词匹配检索"""
    query_terms = set(query.lower().split())
    
    scores = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk['text'].lower()
        matching_terms = query_terms & set(chunk_text.split())
        score = len(matching_terms) / len(query_terms) if query_terms else 0
        scores.append((i, score))
    
    # 排序
    scores.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for idx, score in scores[:top_k]:
        if score > 0:
            results.append({
                "chunk": chunks[idx],
                "score": score,
                "rank": len(results) + 1
            })
    
    return results


def retrieve(query: str, chunks_path: str, method: str = 'bm25', top_k: int = 5) -> Dict[str, Any]:
    """检索主函数"""
    chunks = load_chunks(chunks_path)
    
    if method == 'bm25':
        if not BM25_AVAILABLE:
            print("警告: rank-bm25 未安装，使用 keyword_search")
            results = keyword_search(query, chunks, top_k)
        else:
            results = bm25_search(query, chunks, top_k)
    elif method == 'tfidf':
        results = tfidf_search(query, chunks, top_k)
    elif method == 'keyword':
        results = keyword_search(query, chunks, top_k)
    else:
        raise ValueError(f"不支持的检索方法: {method}")
    
    return {
        "query": query,
        "method": method,
        "total_chunks": len(chunks),
        "results_count": len(results),
        "results": results
    }


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """保存检索结果"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    output = {
        "query": results['query'],
        "method": results['method'],
        "total_chunks": results['total_chunks'],
        "results_count": results['results_count'],
        "timestamp": Path(results['query']).name if Path(results['query']).exists() else results['query'][:50],
        "results": [
            {
                "doc_id": r['chunk']['doc_id'],
                "chunk_id": r['chunk']['chunk_id'],
                "source": r['chunk']['source'],
                "text": r['chunk']['text'],
                "score": r['score'],
                "rank": r['rank']
            }
            for r in results['results']
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='基础检索模块')
    parser.add_argument('--query', '-q', required=True, help='查询语句')
    parser.add_argument('--chunks', '-c', default='./data/processed/docs_chunks.jsonl',
                        help='chunk 数据文件路径')
    parser.add_argument('--method', '-m', default='bm25',
                        choices=['bm25', 'tfidf', 'keyword'],
                        help='检索方法: bm25, tfidf, keyword')
    parser.add_argument('--top-k', '-k', type=int, default=5,
                        help='返回 top-k 个结果')
    parser.add_argument('--output', '-o', default='./data/processed/retrieved_context.json',
                        help='输出文件路径')
    
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


if __name__ == "__main__":
    sys.exit(main())
