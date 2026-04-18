#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""向量索引构建与查询模块。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


_MODEL_CACHE: Dict[str, Any] = {}


def load_chunks(chunks_path: str) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with open(chunks_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def _get_model(model_name: str):
    if model_name not in _MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                '未安装 sentence-transformers，请运行: pip install sentence-transformers'
            ) from exc
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def encode_texts(texts: List[str], model_name: str = 'BAAI/bge-m3', batch_size: int = 32) -> np.ndarray:
    model = _get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype='float32')


def build_index(chunks_path: str,
                output_dir: str,
                model_name: str = 'BAAI/bge-m3',
                batch_size: int = 32) -> Dict[str, Any]:
    chunks = load_chunks(chunks_path)
    if not chunks:
        raise ValueError('chunks 为空，无法构建向量索引')

    texts = [chunk.get('text', '') for chunk in chunks]
    embeddings = encode_texts(texts, model_name=model_name, batch_size=batch_size)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = [
        {
            'doc_id': chunk.get('doc_id'),
            'chunk_id': chunk.get('chunk_id'),
            'source': chunk.get('source'),
            'text': chunk.get('text', ''),
            'position': idx,
        }
        for idx, chunk in enumerate(chunks)
    ]

    config = {
        'model_name': model_name,
        'dimension': int(embeddings.shape[1]),
        'total_chunks': len(chunks),
        'backend': 'faiss' if FAISS_AVAILABLE else 'numpy',
    }

    with open(output_path / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with open(output_path / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    np.save(output_path / 'embeddings.npy', embeddings)

    if FAISS_AVAILABLE:
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype('float32'))
        faiss.write_index(index, str(output_path / 'index.faiss'))

    return {
        'output_dir': str(output_path),
        'total_chunks': len(chunks),
        'dimension': int(embeddings.shape[1]),
        'backend': config['backend'],
        'model_name': model_name,
    }


def load_index_bundle(index_dir: str) -> Dict[str, Any]:
    base = Path(index_dir)
    if not base.exists():
        raise FileNotFoundError(f'索引目录不存在: {index_dir}')

    config_path = base / 'config.json'
    metadata_path = base / 'metadata.json'
    embeddings_path = base / 'embeddings.npy'
    index_path = base / 'index.faiss'

    if not config_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f'索引目录缺少配置文件: {index_dir}')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    index = None
    embeddings = None
    if index_path.exists() and FAISS_AVAILABLE:
        index = faiss.read_index(str(index_path))
    elif embeddings_path.exists():
        embeddings = np.load(embeddings_path).astype('float32')
    else:
        raise FileNotFoundError(f'索引目录缺少 embeddings.npy: {index_dir}')

    return {
        'config': config,
        'metadata': metadata,
        'index': index,
        'embeddings': embeddings,
    }


def score_all(query: str, index_dir: str, model_name: str | None = None) -> Dict[str, Any]:
    bundle = load_index_bundle(index_dir)
    config = bundle['config']
    metadata = bundle['metadata']
    query_model = model_name or config.get('model_name', 'BAAI/bge-m3')
    query_embedding = encode_texts([query], model_name=query_model, batch_size=1)

    if bundle['index'] is not None:
        total = len(metadata)
        scores, indices = bundle['index'].search(query_embedding.astype('float32'), total)
        ordered_scores = [0.0] * total
        for pos, idx in enumerate(indices[0]):
            if int(idx) >= 0:
                ordered_scores[int(idx)] = float(scores[0][pos])
    else:
        embeddings = bundle['embeddings']
        ordered_scores = (embeddings @ query_embedding[0]).astype('float32').tolist()

    return {
        'scores': [float(score) for score in ordered_scores],
        'metadata': metadata,
        'config': config,
    }


def search_index(query: str,
                 index_dir: str,
                 top_k: int = 5,
                 model_name: str | None = None) -> List[Dict[str, Any]]:
    payload = score_all(query, index_dir=index_dir, model_name=model_name)
    scores = payload['scores']
    metadata = payload['metadata']

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results: List[Dict[str, Any]] = []
    for idx in ranked_indices:
        if scores[idx] > 0:
            results.append(
                {
                    'chunk': metadata[idx],
                    'score': float(scores[idx]),
                    'rank': len(results) + 1,
                }
            )
    return results


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description='向量索引构建与查询模块')
    subparsers = parser.add_subparsers(dest='command', required=True)

    build_parser = subparsers.add_parser('build', help='构建向量索引')
    build_parser.add_argument('--chunks', '-c', required=True, help='chunk 数据文件路径')
    build_parser.add_argument('--output', '-o', required=True, help='索引输出目录')
    build_parser.add_argument('--model', '-m', default='BAAI/bge-m3', help='embedding 模型名称')
    build_parser.add_argument('--batch-size', '-b', type=int, default=32, help='编码 batch size')

    query_parser = subparsers.add_parser('query', help='查询向量索引')
    query_parser.add_argument('--query', '-q', required=True, help='查询语句')
    query_parser.add_argument('--index-dir', '-i', required=True, help='索引目录')
    query_parser.add_argument('--model', '-m', default='', help='可选：覆盖索引里保存的 embedding 模型')
    query_parser.add_argument('--top-k', '-k', type=int, default=5, help='返回 top-k 个结果')

    args = parser.parse_args()

    if args.command == 'build':
        result = build_index(args.chunks, args.output, model_name=args.model, batch_size=args.batch_size)
        print(f"向量索引构建完成: {result['output_dir']}")
        print(f"总块数: {result['total_chunks']}")
        print(f"维度: {result['dimension']}")
        print(f"后端: {result['backend']}")
        print(f"模型: {result['model_name']}")
        return 0

    if args.command == 'query':
        results = search_index(
            args.query,
            index_dir=args.index_dir,
            top_k=args.top_k,
            model_name=args.model or None,
        )
        print(f'查询: {args.query}')
        print(f'命中数: {len(results)}')
        for item in results:
            print(f"[{item['rank']}] score={item['score']:.4f}")
            print(f"    {item['chunk']['text'][:120]}...")
        return 0

    return 1


if __name__ == '__main__':
    sys.exit(main())
