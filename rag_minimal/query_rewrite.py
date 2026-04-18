#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""查询改写：为 RAG / Transformer / BERT / GPT / 舵机诊断等主题生成多查询候选。"""
from __future__ import annotations

from typing import List


def _dedupe_keep_order(values: List[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for value in values:
        normalized = ' '.join((value or '').split()).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(normalized)
    return output


def generate_rewrites(query: str, max_queries: int = 4) -> List[str]:
    query = ' '.join((query or '').split()).strip()
    if not query:
        return []

    candidates: List[str] = [query]
    lower = query.lower()

    if 'rag' in lower or '检索增强' in query or '检索增强生成' in query:
        candidates.extend([
            f'{query} 检索 生成 上下文 流程',
            f'{query} retrieval generation context pipeline',
            'RAG 核心组成 检索 生成 文档 上下文',
        ])

    if 'transformer' in lower:
        candidates.extend([
            f'{query} self-attention multi-head encoder decoder',
            'Transformer 核心机制 self-attention multi-head positional encoding',
            f'{query} 并行建模 长距离依赖',
        ])

    if 'bert' in lower and 'gpt' in lower:
        candidates.extend([
            'BERT 与 GPT 区别 bidirectional autoregressive masked language model causal language model',
            f'{query} 预训练目标 双向 自回归',
        ])
    elif 'bert' in lower:
        candidates.extend([
            f'{query} bidirectional masked language model',
            'BERT 核心特点 bidirectional masked language model',
        ])
    elif 'gpt' in lower:
        candidates.extend([
            f'{query} autoregressive causal language model',
            'GPT 核心特点 autoregressive causal language model',
        ])

    if 'gpt-2' in lower or 'gpt2' in lower or 'gpt-3' in lower or 'gpt3' in lower:
        candidates.extend([
            f'{query} scaling parameters few-shot',
            'GPT-2 GPT-3 参数规模 few-shot 能力 区别',
        ])

    if '算法' in query or '复杂度' in query or 'hello-algo' in lower:
        candidates.extend([
            f'{query} 时间复杂度 空间复杂度 数据结构',
            '算法教程 时间复杂度 空间复杂度 数据结构',
        ])

    if 'signal' in lower or '信号' in query or '舵机' in query or '故障诊断' in query:
        candidates.extend([
            f'{query} 信号 模态 检索 诊断',
            '电动舵机 故障诊断 人工智能 信号 模态',
            'signal rag guide 信号处理 RAG 指南',
        ])

    candidates.extend([
        f'{query} 核心组成',
        f'{query} 关键机制',
        f'{query} 基本流程',
    ])

    return _dedupe_keep_order(candidates)[:max(1, max_queries)]


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description='生成多查询改写候选')
    parser.add_argument('--query', '-q', required=True, help='原始问题')
    parser.add_argument('--max-queries', '-k', type=int, default=4, help='最大候选数')
    args = parser.parse_args()

    rewrites = generate_rewrites(args.query, max_queries=args.max_queries)
    print(json.dumps({'query': args.query, 'rewrites': rewrites}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
