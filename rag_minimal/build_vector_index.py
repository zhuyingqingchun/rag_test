#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建向量索引的便捷脚本。"""
from __future__ import annotations

import sys

from vector_store import build_index


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description='构建向量索引')
    parser.add_argument('--chunks', '-c', default='./data/processed/docs_chunks.jsonl', help='chunk 数据文件路径')
    parser.add_argument('--output', '-o', default='./data/processed/vector_index', help='索引输出目录')
    parser.add_argument('--model', '-m', default='BAAI/bge-m3', help='embedding 模型名称')
    parser.add_argument('--batch-size', '-b', type=int, default=32, help='编码 batch size')
    args = parser.parse_args()

    result = build_index(args.chunks, args.output, model_name=args.model, batch_size=args.batch_size)
    print(f"向量索引构建完成: {result['output_dir']}")
    print(f"总块数: {result['total_chunks']}")
    print(f"维度: {result['dimension']}")
    print(f"后端: {result['backend']}")
    print(f"模型: {result['model_name']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
