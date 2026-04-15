#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""完整流程演示脚本。"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    cmd = [c for c in cmd if c]
    print(f"\n{'=' * 60}")
    print(description)
    print(f"命令: {' '.join(cmd)}")
    print('=' * 60)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
        if result.stderr:
            print(f"[警告] {result.stderr}")
        return True
    print('[错误] 命令执行失败')
    if result.stderr:
        print(result.stderr)
    return False


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description='完整流程演示：导入 → 切分 → 检索 → 生成报告')
    parser.add_argument('--query', '-q', required=True, help='用户查询问题')
    parser.add_argument('--input', '-i', default='./data/input_docs', help='输入文档目录')
    parser.add_argument('--model', '-m', default='next80b_fp8', help='Qwen 模型名称')
    parser.add_argument('--base-url', '-b', default='http://127.0.0.1:8000/v1', help='Qwen 服务地址')
    parser.add_argument('--chunk-method', '-c', default='paragraph', choices=['paragraph', 'char', 'sentence'], help='切分方法')
    parser.add_argument('--retrieve-method', '-r', default='bm25', choices=['bm25', 'tfidf', 'keyword'], help='检索方法')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='检索 top-k 结果')
    parser.add_argument('--save-run', '-s', action='store_true', help='保存运行记录')
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data'
    processed_dir = data_dir / 'processed'
    runs_dir = data_dir / 'runs'
    processed_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    ingest_cmd = [sys.executable, str(base_dir / 'ingest.py'), '--input', args.input, '--output', str(processed_dir / 'docs_raw.jsonl')]
    if not run_command(ingest_cmd, '文档导入'):
        return 1

    chunk_cmd = [sys.executable, str(base_dir / 'chunk.py'), '--input', str(processed_dir / 'docs_raw.jsonl'), '--output', str(processed_dir / 'docs_chunks.jsonl'), '--method', args.chunk_method]
    if not run_command(chunk_cmd, '文本切分'):
        return 1

    retrieve_cmd = [sys.executable, str(base_dir / 'retrieve.py'), '--query', args.query, '--chunks', str(processed_dir / 'docs_chunks.jsonl'), '--method', args.retrieve_method, '--top-k', str(args.top_k), '--output', str(processed_dir / 'retrieved_context.json')]
    if not run_command(retrieve_cmd, '检索相关片段'):
        return 1

    report_cmd = [sys.executable, str(base_dir / 'generate_report.py'), '--query', args.query, '--context', str(processed_dir / 'retrieved_context.json'), '--model', args.model, '--base-url', args.base_url, '--runs-dir', str(runs_dir)]
    if args.save_run:
        report_cmd.append('--save-run')
    if not run_command(report_cmd, '生成结构化报告'):
        return 1

    print("\n" + '=' * 60)
    print('流程完成!')
    print('=' * 60)
    print(f"\n查询: {args.query}")
    print(f"输入目录: {args.input}")
    print(f"切分方法: {args.chunk_method}")
    print(f"检索方法: {args.retrieve_method}")
    print(f"top-k: {args.top_k}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
