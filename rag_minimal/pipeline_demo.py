#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整流程演示脚本
一条命令完成：导入 → 切分 → 检索 → 生成报告
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


def run_command(cmd: list, description: str) -> bool:
    """运行命令并检查结果"""
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"命令: {' '.join(cmd)}")
    print('=' * 60)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
        if result.stderr:
            print(f"[警告] {result.stderr}")
        return True
    else:
        print(f"[错误] 命令执行失败")
        if result.stderr:
            print(result.stderr)
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='完整流程演示：导入 → 切分 → 检索 → 生成报告'
    )
    parser.add_argument('--query', '-q', required=True, 
                        help='用户查询问题')
    parser.add_argument('--input', '-i', default='./data/input_docs',
                        help='输入文档目录')
    parser.add_argument('--model', '-m', default='next80b_fp8',
                        help='Qwen 模型名称')
    parser.add_argument('--base-url', '-b', default='http://127.0.0.1:8000/v1',
                        help='Qwen 服务地址')
    parser.add_argument('--chunk-method', '-c', default='paragraph',
                        choices=['paragraph', 'char', 'sentence'],
                        help='切分方法')
    parser.add_argument('--retrieve-method', '-r', default='bm25',
                        choices=['bm25', 'tfidf', 'keyword'],
                        help='检索方法')
    parser.add_argument('--top-k', '-k', type=int, default=5,
                        help='检索 top-k 结果')
    parser.add_argument('--save-run', '-s', action='store_true',
                        help='保存运行记录')
    
    args = parser.parse_args()
    
    # 设置路径
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data'
    processed_dir = data_dir / 'processed'
    runs_dir = data_dir / 'runs'
    
    # 创建目录
    processed_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    # 步骤 1: 文档导入
    print("\n" + "=" * 60)
    print("步骤 1: 文档导入")
    print("=" * 60)
    
    ingest_cmd = [
        sys.executable, str(base_dir / 'ingest.py'),
        '--input', args.input,
        '--output', str(processed_dir / 'docs_raw.jsonl')
    ]
    
    if not run_command(ingest_cmd, "文档导入"):
        return 1
    
    # 步骤 2: 文本切分
    print("\n" + "=" * 60)
    print("步骤 2: 文本切分")
    print("=" * 60)
    
    chunk_cmd = [
        sys.executable, str(base_dir / 'chunk.py'),
        '--input', str(processed_dir / 'docs_raw.jsonl'),
        '--output', str(processed_dir / 'docs_chunks.jsonl'),
        '--method', args.chunk_method
    ]
    
    if not run_command(chunk_cmd, "文本切分"):
        return 1
    
    # 步骤 3: 检索
    print("\n" + "=" * 60)
    print("步骤 3: 检索")
    print("=" * 60)
    
    retrieve_cmd = [
        sys.executable, str(base_dir / 'retrieve.py'),
        '--query', args.query,
        '--chunks', str(processed_dir / 'docs_chunks.jsonl'),
        '--method', args.retrieve_method,
        '--top-k', str(args.top_k),
        '--output', str(processed_dir / 'retrieved_context.json')
    ]
    
    if not run_command(retrieve_cmd, "检索相关片段"):
        return 1
    
    # 步骤 4: 生成报告
    print("\n" + "=" * 60)
    print("步骤 4: 生成报告")
    print("=" * 60)
    
    report_cmd = [
        sys.executable, str(base_dir / 'generate_report.py'),
        '--query', args.query,
        '--context', str(processed_dir / 'retrieved_context.json'),
        '--model', args.model,
        '--base-url', args.base_url,
        '--save-run' if args.save_run else '',
        '--runs-dir', str(runs_dir)
    ]
    
    if not run_command(report_cmd, "生成结构化报告"):
        return 1
    
    # 总结
    print("\n" + "=" * 60)
    print("流程完成!")
    print("=" * 60)
    
    print(f"\n查询: {args.query}")
    print(f"输入目录: {args.input}")
    print(f"切分方法: {args.chunk_method}")
    print(f"检索方法: {args.retrieve_method}")
    print(f"top-k: {args.top_k}")
    
    if args.save_run:
        run_dirs = sorted(runs_dir.glob('*'), key=lambda x: x.stat().st_mtime, reverse=True)
        if run_dirs:
            print(f"\n最新运行记录: {run_dirs[0]}")
    
    print("\n输出文件:")
    print(f"  - 原始文档: {processed_dir / 'docs_raw.jsonl'}")
    print(f"  - 切分结果: {processed_dir / 'docs_chunks.jsonl'}")
    print(f"  - 检索结果: {processed_dir / 'retrieved_context.json'}")
    
    report_files = list(processed_dir.glob('report_*.md'))
    if report_files:
        print(f"  - 报告文件: {report_files[-1]}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
