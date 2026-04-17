#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 评估主脚本
运行完整的评估流程并生成评估报告
"""

import json
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from evaluate_retrieval import evaluate_retrieval, print_evaluation as print_retrieval
from evaluate_report import evaluate_report, print_evaluation as print_report


# 评估问题集
EVALUATION_QUESTIONS = [
    {
        "query": "Transformer 的核心结构和关键机制是什么？",
        "relevant_docs": [
            "transformer.pdf"
        ],
        "expected_topics": ["self-attention", "multi-head attention", "positional encoding", "encoder", "decoder"]
    },
    {
        "query": "BERT 与 GPT 的主要区别是什么？",
        "relevant_docs": [
            "bert.pdf",
            "gpt.pdf",
            "gpt2.pdf",
            "gpt3.pdf"
        ],
        "expected_topics": ["bidirectional", "autoregressive", "masked language model", "pretraining", "decoder-only"]
    },
    {
        "query": "GPT-2 与 GPT-3 在模型规模和能力方面有哪些差异？",
        "relevant_docs": [
            "gpt2.pdf",
            "gpt3.pdf"
        ],
        "expected_topics": ["parameters", "few-shot", "zero-shot", "scale", "language generation"]
    },
    {
        "query": "RAG 系统的核心组成和基本流程是什么？",
        "relevant_docs": [
            "signal_rag_guide.md"
        ],
        "expected_topics": ["retrieval", "generation", "knowledge base", "embedding", "rerank"]
    },
    {
        "query": "hello-algo 中介绍了哪些常见基础数据结构与算法思想？",
        "relevant_docs": [
            "hello-algo_1.3.0_zh_cpp.pdf"
        ],
        "expected_topics": ["array", "linked list", "stack", "queue", "tree", "graph", "sorting", "search"]
    }
]


def prepare_corpus(input_dir: str, processed_dir: Path) -> Dict[str, Any]:
    """预处理一次文档导入和文本切分，供所有 query 复用。"""
    import subprocess

    result: Dict[str, Any] = {}

    print("\n[预处理 1/2] 文档导入...")
    start_time = time.time()
    ingest_cmd = [
        sys.executable, str(Path(__file__).parent / 'ingest.py'),
        '--input', input_dir,
        '--output', str(processed_dir / 'eval_docs_raw.jsonl')
    ]
    ingest_result = subprocess.run(ingest_cmd, capture_output=True, text=True)
    ingest_time = time.time() - start_time
    result['ingest'] = {'status': 'success' if ingest_result.returncode == 0 else 'failed', 'time': round(ingest_time, 2), 'stderr': ingest_result.stderr}
    if ingest_result.returncode != 0:
        return result

    print(f"  完成 (耗时: {ingest_time:.2f}s)")

    print("\n[预处理 2/2] 文本切分...")
    start_time = time.time()
    chunk_cmd = [
        sys.executable, str(Path(__file__).parent / 'chunk.py'),
        '--input', str(processed_dir / 'eval_docs_raw.jsonl'),
        '--output', str(processed_dir / 'eval_docs_chunks.jsonl'),
        '--method', 'paragraph'
    ]
    chunk_result = subprocess.run(chunk_cmd, capture_output=True, text=True)
    chunk_time = time.time() - start_time
    result['chunk'] = {'status': 'success' if chunk_result.returncode == 0 else 'failed', 'time': round(chunk_time, 2), 'stderr': chunk_result.stderr}

    if chunk_result.returncode == 0:
        print(f"  完成 (耗时: {chunk_time:.2f}s)")
    return result


def run_single_evaluation(query_info: Dict[str, Any],
                          chunks_path: Path,
                          processed_dir: Path,
                          runs_dir: Path,
                          model: str = "next80b_fp8",
                          base_url: str = "http://localhost:8000/v1",
                          retrieve_method: str = "bm25",
                          top_k: int = 5) -> Dict[str, Any]:
    """运行单个问题的检索、生成与评估流程。"""

    query = query_info['query']
    relevant_docs = query_info.get('relevant_docs', [])

    print(f"\n{'='*60}")
    print(f"评估问题: {query}")
    print(f"{'='*60}")

    result = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "relevant_docs": relevant_docs
    }

    # 步骤 1: 检索
    print("\n[1/2] 检索...")
    start_time = time.time()

    retrieved_path = processed_dir / f"eval_retrieved_{abs(hash(query)) % 10**8}.json"
    retrieve_cmd = [
        sys.executable, str(Path(__file__).parent / 'retrieve.py'),
        '--query', query,
        '--chunks', str(chunks_path),
        '--method', retrieve_method,
        '--top-k', str(top_k),
        '--output', str(retrieved_path)
    ]

    retrieve_result = subprocess.run(retrieve_cmd, capture_output=True, text=True)
    retrieve_time = time.time() - start_time

    if retrieve_result.returncode != 0:
        result['retrieve'] = {'status': 'failed', 'time': retrieve_time, 'stderr': retrieve_result.stderr}
        return result

    result['retrieve'] = {
        'status': 'success',
        'time': round(retrieve_time, 2),
        'context_file': str(retrieved_path)
    }
    print(f"  完成 (耗时: {retrieve_time:.2f}s)")

    # 评估检索结果
    print("\n[1.1] 评估检索结果...")
    retrieval_result = evaluate_retrieval(
        query=query,
        context_path=str(retrieved_path),
        relevant_docs=relevant_docs,
        k=5
    )
    result['retrieval'] = retrieval_result
    print_retrieval(retrieval_result)

    # 步骤 2: 生成报告
    print("\n[2/2] 生成报告...")
    start_time = time.time()

    report_cmd = [
        sys.executable, str(Path(__file__).parent / 'generate_report.py'),
        '--query', query,
        '--context', str(retrieved_path),
        '--model', model,
        '--base-url', base_url,
        '--save-run',
        '--runs-dir', str(runs_dir)
    ]

    report_result = subprocess.run(report_cmd, capture_output=True, text=True)
    report_time = time.time() - start_time

    if report_result.returncode != 0:
        result['report'] = {'status': 'failed', 'time': report_time}
        return result

    # 找到生成的报告文件 - 优先找本轮新增的 report.md
    report_files = list(runs_dir.glob('**/report.md'))
    if report_files:
        latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
        result['report'] = {
            'status': 'success',
            'time': round(report_time, 2),
            'file': str(latest_report)
        }
    else:
        result['report'] = {
            'status': 'success',
            'time': round(report_time, 2),
            'file': None
        }
    print(f"  完成 (耗时: {report_time:.2f}s)")

    # 评估报告质量
    if result['report']['file']:
        print("\n[2.1] 评估报告质量...")
        report_eval = evaluate_report(
            report_path=result['report']['file'],
            context_path=str(retrieved_path)
        )
        result['report_evaluation'] = report_eval
        print_report(report_eval)

    return result


def run_all_evaluations(input_dir: str,
                        processed_dir: Path,
                        runs_dir: Path,
                        questions: List[Dict[str, Any]] = None,
                        retrieve_method: str = "bm25",
                        top_k: int = 5) -> Dict[str, Any]:
    """运行所有评估问题"""

    if questions is None:
        questions = EVALUATION_QUESTIONS

    print("\n" + "="*60)
    print("RAG 评估系统 - 开始评估")
    print("="*60)
    print(f"问题数量: {len(questions)}")
    print(f"输入目录: {input_dir}")
    print(f"处理目录: {processed_dir}")
    print(f"运行目录: {runs_dir}")
    print()

    # 预处理一次文档导入和切分
    corpus_prep = prepare_corpus(input_dir, processed_dir)
    if corpus_prep['ingest']['status'] != 'success' or corpus_prep['chunk']['status'] != 'success':
        print("预处理失败，无法继续评估")
        return {
            "summary": {"error": "预处理失败"},
            "details": [],
            "total_time": 0,
            "timestamp": datetime.now().isoformat()
        }

    chunks_path = processed_dir / 'eval_docs_chunks.jsonl'

    results = []
    total_start = time.time()

    for i, question in enumerate(questions, 1):
        print(f"\n\n{'#'*60}")
        print(f"问题 {i}/{len(questions)}")
        print(f"{'#'*60}")

        result = run_single_evaluation(
            query_info=question,
            chunks_path=chunks_path,
            processed_dir=processed_dir,
            runs_dir=runs_dir,
            retrieve_method=retrieve_method,
            top_k=top_k
        )
        results.append(result)

    total_time = time.time() - total_start

    # 生成汇总报告
    summary = generate_summary(results, total_time)

    return {
        "summary": summary,
        "details": results,
        "total_time": round(total_time, 2),
        "timestamp": datetime.now().isoformat()
    }


def generate_summary(results: List[Dict], total_time: float) -> Dict[str, Any]:
    """生成评估汇总"""

    # 统计指标
    ingest_times = [0] * len(results)
    chunk_times = [0] * len(results)
    retrieve_times = [r.get('retrieve', {}).get('time', 0) for r in results]
    report_times = [r.get('report', {}).get('time', 0) for r in results]

    retrieval_metrics = []
    report_metrics = []

    for r in results:
        if 'retrieval' in r and r['retrieval'].get('status') == 'success':
            retrieval_metrics.append(r['retrieval']['metrics'])
        if 'report_evaluation' in r and r['report_evaluation'].get('status') == 'success':
            report_metrics.append(r['report_evaluation']['metrics'])

    # 计算平均值
    summary = {
        "总耗时": round(total_time, 2),
        "问题数量": len(results),
        "成功数量": sum(1 for r in results if r.get('report', {}).get('status') == 'success'),
        "失败数量": sum(1 for r in results if r.get('report', {}).get('status') == 'failed'),

        "平均耗时": {
            "导入": round(sum(ingest_times) / len(ingest_times), 2) if ingest_times else 0,
            "切分": round(sum(chunk_times) / len(chunk_times), 2) if chunk_times else 0,
            "检索": round(sum(retrieve_times) / len(retrieve_times), 2) if retrieve_times else 0,
            "生成": round(sum(report_times) / len(report_times), 2) if report_times else 0
        },

        "检索指标平均值": {
            "Recall@5": round(sum(m.get('recall@k', 0) for m in retrieval_metrics) / len(retrieval_metrics), 3) if retrieval_metrics else 0,
            "Precision@5": round(sum(m.get('precision@k', 0) for m in retrieval_metrics) / len(retrieval_metrics), 3) if retrieval_metrics else 0,
            "Hit@5": round(sum(1 for m in retrieval_metrics if m.get('hit_rate', False)) / len(retrieval_metrics), 2) if retrieval_metrics else 0,
            "平均相关性": round(sum(m.get('avg_relevance', 0) for m in retrieval_metrics) / len(retrieval_metrics), 2) if retrieval_metrics else 0
        },

        "报告指标平均值": {
            "结构完整率": round(sum(m.get('结构完整率', 0) for m in report_metrics) / len(report_metrics), 2) if report_metrics else 0,
            "引用覆盖率": round(sum(m.get('引用覆盖率', 0) for m in report_metrics) / len(report_metrics), 2) if report_metrics else 0,
            "事实一致性": round(sum(m.get('事实一致性', 0) for m in report_metrics) / len(report_metrics), 2) if report_metrics else 0,
            "综合得分": round(sum(m.get('综合得分', 0) for m in report_metrics) / len(report_metrics), 2) if report_metrics else 0
        }
    }

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    """打印汇总报告"""
    print("\n" + "="*60)
    print("评估汇总报告")
    print("="*60)

    print(f"\n总体统计:")
    print(f"  总耗时: {summary['总耗时']}s")
    print(f"  问题数量: {summary['问题数量']}")
    print(f"  成功: {summary['成功数量']}")
    print(f"  失败: {summary['失败数量']}")

    print(f"\n平均耗时 (秒):")
    for name, time in summary['平均耗时'].items():
        print(f"  {name}: {time}")

    print(f"\n检索指标 (平均):")
    for name, value in summary['检索指标平均值'].items():
        print(f"  {name}: {value}")

    print(f"\n报告指标 (平均):")
    for name, value in summary['报告指标平均值'].items():
        print(f"  {name}: {value}")

    print()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='RAG 评估系统')
    parser.add_argument('--input', '-i', default='./data/input_docs', help='输入文档目录')
    parser.add_argument('--processed', '-p', default='./data/processed', help='处理结果目录')
    parser.add_argument('--runs', '-r', default='./data/runs/evaluation_v3', help='运行记录目录')
    parser.add_argument('--output', '-o', default='./evaluation_results_v3.json', help='输出结果文件')
    parser.add_argument('--questions', '-q', type=int, default=None, help='评估问题数量（默认全部）')
    parser.add_argument('--retrieve-method', '-m', default='bm25', choices=['bm25', 'tfidf', 'keyword'], help='检索方法')
    parser.add_argument('--top-k', '-k', type=int, default=5, help='评估 top-k')

    args = parser.parse_args()

    # 创建目录
    processed_dir = Path(args.processed)
    runs_dir = Path(args.runs)
    processed_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # 运行评估
    selected_questions = EVALUATION_QUESTIONS[:args.questions] if args.questions else EVALUATION_QUESTIONS
    results = run_all_evaluations(
        input_dir=args.input,
        processed_dir=processed_dir,
        runs_dir=runs_dir,
        questions=selected_questions,
        retrieve_method=args.retrieve_method,
        top_k=args.top_k
    )

    # 打印汇总
    print_summary(results['summary'])

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"评估结果已保存到: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
