#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多报告模板对比评估脚本。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).parent))

from evaluate_main import run_all_evaluations, EVALUATION_QUESTIONS
from report_templates import REPORT_TEMPLATES, DEFAULT_REPORT_TYPE


def compact_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    report_metrics = summary.get("报告指标平均值", {})
    retrieval_metrics = summary.get("检索指标平均值", {})
    return {
        "report_type": summary.get("report_type"),
        "report_type_name": summary.get("report_type_name"),
        "success_count": summary.get("成功数量", 0),
        "failure_count": summary.get("失败数量", 0),
        "综合得分": report_metrics.get("综合得分", 0),
        "事实一致性": report_metrics.get("事实一致性", 0),
        "引用覆盖率": report_metrics.get("引用覆盖率", 0),
        "结构完整率": report_metrics.get("结构完整率", 0),
        "Recall@5": retrieval_metrics.get("Recall@5", 0),
        "Hit@5": retrieval_metrics.get("Hit@5", 0),
    }


def run_template_matrix(input_dir: str,
                        processed_root: Path,
                        runs_root: Path,
                        report_types: List[str],
                        questions: List[Dict[str, Any]],
                        retrieve_method: str,
                        top_k: int) -> Dict[str, Any]:
    all_results = []
    for report_type in report_types:
        print("\n" + "=" * 80)
        print(f"开始模板评估: {report_type} ({REPORT_TEMPLATES[report_type]['name']})")
        print("=" * 80)

        processed_dir = processed_root / report_type
        runs_dir = runs_root / report_type
        processed_dir.mkdir(parents=True, exist_ok=True)
        runs_dir.mkdir(parents=True, exist_ok=True)

        result = run_all_evaluations(
            input_dir=input_dir,
            processed_dir=processed_dir,
            runs_dir=runs_dir,
            questions=questions,
            retrieve_method=retrieve_method,
            top_k=top_k,
            report_type=report_type,
        )
        all_results.append({
            "report_type": report_type,
            "report_type_name": REPORT_TEMPLATES[report_type]["name"],
            "result": result,
            "compact_summary": compact_summary(result.get("summary", {})),
        })

    ranking = sorted(
        [item["compact_summary"] for item in all_results],
        key=lambda x: (x.get("综合得分", 0), x.get("事实一致性", 0), x.get("引用覆盖率", 0)),
        reverse=True,
    )

    best_template = ranking[0] if ranking else {
        "report_type": DEFAULT_REPORT_TYPE,
        "report_type_name": REPORT_TEMPLATES[DEFAULT_REPORT_TYPE]["name"],
    }

    return {
        "report_types": report_types,
        "ranking": ranking,
        "best_template": best_template,
        "details": all_results,
    }


def print_ranking(ranking: List[Dict[str, Any]]) -> None:
    print("\n" + "=" * 80)
    print("模板对比结果")
    print("=" * 80)
    for idx, item in enumerate(ranking, 1):
        print(
            f"[{idx}] {item['report_type']} / {item['report_type_name']} | "
            f"综合得分={item.get('综合得分', 0)} | "
            f"事实一致性={item.get('事实一致性', 0)} | "
            f"引用覆盖率={item.get('引用覆盖率', 0)}"
        )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="多报告模板对比评估")
    parser.add_argument("--input", "-i", default="./data/input_docs", help="输入文档目录")
    parser.add_argument("--processed-root", default="./data/processed/template_matrix", help="模板评估处理目录")
    parser.add_argument("--runs-root", default="./data/runs/template_matrix", help="模板评估运行记录目录")
    parser.add_argument("--output", "-o", default="./template_matrix_results.json", help="输出结果 JSON 文件")
    parser.add_argument("--questions", "-q", type=int, default=None, help="评估问题数量（默认全部）")
    parser.add_argument("--retrieve-method", "-m", default="bm25", choices=["bm25", "tfidf", "keyword"], help="检索方法")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="评估 top-k")
    parser.add_argument("--report-types", nargs="+", default=None, choices=sorted(REPORT_TEMPLATES.keys()), help="要评估的报告模板列表")

    args = parser.parse_args()

    selected_questions = EVALUATION_QUESTIONS[:args.questions] if args.questions else EVALUATION_QUESTIONS
    report_types = args.report_types if args.report_types else sorted(REPORT_TEMPLATES.keys())

    result = run_template_matrix(
        input_dir=args.input,
        processed_root=Path(args.processed_root),
        runs_root=Path(args.runs_root),
        report_types=report_types,
        questions=selected_questions,
        retrieve_method=args.retrieve_method,
        top_k=args.top_k,
    )

    print_ranking(result["ranking"])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"模板矩阵结果已保存到: {output_path}")
    print(f"最佳模板: {result['best_template']['report_type']} / {result['best_template']['report_type_name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
