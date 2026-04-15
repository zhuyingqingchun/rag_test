#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告评估模块
评估生成报告的质量和结构完整性
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional


def load_report(report_path: str) -> str:
    """加载报告内容"""
    with open(report_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_retrieved_context(context_path: str) -> Dict[str, Any]:
    """加载检索结果"""
    with open(context_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_structure(report_text: str) -> Dict[str, bool]:
    """检查报告结构完整性"""
    sections = {
        "标题": r"^#\s+.+",
        "任务说明": r"##\s+一、任务说明",
        "检索摘要": r"##\s+二、检索摘要",
        "关键内容整理": r"##\s+三、关键内容整理",
        "综合分析": r"##\s+四、综合分析",
        "结论": r"##\s+五、结论",
        "参考片段": r"##\s+六、参考片段"
    }
    
    results = {}
    for name, pattern in sections.items():
        results[name] = bool(re.search(pattern, report_text, re.MULTILINE))
    
    return results


def count_chapter_coverage(report_text: str) -> Dict[str, int]:
    """统计章节覆盖情况"""
    # 检查关键子章节
    sub_sections = {
        "用户问题": r"用户问题",
        "生成时间": r"生成时间",
        "共召回片段数": r"共召回片段数",
        "主要来源": r"主要来源",
        "关键点": r"\d+\.\s+\[?关键",
        "结论内容": r"##\s+五、结论"
    }
    
    results = {}
    for name, pattern in sub_sections.items():
        matches = re.findall(pattern, report_text, re.MULTILINE | re.IGNORECASE)
        results[name] = len(matches)
    
    return results


def check_citation_coverage(report_text: str, context_path: str) -> Dict[str, Any]:
    """检查引用覆盖情况"""
    context = load_retrieved_context(context_path)
    results = context.get('results', [])
    
    if not results:
        return {
            "total_references": 0,
            "cited_count": 0,
            "coverage": 0.0,
            "cited_sources": []
        }
    
    # 提取报告中引用的来源
    cited_sources = re.findall(r"\[(\d+)\]\s+(.+?)(?:\n|$)", report_text)
    cited_source_paths = [path.strip() for _, path in cited_sources]
    
    # 提取检索结果的来源
    all_sources = list(set([r['source'] for r in results]))
    
    # 计算覆盖率
    cited_count = sum(1 for src in all_sources if any(src in cited for cited in cited_source_paths))
    coverage = cited_count / len(all_sources) if all_sources else 0.0
    
    return {
        "total_references": len(all_sources),
        "cited_count": cited_count,
        "coverage": round(coverage, 2),
        "cited_sources": cited_source_paths[:5],
        "all_sources": all_sources
    }


def check_factual_consistency(report_text: str, context_path: str) -> Dict[str, Any]:
    """检查事实一致性（简单版本）"""
    context = load_retrieved_context(context_path)
    results = context.get('results', [])
    
    if not results:
        return {
            "status": "warning",
            "message": "没有检索结果可供比对",
            "consistency_score": 0.0
        }
    
    # 提取报告中的关键信息
    key_points = re.findall(r"\d+\.\s+(.+?)(?:\n|$)", report_text)
    
    # 检查关键信息是否能在检索结果中找到依据
    consistency_count = 0
    total_points = len(key_points)
    
    for point in key_points[:10]:  # 只检查前10个关键点
        point_lower = point.lower()
        for r in results:
            if any(word in r['text'].lower() for word in point_lower.split() if len(word) > 3):
                consistency_count += 1
                break
    
    consistency_score = consistency_count / total_points if total_points > 0 else 0.0
    
    return {
        "total_key_points": total_points,
        "consistent_points": consistency_count,
        "consistency_score": round(consistency_score, 2)
    }


def evaluate_report(report_path: str, context_path: str) -> Dict[str, Any]:
    """
    评估报告质量
    
    Args:
        report_path: 报告文件路径
        context_path: 检索结果文件路径
    
    Returns:
        评估结果字典
    """
    report_text = load_report(report_path)
    
    # 结构检查
    structure = check_structure(report_text)
    structure_complete = all(structure.values())
    
    # 章节覆盖
    chapter_coverage = count_chapter_coverage(report_text)
    
    # 引用覆盖
    citation = check_citation_coverage(report_text, context_path)
    
    # 事实一致性
    consistency = check_factual_consistency(report_text, context_path)
    
    # 综合评分
    structure_score = sum(structure.values()) / len(structure) if structure else 0
    citation_score = citation['coverage']
    consistency_score = consistency['consistency_score']
    
    overall_score = (structure_score + citation_score + consistency_score) / 3
    
    return {
        "status": "success",
        "report_file": report_path,
        "context_file": context_path,
        "metrics": {
            "结构完整率": round(structure_score, 2),
            "引用覆盖率": round(citation_score, 2),
            "事实一致性": round(consistency_score, 2),
            "综合得分": round(overall_score, 2)
        },
        "structure": structure,
        "chapter_coverage": chapter_coverage,
        "citation": citation,
        "consistency": consistency
    }


def print_evaluation(result: Dict[str, Any]) -> None:
    """打印评估结果"""
    print("=" * 60)
    print("报告评估结果")
    print("=" * 60)
    print(f"报告文件: {result['report_file']}")
    print()
    print("综合指标:")
    metrics = result['metrics']
    print(f"  结构完整率:  {metrics['结构完整率']:.2f}")
    print(f"  引用覆盖率:  {metrics['引用覆盖率']:.2f}")
    print(f"  事实一致性:  {metrics['事实一致性']:.2f}")
    print(f"  综合得分:    {metrics['综合得分']:.2f}/1.0")
    print()
    
    print("结构检查:")
    structure = result['structure']
    for name, present in structure.items():
        status = "✓" if present else "✗"
        print(f"  {status} {name}")
    print()
    
    print("引用情况:")
    citation = result['citation']
    print(f"  总引用数: {citation['total_references']}")
    print(f"  已引用:   {citation['cited_count']}")
    print(f"  覆盖率:   {citation['coverage']:.2f}")
    if citation['cited_sources']:
        print(f"  引用来源:")
        for src in citation['cited_sources'][:3]:
            print(f"    - {src}")
    print()
    
    print("事实一致性:")
    consistency = result['consistency']
    print(f"  关键点数:     {consistency.get('total_key_points', 0)}")
    print(f"  一致点数:     {consistency.get('consistent_points', 0)}")
    print(f"  一致性得分:   {consistency.get('consistency_score', 0):.2f}")
    print()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='报告评估模块')
    parser.add_argument('--report', '-r', required=True, help='报告文件路径')
    parser.add_argument('--context', '-c', required=True, help='检索结果 JSON 文件')
    parser.add_argument('--output', '-o', default=None, help='输出评估结果文件')
    
    args = parser.parse_args()
    
    result = evaluate_report(
        report_path=args.report,
        context_path=args.context
    )
    
    print_evaluation(result)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"评估结果已保存到: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
