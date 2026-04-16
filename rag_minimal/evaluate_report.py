#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""报告评估模块。"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

_EVAL_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}")


def load_report(report_path: str) -> str:
    return Path(report_path).read_text(encoding='utf-8')


def load_retrieved_context(context_path: str) -> Dict[str, Any]:
    with open(context_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_source(source: str) -> str:
    source = (source or '').strip()
    if not source:
        return ''
    return Path(source).name or Path(source).as_posix()


def evaluation_tokens(text: str) -> set[str]:
    return set(_EVAL_TOKEN_RE.findall((text or '').lower()))


def check_structure(report_text: str) -> Dict[str, bool]:
    sections = {
        '标题': r'^#\s+.+',
        '任务说明': r'##\s+(?:一、)?任务说明',
        '检索摘要': r'##\s+(?:二、)?检索摘要',
        '关键内容整理': r'##\s+(?:三、)?关键内容整理',
        '综合分析': r'##\s+(?:四、)?综合分析',
        '结论': r'##\s+(?:五、)?结论',
        '参考片段': r'##\s+(?:六、)?参考片段',
    }
    return {name: bool(re.search(pattern, report_text, re.MULTILINE)) for name, pattern in sections.items()}


def count_chapter_coverage(report_text: str) -> Dict[str, int]:
    return {
        '用户问题': len(re.findall(r'用户问题', report_text)),
        '生成时间': len(re.findall(r'生成时间', report_text)),
        '共召回片段数': len(re.findall(r'共召回片段数', report_text)),
        '主要来源': len(re.findall(r'主要来源', report_text)),
        '关键点条目': len(re.findall(r'^\s*\d+\.\s+', report_text, re.MULTILINE)),
        '结论章节': len(re.findall(r'##\s+(?:五、)?结论', report_text)),
    }


def check_citation_coverage(report_text: str, context_path: str) -> Dict[str, Any]:
    context = load_retrieved_context(context_path)
    results = context.get('results', [])
    all_sources = list({normalize_source(r['source']) for r in results})
    if not all_sources:
        return {'total_references': 0, 'cited_count': 0, 'coverage': 0.0, 'cited_sources': [], 'all_sources': []}

    cited_sources = re.findall(r'- \[(\d+)\]\s+(.+?)(?:\s*\(score:.*?\))?(?:\n|$)', report_text)
    cited_source_paths = [normalize_source(path.strip()) for _, path in cited_sources]
    cited_count = sum(1 for src in all_sources if src in cited_source_paths)
    coverage = cited_count / len(all_sources)

    return {
        'total_references': len(all_sources),
        'cited_count': cited_count,
        'coverage': round(coverage, 2),
        'cited_sources': cited_source_paths[:5],
        'all_sources': all_sources,
    }


def extract_section(report_text: str, section_pattern: str) -> str:
    match = re.search(section_pattern + r'\s*(.*?)(?=\n##\s+|\Z)', report_text, re.S)
    return match.group(1).strip() if match else ''


def split_sentences(text: str) -> List[str]:
    parts = re.split(r'[。！？；\n]+', text)
    return [part.strip(' -•\t') for part in parts if part and part.strip(' -•\t')]


def extract_claims(report_text: str) -> List[str]:
    numbered_points = re.findall(r'^\s*\d+\.\s+(.+?)(?:\n|$)', report_text, re.MULTILINE)
    analysis_text = extract_section(report_text, r'##\s+(?:四、)?综合分析')
    conclusion_text = extract_section(report_text, r'##\s+(?:五、)?结论')

    claims: List[str] = []
    claims.extend([point.strip() for point in numbered_points if point.strip()])

    for sentence in split_sentences(analysis_text):
        if len(sentence) >= 8:
            claims.append(sentence)

    for sentence in split_sentences(conclusion_text):
        if len(sentence) >= 8:
            claims.append(sentence)

    deduped: List[str] = []
    seen = set()
    for claim in claims:
        if claim not in seen:
            deduped.append(claim)
            seen.add(claim)
    return deduped[:20]


def best_support_for_claim(claim: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    claim_tokens = evaluation_tokens(claim)
    if not claim_tokens:
        return {'supported': False, 'best_overlap': 0, 'best_source': None, 'best_preview': ''}

    best_overlap = 0
    best_result = None
    for result in results:
        chunk_tokens = evaluation_tokens(result.get('text', ''))
        overlap = len(claim_tokens & chunk_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_result = result

    threshold = 1 if len(claim_tokens) <= 2 else 2
    return {
        'supported': best_overlap >= threshold,
        'best_overlap': best_overlap,
        'best_source': normalize_source(best_result['source']) if best_result else None,
        'best_preview': best_result['text'][:120] if best_result else '',
    }


def check_factual_consistency(report_text: str, context_path: str) -> Dict[str, Any]:
    context = load_retrieved_context(context_path)
    results = context.get('results', [])
    if not results:
        return {'status': 'warning', 'message': '没有检索结果可供比对', 'consistency_score': 0.0}

    claims = extract_claims(report_text)
    supported_claims = []
    unsupported_claims = []

    for claim in claims:
        support = best_support_for_claim(claim, results)
        claim_record = {
            'claim': claim,
            'best_overlap': support['best_overlap'],
            'best_source': support['best_source'],
            'best_preview': support['best_preview'],
        }
        if support['supported']:
            supported_claims.append(claim_record)
        else:
            unsupported_claims.append(claim_record)

    total_claims = len(claims)
    supported_count = len(supported_claims)
    consistency_score = supported_count / total_claims if total_claims > 0 else 0.0

    return {
        'total_claims': total_claims,
        'supported_claims': supported_count,
        'unsupported_claims': len(unsupported_claims),
        'consistency_score': round(consistency_score, 2),
        'supported_examples': supported_claims[:5],
        'unsupported_examples': unsupported_claims[:5],
    }


def evaluate_report(report_path: str, context_path: str) -> Dict[str, Any]:
    report_text = load_report(report_path)
    structure = check_structure(report_text)
    chapter_coverage = count_chapter_coverage(report_text)
    citation = check_citation_coverage(report_text, context_path)
    consistency = check_factual_consistency(report_text, context_path)
    structure_score = sum(structure.values()) / len(structure) if structure else 0
    citation_score = citation['coverage']
    consistency_score = consistency.get('consistency_score', 0)
    overall_score = (structure_score + citation_score + consistency_score) / 3
    return {
        'status': 'success',
        'report_file': report_path,
        'context_file': context_path,
        'metrics': {'结构完整率': round(structure_score, 2), '引用覆盖率': round(citation_score, 2), '事实一致性': round(consistency_score, 2), '综合得分': round(overall_score, 2)},
        'structure': structure,
        'chapter_coverage': chapter_coverage,
        'citation': citation,
        'consistency': consistency,
    }


def print_evaluation(result: Dict[str, Any]) -> None:
    print('=' * 60)
    print('报告评估结果')
    print('=' * 60)
    print(f"报告文件: {result['report_file']}")
    metrics = result['metrics']
    print(f"  结构完整率:  {metrics['结构完整率']:.2f}")
    print(f"  引用覆盖率:  {metrics['引用覆盖率']:.2f}")
    print(f"  事实一致性:  {metrics['事实一致性']:.2f}")
    print(f"  综合得分:    {metrics['综合得分']:.2f}/1.0")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description='报告评估模块')
    parser.add_argument('--report', '-r', required=True, help='报告文件路径')
    parser.add_argument('--context', '-c', required=True, help='检索结果 JSON 文件')
    parser.add_argument('--output', '-o', default=None, help='输出评估结果文件')
    args = parser.parse_args()
    result = evaluate_report(args.report, args.context)
    print_evaluation(result)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
