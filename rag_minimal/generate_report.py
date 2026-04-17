#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告生成模块
调用 Qwen 服务生成结构化报告
"""

import os
import re
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

try:
    from openai import OpenAI
except ImportError:
    print("错误: openai 未安装，请运行: pip install openai")
    sys.exit(1)

from report_templates import (
    REPORT_TEMPLATES,
    DEFAULT_REPORT_TYPE,
    get_report_template,
)


DEFAULT_REPORT_TEMPLATE = """# {title}

## 一、任务说明
- 用户问题：{query}
- 报告类型：{report_type_name}
- 生成时间：{timestamp}

## 二、检索摘要
- 共召回片段数：{retrieved_count}
- 主要来源：{sources}

## 三、关键内容整理
{key_points}

## 四、综合分析
{analysis}

## 五、结论
{conclusion}

## 六、参考片段
{references}
"""


def load_retrieved_context(context_path: str) -> Dict[str, Any]:
    """加载检索结果"""
    with open(context_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_source(source: str) -> str:
    source = (source or '').strip()
    if not source:
        return ''
    return Path(source).name or Path(source).as_posix()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def safe_usage_dict(response: Any) -> Dict[str, int]:
    usage = getattr(response, 'usage', None)
    return {
        "prompt_tokens": getattr(usage, 'prompt_tokens', 0) if usage else 0,
        "completion_tokens": getattr(usage, 'completion_tokens', 0) if usage else 0,
        "total_tokens": getattr(usage, 'total_tokens', 0) if usage else 0,
    }


def build_evidence_items(retrieved_results: List[Dict[str, Any]],
                         max_evidence: int = 5,
                         max_chunk_chars: int = 500) -> List[Dict[str, Any]]:
    """将检索结果规整为可引用证据。"""
    evidence_items: List[Dict[str, Any]] = []
    for idx, result in enumerate(retrieved_results[:max_evidence], 1):
        raw_text = normalize_whitespace(result.get('text', ''))
        snippet = raw_text[:max_chunk_chars]
        evidence_items.append({
            "ref_id": idx,
            "source": result.get('source', ''),
            "source_label": normalize_source(result.get('source', '')),
            "score": float(result.get('score', 0.0)),
            "text": snippet,
            "full_text": raw_text,
        })
    return evidence_items


def format_citations(ref_ids: List[int]) -> str:
    ordered = []
    seen = set()
    for ref_id in ref_ids:
        if isinstance(ref_id, int) and ref_id > 0 and ref_id not in seen:
            ordered.append(ref_id)
            seen.add(ref_id)
    return ''.join(f'[{ref_id}]' for ref_id in ordered)


def build_planning_prompt(query: str,
                          evidence_items: List[Dict[str, Any]],
                          report_template: Dict[str, Any]) -> str:
    evidence_lines = []
    for item in evidence_items:
        evidence_lines.append(
            f"证据[{item['ref_id']}] 来源: {item['source']} | 文件名: {item['source_label']} | 分数: {item['score']:.3f}\n"
            f"证据内容: {item['text']}"
        )

    return f"""你是一个严格基于证据编排报告的助手。

任务：
根据给定证据，为下面的问题生成“报告规划 JSON”。

报告类型：
{report_template['name']}

报告目标：
{report_template['goal']}

关键内容整理要求：
{report_template['key_points_guidance']}

综合分析要求：
{report_template['analysis_guidance']}

结论要求：
{report_template['conclusion_guidance']}

用户问题：
{query}

可用证据：
{chr(10).join(evidence_lines)}

硬性要求：
1. 只能基于给定证据生成，不得补充证据中没有出现的事实、方法、结论。
2. 每条关键内容必须绑定 evidence 编号列表，例如 [1, 3]。
3. 综合分析和结论也必须绑定 evidence 编号列表。
4. 如果证据不足，请明确写“根据当前检索证据无法确认”，不要猜测。
5. 只输出 JSON，不要输出解释、Markdown 或代码块。

JSON 模板：
{{
  "title": "报告标题",
  "key_points": [
    {{"point": "关键点1", "evidence": [1]}},
    {{"point": "关键点2", "evidence": [2, 3]}},
    {{"point": "关键点3", "evidence": [4]}}
  ],
  "analysis": "综合分析内容",
  "analysis_evidence": [1, 2],
  "conclusion": "结论内容",
  "conclusion_evidence": [2, 4]
}}
"""


def extract_json_block(text: str) -> str:
    text = (text or '').strip()
    if not text:
        return ''

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)

    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def parse_plan(raw_text: str) -> Dict[str, Any]:
    json_text = extract_json_block(raw_text)
    if not json_text:
        raise ValueError("模型未返回可解析的 JSON 规划")
    return json.loads(json_text)


def validate_plan(plan: Dict[str, Any], valid_ref_ids: set[int]) -> List[str]:
    errors: List[str] = []

    if not isinstance(plan, dict):
        return ["规划结果不是 JSON 对象"]

    if not isinstance(plan.get('title'), str) or not plan.get('title', '').strip():
        errors.append("缺少有效 title")

    key_points = plan.get('key_points')
    if not isinstance(key_points, list) or not key_points:
        errors.append("key_points 必须是非空列表")
    else:
        for idx, item in enumerate(key_points, 1):
            if not isinstance(item, dict):
                errors.append(f"key_points[{idx}] 不是对象")
                continue
            if not isinstance(item.get('point'), str) or not item.get('point', '').strip():
                errors.append(f"key_points[{idx}] 缺少 point")
            evidence = item.get('evidence')
            if not isinstance(evidence, list) or not evidence:
                errors.append(f"key_points[{idx}] 缺少 evidence")
            else:
                invalid_refs = [ref for ref in evidence if ref not in valid_ref_ids]
                if invalid_refs:
                    errors.append(f"key_points[{idx}] 存在非法 evidence 编号: {invalid_refs}")

    for field_name in ['analysis', 'conclusion']:
        if not isinstance(plan.get(field_name), str) or not plan.get(field_name, '').strip():
            errors.append(f"缺少有效 {field_name}")

    for field_name in ['analysis_evidence', 'conclusion_evidence']:
        evidence = plan.get(field_name)
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"缺少有效 {field_name}")
        else:
            invalid_refs = [ref for ref in evidence if ref not in valid_ref_ids]
            if invalid_refs:
                errors.append(f"{field_name} 存在非法 evidence 编号: {invalid_refs}")

    return errors


def build_repair_prompt(query: str, raw_plan_text: str, errors: List[str]) -> str:
    return f"""上一次返回的报告规划 JSON 不符合要求，请修正。

用户问题：
{query}

原始输出：
{raw_plan_text}

错误列表：
{chr(10).join(f'- {error}' for error in errors)}

请重新输出符合要求的 JSON，要求与上次相同：
- 只能基于证据
- 必须保留 title / key_points / analysis / conclusion 结构
- 每个结论都必须绑定合法 evidence 编号
- 只输出 JSON
"""


def call_chat_completion(client: OpenAI,
                         model_name: str,
                         system_prompt: str,
                         user_prompt: str,
                         temperature: float = 0.2,
                         max_tokens: int = 2048) -> Any:
    return client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )


def render_report(query: str,
                  evidence_items: List[Dict[str, Any]],
                  plan: Dict[str, Any],
                  timestamp: str,
                  report_template: Dict[str, Any]) -> str:
    unique_sources = []
    seen_sources = set()
    for item in evidence_items:
        if item['source_label'] not in seen_sources:
            unique_sources.append(item['source_label'])
            seen_sources.add(item['source_label'])

    key_point_lines = []
    for idx, item in enumerate(plan.get('key_points', []), 1):
        point = normalize_whitespace(item.get('point', ''))
        citations = format_citations(item.get('evidence', []))
        key_point_lines.append(f"{idx}. {point} {citations}".rstrip())

    analysis_text = normalize_whitespace(plan.get('analysis', ''))
    analysis_citations = format_citations(plan.get('analysis_evidence', []))
    analysis = f"{analysis_text} {analysis_citations}".rstrip()

    conclusion_text = normalize_whitespace(plan.get('conclusion', ''))
    conclusion_citations = format_citations(plan.get('conclusion_evidence', []))
    conclusion = f"{conclusion_text} {conclusion_citations}".rstrip()

    references = []
    for item in evidence_items:
        references.append(
            f"- [{item['ref_id']}] {item['source']} (score: {item['score']:.3f})\n"
            f"  {item['text']}"
        )

    title = normalize_whitespace(plan.get('title', report_template['default_title'])) or report_template['default_title']
    if report_template.get('title_prefix') and not title.startswith(report_template['title_prefix']):
        title = f"{report_template['title_prefix']}{title}"

    return DEFAULT_REPORT_TEMPLATE.format(
        title=title,
        query=query,
        report_type_name=report_template['name'],
        timestamp=timestamp,
        retrieved_count=len(evidence_items),
        sources=', '.join(unique_sources[:3]) if unique_sources else 'N/A',
        key_points='\n'.join(key_point_lines),
        analysis=analysis,
        conclusion=conclusion,
        references='\n'.join(references),
    )


def validate_rendered_report(report_text: str, evidence_items: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    required_sections = [
        r'^#\s+.+',
        r'##\s+一、任务说明',
        r'##\s+二、检索摘要',
        r'##\s+三、关键内容整理',
        r'##\s+四、综合分析',
        r'##\s+五、结论',
        r'##\s+六、参考片段',
    ]
    for pattern in required_sections:
        if not re.search(pattern, report_text, re.MULTILINE):
            errors.append(f"缺少章节: {pattern}")

    point_lines = re.findall(r'^\s*\d+\.\s+.+$', report_text, re.MULTILINE)
    if not point_lines:
        errors.append("关键内容整理为空")
    elif any(not re.search(r'\[\d+\]', line) for line in point_lines):
        errors.append("关键内容整理存在未标注证据的条目")

    analysis_match = re.search(r'##\s+四、综合分析\s*(.*?)(?=\n##\s+|\Z)', report_text, re.S)
    if not analysis_match or not re.search(r'\[\d+\]', analysis_match.group(1)):
        errors.append("综合分析缺少证据引用")

    conclusion_match = re.search(r'##\s+五、结论\s*(.*?)(?=\n##\s+|\Z)', report_text, re.S)
    if not conclusion_match or not re.search(r'\[\d+\]', conclusion_match.group(1)):
        errors.append("结论缺少证据引用")

    expected_refs = {item['ref_id'] for item in evidence_items}
    found_refs = {int(ref) for ref in re.findall(r'- \[(\d+)\]', report_text)}
    missing_refs = sorted(expected_refs - found_refs)
    if missing_refs:
        errors.append(f"参考片段缺少证据编号: {missing_refs}")

    return errors


def generate_report(query: str, context_path: str,
                    model_name: str = "next80b_fp8",
                    base_url: str = "http://127.0.0.1:8000/v1",
                    api_key: str = "EMPTY",
                    output_path: str = None,
                    strict_grounding: bool = True,
                    max_evidence: int = 5,
                    max_chunk_chars: int = 500,
                    retry_on_validation_fail: bool = True,
                    report_type: str = DEFAULT_REPORT_TYPE) -> Dict[str, Any]:
    """生成报告"""
    context = load_retrieved_context(context_path)
    results = context['results']

    if not results:
        return {
            "status": "error",
            "message": "没有检索结果，无法生成报告",
            "query": query
        }

    report_template = get_report_template(report_type)
    evidence_items = build_evidence_items(
        results,
        max_evidence=max_evidence,
        max_chunk_chars=max_chunk_chars,
    )
    valid_ref_ids = {item['ref_id'] for item in evidence_items}

    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    system_prompt = (
        f"你是一个严格的结构化报告编排助手，当前任务类型是{report_template['name']}。你必须只依据给定证据输出内容，所有关键结论都必须绑定证据编号。"
        if strict_grounding else
        f"你是一个专业的报告生成助手，当前任务类型是{report_template['name']}，请尽量基于检索结果生成结构化报告。"
    )

    try:
        planning_prompt = build_planning_prompt(query, evidence_items, report_template)
        response = call_chat_completion(
            client=client,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=planning_prompt,
            temperature=0.2 if strict_grounding else 0.5,
            max_tokens=1600,
        )
        total_usage = safe_usage_dict(response)
        raw_plan_text = response.choices[0].message.content or ''

        plan = parse_plan(raw_plan_text)
        plan_errors = validate_plan(plan, valid_ref_ids)

        if plan_errors and retry_on_validation_fail:
            repair_prompt = build_repair_prompt(query, raw_plan_text, plan_errors)
            repair_response = call_chat_completion(
                client=client,
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=repair_prompt,
                temperature=0.1,
                max_tokens=1600,
            )
            repair_usage = safe_usage_dict(repair_response)
            total_usage = {
                "prompt_tokens": total_usage["prompt_tokens"] + repair_usage["prompt_tokens"],
                "completion_tokens": total_usage["completion_tokens"] + repair_usage["completion_tokens"],
                "total_tokens": total_usage["total_tokens"] + repair_usage["total_tokens"],
            }
            raw_plan_text = repair_response.choices[0].message.content or ''
            plan = parse_plan(raw_plan_text)
            plan_errors = validate_plan(plan, valid_ref_ids)

        if plan_errors:
            return {
                "status": "error",
                "message": "报告规划校验失败",
                "query": query,
                "report_type": report_type,
                "plan_errors": plan_errors,
                "raw_plan": raw_plan_text,
            }

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report_content = render_report(query, evidence_items, plan, timestamp, report_template)
        report_errors = validate_rendered_report(report_content, evidence_items)

        if report_errors:
            return {
                "status": "error",
                "message": "渲染后的报告未通过校验",
                "query": query,
                "report_type": report_type,
                "report_errors": report_errors,
                "report_content": report_content,
            }

        if output_path is None:
            output_path = f"./data/processed/report_{report_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        output_dir = os.path.dirname(output_path) or '.'
        os.makedirs(output_dir, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        return {
            "status": "success",
            "query": query,
            "report_type": report_type,
            "report_type_name": report_template['name'],
            "model": model_name,
            "retrieved_count": len(results),
            "used_evidence_count": len(evidence_items),
            "output_path": output_path,
            "report_content": report_content,
            "plan": plan,
            "usage": total_usage,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "query": query,
            "report_type": report_type,
            "error_detail": str(e)
        }


def save_run_record(query: str, context_path: str, report_result: Dict[str, Any],
                   runs_dir: str = "./data/runs") -> str:
    """保存运行记录"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(runs_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    context = load_retrieved_context(context_path)
    with open(run_dir / "retrieved_context.json", 'w', encoding='utf-8') as f:
        json.dump(context, f, ensure_ascii=False, indent=2)

    with open(run_dir / "report.md", 'w', encoding='utf-8') as f:
        f.write(report_result.get('report_content', ''))

    plan = report_result.get('plan')
    if plan is not None:
        with open(run_dir / "plan.json", 'w', encoding='utf-8') as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)

    metadata = {
        "query": query,
        "timestamp": timestamp,
        "status": report_result['status'],
        "report_type": report_result.get('report_type', DEFAULT_REPORT_TYPE),
        "report_type_name": report_result.get('report_type_name', get_report_template(DEFAULT_REPORT_TYPE)['name']),
        "model": report_result.get('model', 'N/A'),
        "retrieved_count": report_result.get('retrieved_count', 0),
        "used_evidence_count": report_result.get('used_evidence_count', 0)
    }

    with open(run_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return str(run_dir)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='报告生成模块')
    parser.add_argument('--query', '-q', required=True, help='用户查询')
    parser.add_argument('--context', '-c', default='./data/processed/retrieved_context.json',
                        help='检索结果文件路径')
    parser.add_argument('--model', '-m', default='next80b_fp8',
                        help='模型名称')
    parser.add_argument('--base-url', '-b', default='http://127.0.0.1:8000/v1',
                        help='Qwen 服务地址')
    parser.add_argument('--api-key', '-k', default='EMPTY',
                        help='API 密钥')
    parser.add_argument('--output', '-o', default=None,
                        help='输出报告文件路径')
    parser.add_argument('--save-run', '-s', action='store_true',
                        help='保存运行记录')
    parser.add_argument('--runs-dir', '-r', default='./data/runs',
                        help='运行记录保存目录')
    parser.add_argument('--strict-grounding', action='store_true',
                        help='启用严格证据约束生成')
    parser.add_argument('--max-evidence', type=int, default=5,
                        help='最多使用的证据条数')
    parser.add_argument('--max-chunk-chars', type=int, default=500,
                        help='每条证据保留的最大字符数')
    parser.add_argument('--retry-on-validation-fail', action='store_true',
                        help='规划 JSON 校验失败时自动重试一次')
    parser.add_argument('--report-type', default=DEFAULT_REPORT_TYPE,
                        choices=sorted(REPORT_TEMPLATES.keys()),
                        help='报告类型模板')

    args = parser.parse_args()

    print(f"查询: {args.query}")
    print(f"检索结果: {args.context}")
    print(f"模型: {args.model}")
    print(f"服务: {args.base_url}")
    print(f"报告类型: {args.report_type}")
    print("-" * 60)

    result = generate_report(
        args.query,
        args.context,
        args.model,
        args.base_url,
        args.api_key,
        args.output,
        args.strict_grounding,
        args.max_evidence,
        args.max_chunk_chars,
        args.retry_on_validation_fail,
        args.report_type,
    )

    if result['status'] == 'success':
        print(f"状态: 成功")
        print(f"输出: {result['output_path']}")
        print(f"使用: {result['usage']['total_tokens']} tokens")
        print(f"证据: {result.get('used_evidence_count', 0)} / {result.get('retrieved_count', 0)}")
        print(f"模板: {result.get('report_type_name', args.report_type)}")

        if args.save_run:
            run_dir = save_run_record(args.query, args.context, result, args.runs_dir)
            print(f"记录: {run_dir}")
    else:
        print(f"状态: 失败")
        print(f"错误: {result['message']}")
        if 'plan_errors' in result:
            print(f"规划错误: {result['plan_errors']}")
        if 'report_errors' in result:
            print(f"报告错误: {result['report_errors']}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
