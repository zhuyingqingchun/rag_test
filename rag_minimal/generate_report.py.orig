#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告生成模块
调用 Qwen 服务生成结构化报告
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI
except ImportError:
    print("错误: openai 未安装，请运行: pip install openai")
    sys.exit(1)


DEFAULT_REPORT_TEMPLATE = """# {title}

## 一、任务说明
- 用户问题：{query}
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


def build_prompt(query: str, retrieved_results: List[Dict[str, Any]], 
                 template: str = None) -> str:
    """构建 prompt"""
    if template is None:
        template = DEFAULT_REPORT_TEMPLATE
    
    # 提取关键信息
    retrieved_count = len(retrieved_results)
    sources = list(set([r['source'] for r in retrieved_results]))
    
    # 构建参考片段
    references = []
    for i, r in enumerate(retrieved_results, 1):
        text = r['text'][:500] if len(r['text']) > 500 else r['text']
        references.append(f"- [{i}] {r['source']} (score: {r['score']:.3f})\n  {text}")
    
    # 构建 prompt
    prompt_parts = []
    for i, r in enumerate(retrieved_results):
        prompt_parts.append(f"片段 {i+1} (score: {r['score']:.3f}):")
        prompt_parts.append(r['text'])
    
    prompt = f"""请根据以下检索结果，生成一份结构化报告。

用户问题：{query}

检索结果（按相关性排序）：
{chr(10).join(prompt_parts)}

请按照以下格式生成报告：

# 报告标题

## 一、任务说明
- 用户问题：{query}
- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 二、检索摘要
- 共召回片段数：{retrieved_count}
- 主要来源：{', '.join(sources[:3])}

## 三、关键内容整理
1. [关键点1]
2. [关键点2]
3. [关键点3]

## 四、综合分析
[综合分析内容]

## 五、结论
[结论内容]

## 六、参考片段
{chr(10).join([f'- [{i+1}] {r["source"]} (score: {r["score"]:.3f})' + chr(10) + f'  {r["text"][:200]}' for i, r in enumerate(retrieved_results)])}
"""
    
    return prompt


def generate_report(query: str, context_path: str, 
                   model_name: str = "next80b_fp8",
                   base_url: str = "http://127.0.0.1:8000/v1",
                   api_key: str = "EMPTY",
                   output_path: str = None) -> Dict[str, Any]:
    """生成报告"""
    # 加载检索结果
    context = load_retrieved_context(context_path)
    results = context['results']
    
    if not results:
        return {
            "status": "error",
            "message": "没有检索结果，无法生成报告",
            "query": query
        }
    
    # 构建 prompt
    prompt = build_prompt(query, results)
    
    # 调用 Qwen 服务
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个专业的报告生成助手。请根据检索结果生成结构化报告。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048
        )
        
        report_content = response.choices[0].message.content
        
        # 生成报告
        sources = list(set([r['source'] for r in results]))
        
        if output_path is None:
            output_path = f"./data/processed/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return {
            "status": "success",
            "query": query,
            "model": model_name,
            "retrieved_count": len(results),
            "output_path": output_path,
            "report_content": report_content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "query": query,
            "error_detail": str(e)
        }


def save_run_record(query: str, context_path: str, report_result: Dict[str, Any],
                   runs_dir: str = "./data/runs") -> str:
    """保存运行记录"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(runs_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存检索结果
    context = load_retrieved_context(context_path)
    with open(run_dir / "retrieved_context.json", 'w', encoding='utf-8') as f:
        json.dump(context, f, ensure_ascii=False, indent=2)
    
    # 保存报告
    with open(run_dir / "report.md", 'w', encoding='utf-8') as f:
        f.write(report_result.get('report_content', ''))
    
    # 保存元数据
    metadata = {
        "query": query,
        "timestamp": timestamp,
        "status": report_result['status'],
        "model": report_result.get('model', 'N/A'),
        "retrieved_count": report_result.get('retrieved_count', 0)
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
    
    args = parser.parse_args()
    
    print(f"查询: {args.query}")
    print(f"检索结果: {args.context}")
    print(f"模型: {args.model}")
    print(f"服务: {args.base_url}")
    print("-" * 60)
    
    result = generate_report(
        args.query,
        args.context,
        args.model,
        args.base_url,
        args.api_key,
        args.output
    )
    
    if result['status'] == 'success':
        print(f"状态: 成功")
        print(f"输出: {result['output_path']}")
        print(f"使用: {result['usage']['total_tokens']} tokens")
        
        if args.save_run:
            run_dir = save_run_record(args.query, args.context, result, args.runs_dir)
            print(f"记录: {run_dir}")
    else:
        print(f"状态: 失败")
        print(f"错误: {result['message']}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
