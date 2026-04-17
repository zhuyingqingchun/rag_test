#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""报告模板配置。"""
from __future__ import annotations

from copy import deepcopy

DEFAULT_REPORT_TYPE = "summary"

REPORT_TEMPLATES = {
    "summary": {
        "name": "文档摘要报告",
        "title_prefix": "摘要｜",
        "default_title": "文档摘要报告",
        "goal": "对文档核心内容进行压缩提炼，突出关键信息与结论。",
        "key_points_guidance": "聚焦事实、定义、关键步骤和核心结论，避免扩展性评论。",
        "analysis_guidance": "总结材料主线，说明各证据之间的整体关系。",
        "conclusion_guidance": "给出精炼结论，适合直接阅读或继续汇报使用。",
    },
    "research": {
        "name": "专题调研报告",
        "title_prefix": "调研｜",
        "default_title": "专题调研报告",
        "goal": "围绕用户主题归纳资料现状、方法路线、代表观点与趋势。",
        "key_points_guidance": "重点整理主题背景、主要方法、关键观点与代表性内容。",
        "analysis_guidance": "分析证据之间的联系、差异与共性，适当指出研究/资料脉络。",
        "conclusion_guidance": "输出对专题的阶段性判断，并明确目前证据能支持到什么程度。",
    },
    "comparison": {
        "name": "对比分析报告",
        "title_prefix": "对比｜",
        "default_title": "对比分析报告",
        "goal": "对多个方案、观点或对象进行并列比较，突出差异与优缺点。",
        "key_points_guidance": "按比较维度整理证据，例如方法、性能、成本、适用场景、风险。",
        "analysis_guidance": "明确各对象之间的差异点与取舍关系，避免笼统总结。",
        "conclusion_guidance": "给出相对清晰的对比结论，并说明其依据来自哪些证据。",
    },
    "project": {
        "name": "项目汇报报告",
        "title_prefix": "汇报｜",
        "default_title": "项目汇报报告",
        "goal": "形成适合向团队或导师汇报的阶段性总结材料。",
        "key_points_guidance": "突出任务目标、阶段进展、关键成果、主要问题与待办事项。",
        "analysis_guidance": "围绕项目推进情况解释当前状态、阻塞点与后续重点。",
        "conclusion_guidance": "给出当前项目状态判断和下一步建议，便于汇报决策。",
    },
    "incident": {
        "name": "故障/事件分析报告",
        "title_prefix": "事件｜",
        "default_title": "故障/事件分析报告",
        "goal": "围绕故障、异常或事件材料，形成面向排查和复盘的分析报告。",
        "key_points_guidance": "优先整理现象、证据、原因线索、影响范围与处置动作。",
        "analysis_guidance": "分析可能原因、触发链条、风险点与证据支撑程度。",
        "conclusion_guidance": "输出更偏复盘/诊断式结论，说明哪些结论已确认、哪些仍待验证。",
    },
}


def get_report_template(report_type: str) -> dict:
    if report_type not in REPORT_TEMPLATES:
        report_type = DEFAULT_REPORT_TYPE
    return deepcopy(REPORT_TEMPLATES[report_type])
