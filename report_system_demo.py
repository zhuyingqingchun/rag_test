#!/usr/bin/env python3
"""
报告系统演示脚本
"""
from report_system.main import ReportSystem

# 初始化报告系统
system = ReportSystem()

# 添加示例文档
print("添加示例文档...")
system.add_document("data/flight_control_doc.md", "markdown")

# 生成摘要报告
print("\n生成摘要报告...")
summary_report = system.generate_summary_report("飞行控制文档的主要内容")
print(summary_report)

# 导出报告
print("\n导出报告...")
system.export_report(summary_report, "outputs/summary_report.md", "markdown")
system.export_report(summary_report, "outputs/summary_report.docx", "docx")
system.export_report(summary_report, "outputs/summary_report.pdf", "pdf")

print("\n报告生成完成！")
