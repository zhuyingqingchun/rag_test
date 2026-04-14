# 系统主入口
from datetime import datetime
from report_system.knowledge_base import KnowledgeBase
from report_system.template_engine import TemplateEngine
from report_system.export import Exporter

class ReportSystem:
    def __init__(self):
        self.knowledge_base = KnowledgeBase()
        self.template_engine = TemplateEngine()
        self.exporter = Exporter()
    
    def add_document(self, file_path, doc_type):
        """添加文档到知识库"""
        return self.knowledge_base.add_document(file_path, doc_type)
    
    def generate_summary_report(self, query=""):
        """生成文档摘要报告"""
        # 检索相关文档
        if query:
            documents = self.knowledge_base.retrieve(query)
        else:
            # 如果没有查询，使用默认查询
            documents = self.knowledge_base.retrieve("文档主要内容是什么")
        
        # 提取文档内容
        doc_contents = [doc.content for doc in documents]
        doc_sources = [doc.meta.get("file_path", "未知来源") for doc in documents]
        
        # 生成报告数据
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "doc_count": len(documents),
            "summary": self._generate_summary(doc_contents),
            "key_points": self._extract_key_points(doc_contents),
            "conclusion": self._generate_conclusion(doc_contents),
            "sources": doc_sources
        }
        
        # 生成报告
        return self.template_engine.generate_report("summary_report", data)
    
    def generate_research_report(self, topic, query):
        """生成专题调研报告"""
        # 检索相关文档
        documents = self.knowledge_base.retrieve(query)
        
        # 提取文档内容
        doc_contents = [doc.content for doc in documents]
        doc_sources = [doc.meta.get("file_path", "未知来源") for doc in documents]
        
        # 生成报告数据
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "topic": topic,
            "doc_count": len(documents),
            "background": self._generate_background(topic),
            "findings": self._extract_findings(doc_contents, topic),
            "analysis": self._generate_analysis(doc_contents),
            "conclusion": self._generate_conclusion(doc_contents),
            "sources": doc_sources
        }
        
        # 生成报告
        return self.template_engine.generate_report("research_report", data)
    
    def generate_comparison_report(self, topic, objects):
        """生成对比分析报告"""
        # 为每个对象检索相关文档
        all_documents = []
        for obj in objects:
            documents = self.knowledge_base.retrieve(f"{topic} {obj}")
            all_documents.extend(documents)
        
        # 提取文档内容
        doc_contents = [doc.content for doc in all_documents]
        doc_sources = [doc.meta.get("file_path", "未知来源") for doc in all_documents]
        
        # 生成报告数据
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "topic": topic,
            "object_count": len(objects),
            "background": self._generate_background(topic),
            "dimensions": self._extract_dimensions(doc_contents, objects),
            "objects": self._extract_object_analysis(doc_contents, objects),
            "conclusion": self._generate_comparison_conclusion(objects),
            "sources": doc_sources
        }
        
        # 生成报告
        return self.template_engine.generate_report("comparison_report", data)
    
    def generate_project_report(self, project_name, period, query):
        """生成项目汇报报告"""
        # 检索相关文档
        documents = self.knowledge_base.retrieve(query)
        
        # 提取文档内容
        doc_contents = [doc.content for doc in documents]
        doc_sources = [doc.meta.get("file_path", "未知来源") for doc in documents]
        
        # 生成报告数据
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "project_name": project_name,
            "period": period,
            "overview": self._generate_project_overview(doc_contents),
            "progress": self._extract_progress(doc_contents),
            "issues": self._extract_issues(doc_contents),
            "next_steps": self._generate_next_steps(doc_contents),
            "sources": doc_sources
        }
        
        # 生成报告
        return self.template_engine.generate_report("project_report", data)
    
    def generate_incident_report(self, incident_name, incident_time, query):
        """生成故障/事件分析报告"""
        # 检索相关文档
        documents = self.knowledge_base.retrieve(query)
        
        # 提取文档内容
        doc_contents = [doc.content for doc in documents]
        doc_sources = [doc.meta.get("file_path", "未知来源") for doc in documents]
        
        # 生成报告数据
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "incident_name": incident_name,
            "incident_time": incident_time,
            "overview": self._generate_incident_overview(doc_contents),
            "root_causes": self._extract_root_causes(doc_contents),
            "impact": self._extract_impact(doc_contents),
            "solutions": self._extract_solutions(doc_contents),
            "preventive_measures": self._generate_preventive_measures(doc_contents),
            "sources": doc_sources
        }
        
        # 生成报告
        return self.template_engine.generate_report("incident_report", data)
    
    def export_report(self, content, output_path, format):
        """导出报告"""
        return self.exporter.export(content, output_path, format)
    
    # 辅助方法
    def _generate_summary(self, doc_contents):
        """生成摘要"""
        # 这里可以使用LLM生成摘要
        return "文档的核心内容摘要..."
    
    def _extract_key_points(self, doc_contents):
        """提取关键要点"""
        return ["要点1", "要点2", "要点3"]
    
    def _generate_conclusion(self, doc_contents):
        """生成结论"""
        return "基于文档内容的结论..."
    
    def _generate_background(self, topic):
        """生成背景信息"""
        return f"关于{topic}的背景信息..."
    
    def _extract_findings(self, doc_contents, topic):
        """提取发现"""
        return [
            {"title": "发现1", "content": "详细内容..."},
            {"title": "发现2", "content": "详细内容..."}
        ]
    
    def _generate_analysis(self, doc_contents):
        """生成分析"""
        return "详细分析内容..."
    
    def _extract_dimensions(self, doc_contents, objects):
        """提取对比维度"""
        return [
            {
                "name": "维度1",
                "objects": [
                    {"name": obj, "value": "值"} for obj in objects
                ]
            }
        ]
    
    def _extract_object_analysis(self, doc_contents, objects):
        """提取对象分析"""
        return [
            {
                "name": obj,
                "advantages": "优势...",
                "disadvantages": "劣势..."
            } for obj in objects
        ]
    
    def _generate_comparison_conclusion(self, objects):
        """生成对比结论"""
        return "对比分析结论..."
    
    def _generate_project_overview(self, doc_contents):
        """生成项目概述"""
        return "项目概述..."
    
    def _extract_progress(self, doc_contents):
        """提取进度"""
        return [
            {"milestone": "里程碑1", "status": "完成", "details": "详细信息..."}
        ]
    
    def _extract_issues(self, doc_contents):
        """提取问题"""
        return [
            {"name": "问题1", "description": "描述...", "solution": "解决方案..."}
        ]
    
    def _generate_next_steps(self, doc_contents):
        """生成下一步计划"""
        return "下一步计划..."
    
    def _generate_incident_overview(self, doc_contents):
        """生成事件概述"""
        return "事件概述..."
    
    def _extract_root_causes(self, doc_contents):
        """提取根因"""
        return [
            {"name": "根因1", "analysis": "详细分析..."}
        ]
    
    def _extract_impact(self, doc_contents):
        """提取影响"""
        return "影响范围..."
    
    def _extract_solutions(self, doc_contents):
        """提取解决方案"""
        return [
            {"name": "解决方案1", "description": "详细描述..."}
        ]
    
    def _generate_preventive_measures(self, doc_contents):
        """生成预防措施"""
        return "预防措施..."
