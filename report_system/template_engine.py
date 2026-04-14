# 报告模板引擎
import os
import jinja2

class MockGenerator:
    def run(self, prompt):
        return {"replies": [prompt]}

class TemplateEngine:
    def __init__(self):
        self.generator = MockGenerator()
        self.templates = self._load_templates()
    
    def _load_templates(self):
        """加载报告模板"""
        templates = {}
        template_dir = "templates"
        if not os.path.exists(template_dir):
            os.makedirs(template_dir)
            # 创建默认模板
            self._create_default_templates(template_dir)
        
        # 加载模板
        for template_name in os.listdir(template_dir):
            if template_name.endswith(".md"):
                template_key = template_name.split(".")[0]
                with open(os.path.join(template_dir, template_name), 'r', encoding='utf-8') as f:
                    templates[template_key] = f.read()
        return templates
    
    def _create_default_templates(self, template_dir):
        """创建默认报告模板"""
        # 摘要报告模板
        summary_template = """# 文档摘要报告

## 基本信息
- **报告生成时间**: {{ timestamp }}
- **文档数量**: {{ doc_count }}

## 核心摘要
{{ summary }}

## 关键要点
{% for point in key_points %}
- {{ point }}
{% endfor %}

## 结论
{{ conclusion }}

## 引用来源
{% for source in sources %}
- {{ source }}
{% endfor %}
"""
        
        # 调研报告模板
        research_template = """# 专题调研报告

## 基本信息
- **报告生成时间**: {{ timestamp }}
- **调研主题**: {{ topic }}
- **参考文档数量**: {{ doc_count }}

## 调研背景
{{ background }}

## 核心发现
{% for finding in findings %}
### {{ finding.title }}
{{ finding.content }}
{% endfor %}

## 分析与讨论
{{ analysis }}

## 建议与结论
{{ conclusion }}

## 引用来源
{% for source in sources %}
- {{ source }}
{% endfor %}
"""
        
        # 对比分析报告模板
        comparison_template = """# 对比分析报告

## 基本信息
- **报告生成时间**: {{ timestamp }}
- **对比主题**: {{ topic }}
- **对比对象数量**: {{ object_count }}

## 对比背景
{{ background }}

## 对比维度
{% for dimension in dimensions %}
### {{ dimension.name }}
{% for obj in dimension.objects %}
- **{{ obj.name }}**: {{ obj.value }}
{% endfor %}
{% endfor %}

## 优劣势分析
{% for obj in objects %}
### {{ obj.name }}
- **优势**: {{ obj.advantages }}
- **劣势**: {{ obj.disadvantages }}
{% endfor %}

## 结论与建议
{{ conclusion }}

## 引用来源
{% for source in sources %}
- {{ source }}
{% endfor %}
"""
        
        # 项目汇报报告模板
        project_template = """# 项目汇报报告

## 基本信息
- **报告生成时间**: {{ timestamp }}
- **项目名称**: {{ project_name }}
- **汇报周期**: {{ period }}

## 项目概述
{{ overview }}

## 进度与成果
{% for item in progress %}
- **{{ item.milestone }}**: {{ item.status }}
  {{ item.details }}
{% endfor %}

## 问题与挑战
{% for issue in issues %}
- **{{ issue.name }}**: {{ issue.description }}
  **解决方案**: {{ issue.solution }}
{% endfor %}

## 下一步计划
{{ next_steps }}

## 引用来源
{% for source in sources %}
- {{ source }}
{% endfor %}
"""
        
        # 事件分析报告模板
        incident_template = """# 故障/事件分析报告

## 基本信息
- **报告生成时间**: {{ timestamp }}
- **事件名称**: {{ incident_name }}
- **发生时间**: {{ incident_time }}

## 事件概述
{{ overview }}

## 根因分析
{% for cause in root_causes %}
### {{ cause.name }}
{{ cause.analysis }}
{% endfor %}

## 影响范围
{{ impact }}

## 解决方案
{% for solution in solutions %}
- **{{ solution.name }}**: {{ solution.description }}
{% endfor %}

## 预防措施
{{ preventive_measures }}

## 引用来源
{% for source in sources %}
- {{ source }}
{% endfor %}
"""
        
        # 写入模板文件
        templates = {
            "summary_report": summary_template,
            "research_report": research_template,
            "comparison_report": comparison_template,
            "project_report": project_template,
            "incident_report": incident_template
        }
        
        for name, content in templates.items():
            with open(os.path.join(template_dir, f"{name}.md"), 'w', encoding='utf-8') as f:
                f.write(content)
    
    def generate_report(self, report_type, data):
        """生成报告"""
        if report_type not in self.templates:
            raise ValueError(f"Unknown report type: {report_type}")
        
        # 使用Jinja2渲染模板
        template = jinja2.Template(self.templates[report_type])
        rendered = template.render(**data)
        
        # 使用LLM优化报告内容
        prompt = f"请优化以下报告内容，使其更加专业、清晰、结构合理：\n\n{rendered}"
        result = self.generator.run(prompt=prompt)
        
        return result["replies"][0]
