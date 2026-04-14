# 文档摘要报告

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
