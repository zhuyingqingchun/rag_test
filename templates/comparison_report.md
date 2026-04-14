# 对比分析报告

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
