# 故障/事件分析报告

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
