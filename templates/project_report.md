# 项目汇报报告

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
