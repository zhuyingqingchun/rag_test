#!/usr/bin/env python3
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 创建 PDF
c = canvas.Canvas("pdf_test.pdf", pagesize=A4)
width, height = A4

# 设置中文字体
from reportlab.pdfbase.pdfmetrics import registerFont
from reportlab.pdfbase.ttfonts import TTFont
registerFont(TTFont("SimSun", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))

# 标题
c.setFont("SimSun", 16)
c.drawString(100, height - 100, "PDF 测试文档")

# 章节 1
c.setFont("SimSun", 14)
c.drawString(100, height - 150, "第一章：RAG 简介")
c.setFont("SimSun", 12)
c.drawString(100, height - 180, "RAG（Retrieval-Augmented Generation）是一种结合检索与生成的智能问答技术。")

# 章节 2
c.setFont("SimSun", 14)
c.drawString(100, height - 230, "第二章：核心组件")
c.setFont("SimSun", 12)
c.drawString(100, height - 260, "RAG 系统包含三个核心组件：")
c.drawString(100, height - 280, "1. 检索器：从知识库中查找相关文档")
c.drawString(100, height - 300, "2. 知识库：存储外部信息的数据源")
c.drawString(100, height - 320, "3. 生成器：基于检索结果生成回答")

# 章节 3
c.setFont("SimSun", 14)
c.drawString(100, height - 370, "第三章：应用场景")
c.setFont("SimSun", 12)
c.drawString(100, height - 400, "RAG 技术广泛应用于：")
c.drawString(100, height - 420, "- 问答系统")
c.drawString(100, height - 440, "- 文档摘要")
c.drawString(100, height - 460, "- 知识库构建")
c.drawString(100, height - 480, "- 智能客服")

c.save()
print("PDF created successfully: pdf_test.pdf")
