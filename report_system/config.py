# 报告系统配置

class Config:
    # 模型配置
    LLM_API_KEY = "your_api_key_here"
    LLM_MODEL = "gpt-4"
    EMBEDDING_MODEL = "text-embedding-ada-002"
    
    # 检索配置
    TOP_K = 5
    TOP_P = 0.8
    
    # 存储配置
    INDEX_NAME = "report_system_index"
    
    # 报告配置
    REPORT_TEMPLATES = {
        "summary": "templates/summary_report.md",
        "research": "templates/research_report.md",
        "comparison": "templates/comparison_report.md",
        "project": "templates/project_report.md",
        "incident": "templates/incident_report.md"
    }
    
    # 导出配置
    EXPORT_FORMATS = ["markdown", "docx", "pdf"]
