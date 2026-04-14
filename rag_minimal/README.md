# RAG 最小闭环系统

## 系统概述

这是一个基于 Python 的最小 RAG（Retrieval-Augmented Generation）系统，用于文档检索和结构化报告生成。

## 功能特性

- ✅ 文档导入（支持 txt、md、pdf）
- ✅ 文本切分（按段落、字符、句子）
- ✅ 基础检索（BM25、TF-IDF、关键词匹配）
- ✅ 结构化报告生成（调用 Qwen API）
- ✅ 完整流程自动化（pipeline_demo.py）
- ✅ 运行记录留档

## 目录结构

```
rag_minimal/
├── data/
│   ├── input_docs/      # 输入文档目录
│   ├── processed/       # 处理结果目录
│   └── runs/            # 运行记录目录
├── ingest.py            # 文档导入模块
├── chunk.py             # 文本切分模块
├── retrieve.py          # 检索模块
├── generate_report.py   # 报告生成模块
├── pipeline_demo.py     # 完整流程脚本
├── qwen_service_readme.md  # Qwen 服务说明
└── README.md            # 本文档
```

## 安装依赖

```bash
conda activate swdtorch12

# 基础依赖
pip install requests openai

# 检索依赖（可选）
pip install rank-bm25 scikit-learn

# PDF 支持（可选）
pip install PyPDF2
```

## 快速开始

### 1. 准备测试文档

将文档放入 `data/input_docs/` 目录，支持格式：
- `.txt` - 纯文本文件
- `.md` - Markdown 文件
- `.pdf` - PDF 文件（需要 PyPDF2）

### 2. 运行完整流程

```bash
python pipeline_demo.py \
    --query "RAG 系统的核心组成是什么？" \
    --input ./data/input_docs \
    --chunk-method paragraph \
    --retrieve-method bm25 \
    --top-k 5 \
    --save-run
```

### 3. 查看结果

- 检索结果：`data/processed/retrieved_context.json`
- 生成报告：`data/processed/report_*.md`
- 运行记录：`data/runs/<timestamp>/`

## 模块说明

### 1. 文档导入 (ingest.py)

```bash
# 导入单个文件
python ingest.py -i ./data/input_docs/doc.md -o ./data/processed/docs_raw.jsonl

# 导入整个目录
python ingest.py -i ./data/input_docs -o ./data/processed/docs_raw.jsonl
```

### 2. 文本切分 (chunk.py)

```bash
python chunk.py \
    -i ./data/processed/docs_raw.jsonl \
    -o ./data/processed/docs_chunks.jsonl \
    -m paragraph
```

切分方法：
- `paragraph` - 按段落切分
- `char` - 按字符数切分（支持重叠）
- `sentence` - 按句子切分

### 3. 检索 (retrieve.py)

```bash
python retrieve.py \
    -q "RAG 系统的核心组成" \
    -c ./data/processed/docs_chunks.jsonl \
    -m bm25 \
    -k 5 \
    -o ./data/processed/retrieved_context.json
```

检索方法：
- `bm25` - BM25 算法（推荐）
- `tfidf` - TF-IDF 算法
- `keyword` - 关键词匹配

### 4. 报告生成 (generate_report.py)

```bash
python generate_report.py \
    -q "RAG 系统的核心组成" \
    -c ./data/processed/retrieved_context.json \
    -m next80b_fp8 \
    -b http://127.0.0.1:8000/v1 \
    -o ./data/processed/report.md
```

## 结构化报告模板

生成的报告包含以下部分：

```markdown
# 报告标题

## 一、任务说明
- 用户问题：
- 生成时间：

## 二、检索摘要
- 共召回片段数：
- 主要来源：

## 三、关键内容整理
1. ...
2. ...
3. ...

## 四、综合分析
...

## 五、结论
...

## 六、参考片段
- [1] ...
- [2] ...
- [3] ...
```

## Qwen 服务配置

确保 Qwen 服务正在运行：

```bash
cd /mnt/PRO6000_disk/swd/servo_0/refrag/qwen_service
python run_server.py
```

服务配置文件：`qwen_service/service_config.json`

## 环境变量

如果系统设置了代理，可能导致本地请求失败：

```bash
unset http_proxy https_proxy ALL_PROXY all_proxy
```

或在代码中禁用代理：

```python
session = requests.Session()
session.trust_env = False
```

## 完整流程示例

```bash
# 1. 准备文档
echo "RAG 是检索增强生成技术..." > ./data/input_docs/test.md

# 2. 运行完整流程
python pipeline_demo.py \
    --query "什么是 RAG？" \
    --input ./data/input_docs \
    --chunk-method paragraph \
    --retrieve-method bm25 \
    --top-k 3 \
    --save-run

# 3. 查看报告
cat ./data/processed/report.md
```

## 项目结构

```
refrag/
├── qwen_service/        # Qwen 模型服务
│   ├── run_server.py
│   ├── stop_server.py
│   ├── status_server.py
│   ├── test_openai_client.py
│   ├── self_check.py
│   ├── config.py
│   ├── utils.py
│   └── service_config.json
└── rag_minimal/         # RAG 最小闭环系统
    ├── ingest.py
    ├── chunk.py
    ├── retrieve.py
    ├── generate_report.py
    ├── pipeline_demo.py
    └── data/
        ├── input_docs/
        ├── processed/
        └── runs/
```

## 下一步扩展

- [ ] 向量数据库集成（FAISS、Milvus）
- [ ] 重排序（Rerank）优化
- [ ] 多格式导出（Word、PDF）
- [ ] Web 界面
- [ ] 批量处理
- [ ] 评估框架

## 许可证

MIT License
