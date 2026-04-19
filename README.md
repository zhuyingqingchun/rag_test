# RAG 检索与报告生成系统

## 项目目标

> **构建一个面向本地文档资料的 RAG 检索与报告生成系统，支持稀疏检索、向量检索、混合检索与高级检索增强，并通过离线评测与证据置信度控制，提高回答与报告生成的可靠性。**

本项目不是通用聊天机器人，而是**"拿你的本地知识资料，先检索，再生成报告"的本地知识助手**。标准输入是放进 `rag_minimal/data/input_docs/` 的 PDF / Markdown / TXT 文档；标准输出不是一句简短回答，而是**带证据支撑的结构化报告**。

### 核心能力

| 能力 | 说明 | 当前状态 |
|------|------|----------|
| **知识检索** | 从本地文档库中检索最相关证据，输出 top-k chunk、来源文件、分数、检索方式 | ✅ 已完成（bm25/tfidf/keyword/vector/hybrid） |
| **证据化报告** | 基于检索结果生成摘要/调研/对比/项目/故障分析等结构化报告 | ✅ 已完成（5 种模板） |
| **质量控制** | 检索可靠性评估、证据充分性判断、abstain 决策、权重稳定性分析 | 🔄 进行中（评测框架已建立，负样本测试待补充） |

### 系统架构演进

```
第一层：最小闭环（已完成）
  导入文档 → 切分 chunk → 检索证据 → 生成报告

第二层：检索能力升级（已完成）
  单一检索 → 多路检索 → 混合检索 → 离线评测 → 权重扫描

第三层：高级检索增强（已完成基础）
  单次查询 → query rewrite → RRF 融合 → rerank → abstain 决策

第四层：质量闭环（进行中）
  高级检索统一评测 → abstain 负样本测试 → rerank benchmark → 报告质量一体化评估
```

### 当前知识库覆盖

| 类别 | 示例文件 |
|------|----------|
| LLM / NLP 基础 | transformer.pdf, bert.pdf, gpt.pdf, gpt2.pdf, gpt3.pdf |
| RAG 相关 | rag_introduction.md, signal_rag_guide.md |
| 通用算法 | hello-algo_1.3.0_zh_cpp.pdf |
| 垂直领域 | 基于人工智能的飞机电动舵机故障诊断方法研究_王剑.pdf |

---

## 项目概述

这是一个完整的 RAG（Retrieval-Augmented Generation）检索与报告生成系统，支持多种检索策略、多模板报告生成和离线评估。系统基于本地 GPU 环境运行，使用 Qwen 大语言模型生成结构化报告。

---

## 系统架构

```
refrag/
├── rag_minimal/              # RAG 核心系统（当前主项目）
├── qwen_service/             # Qwen 模型服务（本地 API）
├── report_system/            # 基于 Haystack 的报告系统（早期版本）
├── plan_patch/               # 补丁文件目录
├── configs/                  # 配置文件
├── scripts/                  # 工具脚本
├── src/                      # 源码（训练/推理/信号处理）
├── tests/                    # 测试代码
├── templates/                # 模板文件
├── docs/                     # 文档
├── outputs/                  # 输出结果
└── data/                     # 数据目录
```

---

## 核心模块说明

### 1. `rag_minimal/` - RAG 最小闭环系统（主项目）

当前正在开发的核心项目，实现了完整的 RAG 检索和报告生成流程。

```
rag_minimal/
├── data/
│   ├── input_docs/           # 输入文档目录（PDF/MD/TXT）
│   ├── processed/            # 处理结果目录
│   │   ├── vector_index/     # 向量索引（FAISS）
│   │   ├── eval_docs_chunks.jsonl  # 切分后的文档块
│   │   ├── retrieval_eval_report.json  # 检索评估报告
│   │   ├── advanced_retrieved_context_vnext.json  # 高级检索结果
│   │   └── hybrid_weight_scan.json  # 混合权重扫描结果
│   ├── eval/                 # 评估数据集
│   │   ├── retrieval_eval.json  # 检索评估样例（冒烟测试）
│   │   ├── retrieval_eval_formal_v1.json  # 正式评测集（覆盖多主题）
│   │   └── advanced_abstain_thresholds.example.json  # abstain 阈值配置示例
│   └── runs/                 # 运行记录目录
├── 核心模块
│   ├── ingest.py             # 文档导入（支持 txt/md/pdf）
│   ├── chunk.py              # 文本切分（段落/字符/句子）
│   ├── retrieve.py           # 检索模块（BM25/TF-IDF/Keyword/Vector/Hybrid）
│   ├── vector_store.py       # 向量索引构建与查询（Sentence-Transformers + FAISS）
│   ├── build_vector_index.py # 向量索引构建脚本
│   ├── generate_report.py    # 报告生成（调用 Qwen API）
│   └── pipeline_demo.py      # 完整流程演示
├── 高级检索
│   ├── query_rewrite.py      # 多查询改写（支持 RAG/Transformer/BERT/GPT/舵机诊断等主题）
│   ├── rerank.py             # 结果重排序（CrossEncoder + 启发式 rerank）
│   └── advanced_retrieve.py  # 高级检索（rewrite + RRF + rerank + abstain）
├── 评估模块
│   ├── evaluate_main.py      # 主评估流程（检索+报告质量）
│   ├── evaluate_retrieval.py # 检索离线评估（对比 bm25/vector/hybrid）
│   ├── evaluate_advanced_retrieval.py  # 高级检索评测（rewrite/rerank/abstain）
│   ├── evaluate_report.py    # 报告质量评估
│   ├── evaluate_template_matrix.py  # 多模板对比评估
│   └── tune_hybrid_weights.py  # 混合检索权重扫描
├── 配置与模板
│   ├── report_templates.py   # 报告模板（summary/research/comparison/project/incident）
│   ├── text_utils.py         # 文本处理工具
│   └── README.md             # 模块说明
└── 其他
    ├── logs/                 # 日志目录
    └── models/               # 模型缓存目录
```

**检索方法支持：**
- `bm25` - BM25 稀疏检索
- `tfidf` - TF-IDF 余弦相似度
- `keyword` - 关键词重叠匹配
- `vector` - 向量语义检索（BGE-M3 + FAISS）
- `hybrid` - 混合检索（四路加权融合）

**高级检索功能：**
- `query_rewrite` - 多查询改写（按主题生成候选查询）
- `RRF 融合` - Reciprocal Rank Fusion 多查询结果融合
- `rerank` - 结果重排序（CrossEncoder 或启发式 rerank）
- `abstain` - 置信度不足时拒绝回答机制
- `confidence_detail` - 双层置信度（RRF 证据强度 + 最终排序强度）

**报告模板支持：**
- `summary` - 文档摘要报告
- `research` - 专题调研报告
- `comparison` - 对比分析报告
- `project` - 项目汇报报告
- `incident` - 故障/事件分析报告

### 2. `qwen_service/` - Qwen 模型服务

本地 Qwen 大模型 API 服务，提供 OpenAI 兼容接口。

```
qwen_service/
├── config.py                 # 服务配置
├── run_server.py             # 启动服务
├── stop_server.py            # 停止服务
├── status_server.py          # 查看服务状态
├── test_openai_client.py     # 客户端测试
├── self_check.py             # 自检脚本
├── utils.py                  # 工具函数
└── logs/                     # 服务日志
```

**服务信息：**
- 端口：8000
- 接口：OpenAI 兼容 API
- 模型：next80b_fp8
- 框架：vLLM

### 3. `report_system/` - 基于 Haystack 的报告系统（早期版本）

```
report_system/
├── __init__.py
├── config.py                 # 配置管理
├── knowledge_base.py         # 知识库管理
├── template_engine.py        # 模板引擎
├── export.py                 # 导出模块
└── main.py                   # 主入口
```

### 4. `plan_patch/` - 补丁文件目录

存放各阶段开发的补丁文件，用于代码升级和版本管理。

### 5. `src/` - 源码目录

```
src/
├── data/                     # 数据处理
├── models/                   # 模型定义
├── training/                 # 训练代码
├── inference/                # 推理代码
├── signal_rag/               # 信号处理 RAG
└── utils/                    # 工具函数
```

---

## 硬件环境

### GPU 配置

| GPU ID | 型号 | 显存 | 用途 |
|--------|------|------|------|
| GPU 0 | NVIDIA RTX PRO 6000 Blackwell | 97.9 GB | vLLM 服务（Qwen 模型） |
| GPU 1 | NVIDIA RTX PRO 6000 Blackwell | 97.9 GB | 向量检索（BGE-M3 编码） |
| GPU 2 | NVIDIA RTX PRO 6000 Blackwell | 97.9 GB | 空闲 |
| GPU 3 | NVIDIA RTX PRO 6000 Blackwell | 97.9 GB | 空闲 |

**总计：4 × 97.9 GB = 391.6 GB 显存**

### Conda 环境

- **环境名称**：`swdtorch12`
- **Python 版本**：3.11
- **路径**：`/home/a/miniconda3/envs/swdtorch12`
- **环境说明文件**：`remi.md`

---

## 模型信息

### 1. Embedding 模型（向量检索）

- **模型名称**：`BAAI/bge-m3`
- **本地路径**：`/mnt/PRO6000_disk/models/BAAI/bge-m3`
- **模型大小**：2.2 GB
- **向量维度**：1024
- **下载方式**：魔塔社区（ModelScope）
- **用途**：文本向量化，用于向量检索和混合检索

### 2. LLM 模型（报告生成）

- **模型名称**：`next80b_fp8`
- **服务地址**：`http://127.0.0.1:8000/v1`
- **框架**：vLLM
- **精度**：FP8
- **用途**：结构化报告生成

---

## 数据说明

### 输入文档（`rag_minimal/data/input_docs/`）

| 文件名 | 类型 | 说明 |
|--------|------|------|
| `transformer.pdf` | PDF | Transformer 模型介绍 |
| `bert.pdf` | PDF | BERT 模型介绍 |
| `gpt.pdf` | PDF | GPT 模型介绍 |
| `gpt2.pdf` | PDF | GPT-2 模型介绍 |
| `gpt3.pdf` | PDF | GPT-3 模型介绍 |
| `hello-algo_1.3.0_zh_cpp.pdf` | PDF | 算法教程（中文版） |
| `signal_rag_guide.md` | MD | 信号处理 RAG 指南 |
| `rag_introduction.md` | MD | RAG 系统简介 |
| `基于人工智能的飞机电动舵机故障诊断方法研究_王剑.pdf` | PDF | 学术论文（12MB） |

### 评估数据（`rag_minimal/data/eval/`）

- `retrieval_eval.json` - 检索评估样例（3 个测试用例，冒烟测试）
- `retrieval_eval_formal_v1.json` - 正式评测集（覆盖 RAG/Transformer/BERT/GPT/算法/舵机诊断等 12 个用例）
- `advanced_abstain_thresholds.example.json` - abstain 阈值配置示例

### 处理结果（`rag_minimal/data/processed/`）

- `eval_docs_chunks.jsonl` - 切分后的文档块（763 个 chunk）
- `vector_index/` - 向量索引（FAISS 格式）
- `retrieval_eval_report.json` - 检索评估报告
- `hybrid_weight_scan.json` - 混合权重扫描结果

---

## 快速开始

### 1. 激活环境

```bash
conda activate swdtorch12
```

### 2. 启动 Qwen 服务

```bash
cd qwen_service
python run_server.py
```

### 3. 构建向量索引

```bash
cd rag_minimal
CUDA_VISIBLE_DEVICES=1 python vector_store.py build \
  --chunks ./data/processed/eval_docs_chunks.jsonl \
  --output ./data/processed/vector_index
```

### 4. 运行检索评估

```bash
cd rag_minimal
CUDA_VISIBLE_DEVICES=1 python evaluate_retrieval.py \
  --eval-file ./data/eval/retrieval_eval.json \
  --chunks ./data/processed/eval_docs_chunks.jsonl \
  --methods bm25,vector,hybrid \
  --top-k 5 \
  --index-dir ./data/processed/vector_index \
  --embedding-model /mnt/PRO6000_disk/models/BAAI/bge-m3 \
  --output ./data/processed/retrieval_eval_report.json
```

### 5. 运行完整流程

```bash
cd rag_minimal
CUDA_VISIBLE_DEVICES=1 python pipeline_demo.py \
  --query "RAG 系统的核心组成是什么？" \
  --input ./data/input_docs \
  --chunk-method paragraph \
  --retrieve-method hybrid \
  --embedding-model BAAI/bge-m3 \
  --top-k 5 \
  --save-run
```

### 6. 运行高级检索（rewrite + RRF + rerank + abstain）

```bash
cd rag_minimal
CUDA_VISIBLE_DEVICES=1 python advanced_retrieve.py \
  --query "RAG 系统的核心组成是什么？" \
  --chunks ./data/processed/eval_docs_chunks.jsonl \
  --method hybrid \
  --index-dir ./data/processed/vector_index \
  --embedding-model /mnt/PRO6000_disk/models/BAAI/bge-m3 \
  --rewrite \
  --rewrite-max-queries 4 \
  --rerank \
  --rerank-top-n 10 \
  --output ./data/processed/advanced_retrieved_context.json
```

### 7. 运行高级检索评测

```bash
cd rag_minimal
CUDA_VISIBLE_DEVICES=1 python evaluate_advanced_retrieval.py \
  --eval-file ./data/eval/retrieval_eval_formal_v1.json \
  --chunks ./data/processed/eval_docs_chunks.jsonl \
  --method hybrid \
  --top-k 5 \
  --index-dir ./data/processed/vector_index \
  --embedding-model /mnt/PRO6000_disk/models/BAAI/bge-m3 \
  --rewrite \
  --rerank \
  --output ./data/processed/advanced_retrieval_eval_report.json
```

---

## 依赖安装

### 基础依赖

```bash
pip install requests openai
```

### 稀疏检索

```bash
pip install rank-bm25 scikit-learn
```

### 向量检索

```bash
pip install sentence-transformers faiss-cpu
```

### PDF 支持

```bash
pip install PyMuPDF
```

### 模型下载（魔塔社区）

```bash
pip install modelscope
modelscope download --model BAAI/bge-m3 --local_dir /mnt/PRO6000_disk/models/BAAI/bge-m3
```

---

## 评估指标

### 检索评估

- **Hit@K** - 前 K 个结果中是否有命中
- **Precision@K** - 前 K 个结果的精确率
- **Recall@K** - 召回率
- **MRR@K** - 平均倒数排名

### 高级检索评估

- **answered_rate** - 没有 abstain 的比例
- **answered_hit_rate** - 在没有 abstain 的样本中，回答命中的比例
- **useful_answer_rate** - 既没有 abstain，又成功命中的比例
- **abstain_rate** - abstain 的比例

### 报告评估

- **结构完整率** - 报告结构是否完整
- **引用覆盖率** - 证据引用是否充分
- **事实一致性** - 报告内容是否与证据一致
- **综合得分** - 加权综合评分

---

## Git 仓库

- **远程地址**：`https://github.com/zhuyingqingchun/rag_test.git`
- **默认分支**：`main`
- **当前状态**：已推送最新代码

---

## 项目历史

1. **第一阶段**：基于 Haystack 构建文档检索和报告生成系统（`report_system/`）
2. **第二阶段**：实现 Qwen 模型服务（`qwen_service/`）
3. **第三阶段**：开发 RAG 最小闭环系统（`rag_minimal/`）
4. **第四阶段**：添加 PDF 支持和评估实验
5. **第五阶段**：代码审查和改进补丁
6. **第六阶段**：多模板报告生成
7. **第七阶段**：多模板对比评估
8. **第八阶段**：混合检索升级（BM25 + TF-IDF + Keyword + Vector）
9. **第九阶段**：检索评估和混合检索稳定性扫描
10. **第十阶段**：高级检索（rewrite + RRF + rerank + abstain）+ 置信度双层化 + 高级检索评测

---

## 下一步计划

### 检索质量
- [ ] abstain 的正式负样本测试（negative / hard negative 样本）
- [ ] rerank 的正式 benchmark（CrossEncoder vs 启发式）
- [ ] query rewrite 模型化（LLM-based 改写）

### 评估闭环
- [ ] 高级检索统一评测（rewrite/rerank/abstain 联合评估）
- [ ] 报告质量和检索质量的一体化闭环
- [ ] 幻觉专项评估

### 产品化
- [ ] 确定主场景（通用知识库 vs 垂直领域诊断）
- [ ] 面向用户的固定输出界面/流程
- [ ] 单一明确场景收敛

---

## 联系方式

- **项目维护者**：用户
- **代码审查**：通过补丁文件进行版本管理
- **远程仓库**：`https://github.com/zhuyingqingchun/rag_test.git`
