# 为 rag_minimal 增加 PDF 支持的最小改动方案

## 目标

在不改动 `chunk.py`、`retrieve.py`、`generate_report.py`、`pipeline_demo.py` 主逻辑的前提下，给 `ingest.py` 增加 PDF 文本提取能力。

思路很简单：

- `txt / md` 继续按原方式读取
- 新增 `pdf` 分支
- 将 PDF 提取成纯文本后，仍然输出到 `docs_raw.jsonl`
- 后续流程完全复用现有逻辑

---

## 推荐方案

优先使用 **PyMuPDF**。

根据 PyMuPDF 官方文档，可以使用 `pymupdf.open(filename)` 打开 PDF 文档，并逐页处理页面内容；官方文本提取文档提供了页面文本提取做法。 citeturn391034view0turn391034view1

---

## 需要修改的文件

只建议先改这两个地方：

1. `ingest.py`
2. `requirements.txt` 或安装说明文档

---

## 第一步：安装依赖

```bash
/home/a/miniconda3/envs/swdtorch12/bin/pip install pymupdf
```

---

## 第二步：在 ingest.py 中增加 PDF 提取函数

建议新增如下函数：

```python
from pathlib import Path
import pymupdf


def read_pdf_text(file_path: str) -> str:
    text_parts = []
    doc = pymupdf.open(file_path)
    try:
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text_parts.append(page_text.strip())
    finally:
        doc.close()

    return "\n\n".join(part for part in text_parts if part)
```

---

## 第三步：在 ingest.py 中扩展文件类型分支

假设你当前已有类似逻辑：

```python
suffix = Path(file_path).suffix.lower()

if suffix in [".txt", ".md"]:
    text = read_text_file(file_path)
else:
    raise ValueError(...)
```

改成：

```python
suffix = Path(file_path).suffix.lower()

if suffix in [".txt", ".md"]:
    text = read_text_file(file_path)
elif suffix == ".pdf":
    text = read_pdf_text(file_path)
else:
    raise ValueError(f"暂不支持的文件类型: {suffix}")
```

---

## 第四步：建议写入额外元数据

为了后面方便排查，建议在 `docs_raw.jsonl` 中增加这些字段：

```json
{
  "doc_id": "xxx",
  "source": "/abs/path/demo.pdf",
  "file_type": "pdf",
  "title": "demo.pdf",
  "text": "..."
}
```

如果你已经有这些字段，就不用重复加。

---

## 第五步：空文本保护

有些 PDF 可能提取不到文本，所以要加保护：

```python
if not text or not text.strip():
    print(f"[WARN] PDF 未提取到有效文本: {file_path}")
    return None
```

写入 JSONL 前跳过空文档。

---

## 第六步：最小测试命令

先准备一个 PDF 放到：

```text
rag_minimal/data/input_docs/demo.pdf
```

然后执行：

```bash
cd /mnt/PRO6000_disk/swd/servo_0/refrag/rag_minimal

/home/a/miniconda3/envs/swdtorch12/bin/python ingest.py   -i ./data/input_docs   -o ./data/processed/docs_raw.jsonl
```

如果成功，再继续跑：

```bash
/home/a/miniconda3/envs/swdtorch12/bin/python chunk.py   -i ./data/processed/docs_raw.jsonl   -o ./data/processed/docs_chunks.jsonl   -m paragraph

/home/a/miniconda3/envs/swdtorch12/bin/python retrieve.py   -q "请总结 PDF 文档的核心内容"   -c ./data/processed/docs_chunks.jsonl   -m keyword   -k 3   -o ./data/processed/retrieved_context.json

unset http_proxy https_proxy ALL_PROXY all_proxy
/home/a/miniconda3/envs/swdtorch12/bin/python generate_report.py   -q "请总结 PDF 文档的核心内容"   -c ./data/processed/retrieved_context.json   -m next80b_fp8   -b http://localhost:8000/v1   -o ./data/processed/report_pdf.md
```

---

## 验收标准

完成后至少满足以下 5 条：

1. `ingest.py` 能识别 `.pdf`
2. PDF 能被提取为纯文本并写入 `docs_raw.jsonl`
3. `chunk.py` 能正常切分 PDF 提取结果
4. `retrieve.py` 能从 PDF chunk 中召回相关片段
5. `generate_report.py` 能根据召回内容生成报告

---

## 当前阶段不要做的事

这一轮先不要做：

- 不做 OCR
- 不做扫描版 PDF 识别
- 不做表格结构恢复
- 不做图像内容解析
- 不做多栏版面优化

先只支持“可直接提取文本的 PDF”。

---

## 下一版再考虑的增强

等最小版跑通后，再考虑：

1. PDF 元数据提取（标题、页数、作者）
2. 页码级引用
3. 扫描 PDF 的 OCR 兜底
4. 图表/表格提取
5. 混合文档目录批量导入

---

## 一句话结论

最自然的实现方式是：**只在 ingest.py 加一个 PDF 文本提取分支，后面的 RAG 流程完全不动。**  
这样改动最小，最容易验证，也最符合你现在这个最小闭环系统的推进节奏。
