# 信号数据RAG系统使用指南

## 概述

本文档介绍如何使用信号数据专用的RAG（检索增强生成）系统。该系统包含以下核心组件：

1. **信号编码器**：将时序信号转换为向量表示
2. **信号预处理器**：信号预处理和特征提取
3. **信号RAG系统**：基于向量的信号检索
4. **评估工具**：全面的RAG效果评估

## 系统架构

```
信号数据 → 预处理 → 编码器 → 向量数据库 → 相似度检索 → 结果返回
```

## 安装依赖

```bash
pip install torch numpy transformers scikit-learn rouge-score
```

## 快速开始

### 1. 训练信号编码器

```bash
# 使用合成数据训练（用于测试）
python scripts/train_signal_encoder.py \
    --use_synthetic \
    --output_dir outputs/signal_encoder \
    --num_epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4

# 使用真实数据训练
python scripts/train_signal_encoder.py \
    --data_path data/signals.npz \
    --output_dir outputs/signal_encoder \
    --num_epochs 100
```

### 2. 运行信号RAG演示

```bash
# 使用未训练的编码器（随机初始化）
python scripts/signal_rag_demo.py \
    --num_signals 100 \
    --top_k 5

# 使用训练好的编码器
python scripts/signal_rag_demo.py \
    --encoder_path outputs/signal_encoder/best_model.pt \
    --num_signals 100 \
    --save_db data/signal_database.npz
```

### 3. 评估信号RAG系统

```bash
python scripts/evaluate_signal_rag.py \
    --encoder_path outputs/signal_encoder/best_model.pt \
    --database_path data/signal_database.npz \
    --output signal_rag_evaluation.json \
    --num_query_signals 20
```

## 核心组件详解

### SignalEncoder（信号编码器）

```python
from signal_rag.signal_encoder import SignalEncoder

# 创建编码器
encoder = SignalEncoder(
    input_dim=1,        # 输入信号维度
    hidden_dim=256,     # LSTM隐藏层维度
    output_dim=768,     # 输出嵌入维度
    num_layers=4,       # LSTM层数
    dropout=0.1,
    use_attention=True  # 使用注意力机制
)

# 编码单个信号
import numpy as np
signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
embedding = encoder.encode_signal(signal)
print(f"Embedding shape: {embedding.shape}")  # (768,)

# 批量编码
signals = [signal1, signal2, signal3]
embeddings = encoder.encode_batch(signals)
print(f"Batch embeddings shape: {embeddings.shape}")  # (3, 768)
```

### SignalPreprocessor（信号预处理器）

```python
from signal_rag.signal_encoder import SignalPreprocessor

# 创建预处理器
preprocessor = SignalPreprocessor(
    sample_rate=1000,    # 采样率
    window_size=1024,    # 窗口大小
    hop_size=512,        # 步长
    normalize=True       # 标准化
)

# 分割信号
segments = preprocessor.segment_signal(signal)

# 提取特征
features = preprocessor.extract_features(signal)
print(features)
# {
#     'mean': 0.0,
#     'std': 0.707,
#     'max': 1.0,
#     'min': -1.0,
#     'rms': 0.707,
#     'peak_to_peak': 2.0,
#     'zero_crossing_rate': 0.02,
#     'dominant_freq': 10.0,
#     'spectral_energy': 500.0
# }
```

### SignalRAG（信号RAG系统）

```python
from signal_rag.signal_encoder import SignalRAG

# 创建RAG系统
rag = SignalRAG(
    encoder=encoder,
    preprocessor=preprocessor,
    top_k=5
)

# 添加信号到数据库
signals = [signal1, signal2, signal3]
metadata = [
    {'type': 'normal', 'description': '正常运行'},
    {'type': 'fault', 'description': '轴承故障'},
    {'type': 'fault', 'description': '齿轮故障'}
]
rag.add_signals(signals, metadata)

# 检索相似信号
query_signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
results = rag.retrieve(query_signal)

# 处理结果
for idx, similarity, signal, meta in results:
    print(f"Index: {idx}, Similarity: {similarity:.4f}")
    print(f"Type: {meta['type']}, Description: {meta['description']}")

# 保存数据库
rag.save_database('signal_database.npz')

# 加载数据库
rag.load_database('signal_database.npz')
```

## 评估指标

### 检索指标

1. **Retrieval Accuracy（检索准确率）**
   - 定义：正确检索到相关信号的比例
   - 计算：正确检索次数 / 总查询次数

2. **Retrieval Precision（检索精确率）**
   - 定义：检索结果中相关信号的比例
   - 计算：相关信号数 / 检索信号总数

3. **Retrieval Recall（检索召回率）**
   - 定义：所有相关信号中被成功检索的比例
   - 计算：检索到的相关信号数 / 总相关信号数

4. **MRR（Mean Reciprocal Rank）**
   - 定义：平均倒数排名
   - 计算：Σ(1/rank_i) / N

### 生成指标（用于文本RAG）

1. **ROUGE-L**：最长公共子序列匹配度
2. **Answer Relevance**：回答与查询的相关性
3. **Answer Faithfulness**：回答与文档的一致性

### 效率指标

1. **Average Retrieval Time**：平均检索时间（ms）
2. **Average Generation Time**：平均生成时间（ms）
3. **Total Latency**：总延迟（ms）

### 嵌入质量指标

1. **Intra-class Distance**：类内平均距离
2. **Inter-class Distance**：类间平均距离
3. **Embedding Quality**：类间距离 / 类内距离

## 数据格式

### 训练数据格式（NPZ文件）

```python
import numpy as np

# 创建训练数据
signals = np.array([signal1, signal2, ...])  # shape: (num_samples, seq_len)
labels = np.array([0, 1, 0, 2, ...])          # shape: (num_samples,)

# 保存
np.savez('signals.npz', signals=signals, labels=labels)

# 加载
data = np.load('signals.npz')
signals = data['signals']
labels = data['labels']
```

### 信号数据库格式

```python
# 保存
rag.save_database('signal_database.npz')

# 加载
rag.load_database('signal_database.npz')
# 包含：embeddings, signals, metadata
```

## 高级用法

### 自定义损失函数

```python
from signal_rag.signal_encoder import ContrastiveLoss, TripletLoss

# 对比损失
contrastive_loss = ContrastiveLoss(temperature=0.07)
loss = contrastive_loss(embeddings, labels)

# 三元组损失
triplet_loss = TripletLoss(margin=1.0)
loss = triplet_loss(anchor, positive, negative)
```

### 批量处理

```python
# 大批量信号处理
from torch.utils.data import DataLoader

# 创建数据集
dataset = SignalDataset(signals, labels, preprocessor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 批量编码
for batch_signals, batch_labels in dataloader:
    embeddings = encoder(batch_signals)
    # 处理嵌入...
```

### 增量更新

```python
# 添加新信号而不重新编码所有信号
new_signals = [new_signal1, new_signal2]
new_metadata = [{'type': 'new_type'}, {'type': 'new_type'}]
rag.add_signals(new_signals, new_metadata)
```

## 故障诊断应用示例

```python
# 1. 构建故障信号数据库
fault_types = ['normal', 'bearing_fault', 'gear_fault', 'imbalance', 'misalignment']
fault_signals = load_fault_signals()  # 加载故障信号

rag = SignalRAG(encoder, preprocessor, top_k=3)
rag.add_signals(fault_signals['signals'], fault_signals['metadata'])

# 2. 实时监测
while True:
    # 采集实时信号
    realtime_signal = acquire_signal()
    
    # 检索最相似的故障类型
    results = rag.retrieve(realtime_signal)
    
    # 诊断
    top_match = results[0]
    if top_match[1] > 0.9:  # 相似度阈值
        print(f"Detected: {top_match[3]['type']}")
        print(f"Confidence: {top_match[1]:.2%}")
        print(f"Description: {top_match[3]['description']}")
    
    time.sleep(1)
```

## 性能优化建议

1. **编码器优化**
   - 使用更大的隐藏层维度（512或1024）
   - 增加LSTM层数（6-8层）
   - 使用Transformer替代LSTM

2. **检索优化**
   - 使用FAISS进行高效相似度搜索
   - 建立索引加速检索
   - 使用向量量化减少存储

3. **批处理优化**
   - 使用更大的batch size
   - 启用混合精度训练
   - 使用DataLoader多进程加载

## 常见问题

### Q: 如何处理不同长度的信号？
A: 使用SignalPreprocessor进行分割和填充，或者使用全局平均池化。

### Q: 如何提高检索准确率？
A: 1) 使用更多训练数据；2) 增加编码器复杂度；3) 调整temperature参数；4) 使用 harder negative mining。

### Q: 如何保存和加载训练好的模型？
A: 使用torch.save()保存checkpoint，使用torch.load()加载。详见train_signal_encoder.py中的save_checkpoint和load逻辑。

### Q: 支持多通道信号吗？
A: 支持。设置input_dim为通道数（如3通道振动信号），信号形状为(seq_len, input_dim)。

## 参考文献

1. Contrastive Learning for Time Series: A Survey
2. Self-Supervised Learning for Time Series Analysis
3. Transformer-based Models for Time Series Representation Learning

## 联系与支持

如有问题或建议，请提交Issue或Pull Request。
