#!/usr/bin/env python3
from datasets import load_dataset

# 加载数据集
ds = load_dataset("TAAC2026/data_sample_1000")

# 查看数据集结构
print("数据集结构:")
print(ds)

# 保存一小部分数据 (例如前10个样本)
sample_size = 10
sampled_ds = ds["train"].select(range(sample_size))

# 保存为本地文件
sampled_ds.to_json("data_sample_10.jsonl")

print(f"已保存{sample_size}个样本到 data_sample_10.jsonl")
