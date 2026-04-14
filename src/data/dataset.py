"""
REFRAG数据集处理模块
支持重建任务、课程学习、CPT、RL和SFT等不同训练阶段的数据处理
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from transformers import AutoTokenizer
import json
import random


class ReconstructionDataset(Dataset):
    """
    重建任务数据集
    目标：让解码器通过块嵌入重建原始上下文
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 4096,
        block_size: int = 32,
        num_blocks: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.block_size = block_size
        self.num_blocks = num_blocks
        
        # 加载数据
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[str]:
        """加载文本数据"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if 'text' in item:
                    data.append(item['text'])
                elif 'content' in item:
                    data.append(item['content'])
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        
        # 计算需要的块数量
        if self.num_blocks:
            # 固定块数量（课程学习）
            seq_len = self.num_blocks * self.block_size
            input_ids = input_ids[:seq_len]
            attention_mask = attention_mask[:seq_len]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }


class CurriculumDataset(Dataset):
    """
    课程学习数据集
    从单块重建逐步增加到多块重建
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 4096,
        block_size: int = 32,
        curriculum_stages: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
        current_stage: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.block_size = block_size
        self.curriculum_stages = curriculum_stages
        self.current_stage = current_stage
        
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[str]:
        """加载文本数据"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                text = item.get('text', item.get('content', ''))
                if len(text) > 100:  # 过滤太短的数据
                    data.append(text)
        return data
    
    def set_stage(self, stage: int):
        """设置当前课程阶段"""
        assert 0 <= stage < len(self.curriculum_stages)
        self.current_stage = stage
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # 当前阶段的块数量
        num_blocks = self.curriculum_stages[self.current_stage]
        target_length = num_blocks * self.block_size
        
        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=target_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone(),
            'num_blocks': num_blocks
        }


class CPTDataset(Dataset):
    """
    持续预训练数据集
    下一段预测任务
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        context_length: int = 2048,
        prediction_length: int = 2048,
        block_size: int = 32,
    ):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.total_length = context_length + prediction_length
        self.block_size = block_size
        
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[str]:
        """加载长文本数据"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                text = item.get('text', item.get('content', ''))
                if len(text) > self.total_length:
                    data.append(text)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # 随机选择起始位置
        if len(text) > self.total_length:
            start_pos = random.randint(0, len(text) - self.total_length)
            text = text[start_pos:start_pos + self.total_length]
        
        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=self.total_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)
        
        # 分割为上下文和预测目标
        context_ids = input_ids[:self.context_length]
        context_mask = attention_mask[:self.context_length]
        
        # 构建标签：上下文部分为-100（不计算loss），预测部分为实际token
        labels = torch.full_like(input_ids, -100)
        labels[self.context_length:] = input_ids[self.context_length:]
        
        return {
            'context_ids': context_ids,
            'context_mask': context_mask,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class RLDataset(Dataset):
    """
    RL训练数据集
    用于学习选择性压缩策略
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 4096,
        block_size: int = 32,
        num_samples_per_example: int = 4,  # GRPO的组大小
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.block_size = block_size
        self.num_samples_per_example = num_samples_per_example
        
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载数据"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                data.append(item)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 获取上下文和查询
        context = item.get('context', item.get('text', ''))
        query = item.get('query', item.get('question', ''))
        
        # Tokenize context
        context_tokens = self.tokenizer(
            context,
            max_length=self.max_length - 256,  # 预留空间给query和生成
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize query
        query_tokens = self.tokenizer(
            query,
            max_length=256,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'context_ids': context_tokens['input_ids'].squeeze(0),
            'context_mask': context_tokens['attention_mask'].squeeze(0),
            'query_ids': query_tokens['input_ids'].squeeze(0),
            'query_mask': query_tokens['attention_mask'].squeeze(0),
            'num_samples': self.num_samples_per_example
        }


class RAGDataset(Dataset):
    """
    RAG任务数据集
    检索增强生成
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_context_length: int = 2048,
        max_query_length: int = 256,
        max_answer_length: int = 512,
        block_size: int = 32,
    ):
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        self.max_query_length = max_query_length
        self.max_answer_length = max_answer_length
        self.block_size = block_size
        
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载RAG数据"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                data.append(item)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 获取检索到的段落
        retrieved_passages = item.get('retrieved_passages', item.get('passages', []))
        query = item.get('query', item.get('question', ''))
        answer = item.get('answer', item.get('target', ''))
        
        # 拼接检索段落
        context = ' '.join(retrieved_passages) if isinstance(retrieved_passages, list) else retrieved_passages
        
        # Tokenize
        context_tokens = self.tokenizer(
            context,
            max_length=self.max_context_length,
            truncation=True,
            return_tensors='pt'
        )
        
        query_tokens = self.tokenizer(
            query,
            max_length=self.max_query_length,
            truncation=True,
            return_tensors='pt'
        )
        
        answer_tokens = self.tokenizer(
            answer,
            max_length=self.max_answer_length,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'context_ids': context_tokens['input_ids'].squeeze(0),
            'context_mask': context_tokens['attention_mask'].squeeze(0),
            'query_ids': query_tokens['input_ids'].squeeze(0),
            'query_mask': query_tokens['attention_mask'].squeeze(0),
            'answer_ids': answer_tokens['input_ids'].squeeze(0),
            'answer_mask': answer_tokens['attention_mask'].squeeze(0)
        }


class MultiTurnDialogueDataset(Dataset):
    """
    多轮对话数据集
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_history_length: int = 2048,
        max_response_length: int = 512,
        block_size: int = 32,
        max_turns: int = 6,
    ):
        self.tokenizer = tokenizer
        self.max_history_length = max_history_length
        self.max_response_length = max_response_length
        self.block_size = block_size
        self.max_turns = max_turns
        
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载对话数据"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                data.append(item)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 获取对话历史
        history = item.get('history', item.get('turns', []))
        response = item.get('response', item.get('answer', ''))
        
        # 限制轮数
        if len(history) > self.max_turns:
            history = history[-self.max_turns:]
        
        # 构建历史文本
        history_text = ''
        for turn in history:
            if isinstance(turn, dict):
                user = turn.get('user', turn.get('question', ''))
                assistant = turn.get('assistant', turn.get('answer', ''))
                history_text += f"User: {user}\nAssistant: {assistant}\n"
            else:
                history_text += str(turn) + '\n'
        
        # 当前查询
        query = item.get('query', item.get('current_question', ''))
        if query:
            history_text += f"User: {query}\nAssistant:"
        
        # Tokenize
        history_tokens = self.tokenizer(
            history_text,
            max_length=self.max_history_length,
            truncation=True,
            return_tensors='pt'
        )
        
        response_tokens = self.tokenizer(
            response,
            max_length=self.max_response_length,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'context_ids': history_tokens['input_ids'].squeeze(0),
            'context_mask': history_tokens['attention_mask'].squeeze(0),
            'response_ids': response_tokens['input_ids'].squeeze(0),
            'response_mask': response_tokens['attention_mask'].squeeze(0)
        }


class SummarizationDataset(Dataset):
    """
    长文档摘要数据集
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_doc_length: int = 4096,
        max_summary_length: int = 512,
        block_size: int = 32,
    ):
        self.tokenizer = tokenizer
        self.max_doc_length = max_doc_length
        self.max_summary_length = max_summary_length
        self.block_size = block_size
        
        self.data = self._load_data(data_path)
    
    def _load_data(self, data_path: str) -> List[Dict]:
        """加载摘要数据"""
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                data.append(item)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 获取文档和摘要
        document = item.get('document', item.get('text', item.get('article', '')))
        summary = item.get('summary', item.get('abstract', ''))
        
        # Tokenize
        doc_tokens = self.tokenizer(
            document,
            max_length=self.max_doc_length,
            truncation=True,
            return_tensors='pt'
        )
        
        summary_tokens = self.tokenizer(
            summary,
            max_length=self.max_summary_length,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'document_ids': doc_tokens['input_ids'].squeeze(0),
            'document_mask': doc_tokens['attention_mask'].squeeze(0),
            'summary_ids': summary_tokens['input_ids'].squeeze(0),
            'summary_mask': summary_tokens['attention_mask'].squeeze(0)
        }


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    创建DataLoader
    
    Args:
        dataset: 数据集
        batch_size: 批大小
        shuffle: 是否打乱
        num_workers: 数据加载线程数
        pin_memory: 是否固定内存
    
    Returns:
        DataLoader实例
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=None  # 使用默认的collate_fn
    )
