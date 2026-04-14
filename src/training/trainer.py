"""
REFRAG分阶段训练器
支持：重建任务预训练、课程学习、持续预训练(CPT)、RL选择性压缩、监督微调(SFT)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from typing import Optional, Dict, Any, List
import os
import json
from tqdm import tqdm
import logging
from transformers import get_linear_schedule_with_warmup

from ..models import RefragModel
from ..data import (
    ReconstructionDataset, CurriculumDataset, CPTDataset,
    RLDataset, RAGDataset, MultiTurnDialogueDataset, SummarizationDataset
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RefragTrainer:
    """
    REFRAG训练器
    
    支持5个训练阶段：
    1. pretrain: 重建任务预训练（Encoder-Projection对齐）
    2. curriculum: 课程学习
    3. cpt: 持续预训练（Encoder-Decoder对齐）
    4. rl: 强化学习（选择性压缩策略）
    5. sft: 监督微调
    """
    
    def __init__(
        self,
        model: RefragModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        stage: str = "pretrain",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.04,
        max_grad_norm: float = 1.0,
        output_dir: str = "./outputs",
        log_interval: int = 100,
        save_interval: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        mixed_precision: bool = True,
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.stage = stage
        self.device = device
        self.mixed_precision = mixed_precision
        
        # 训练超参数
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        
        # 输出和日志
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置训练阶段
        self.model.set_training_stage(stage)
        
        # 初始化优化器
        self.optimizer = self._create_optimizer()
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision and device == "cuda" else None
        
        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # 日志
        self.train_logs = []
        
    def _create_optimizer(self):
        """创建优化器"""
        # 获取可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        return optimizer
    
    def _create_scheduler(self, num_training_steps: int):
        """创建学习率调度器"""
        warmup_steps = int(num_training_steps * self.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        return scheduler
    
    def train_epoch(self, scheduler: Optional[Any] = None) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # 将数据移到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # 根据训练阶段选择前向传播方式
            if self.stage in ["pretrain", "curriculum"]:
                loss = self._pretrain_step(batch)
            elif self.stage == "cpt":
                loss = self._cpt_step(batch)
            elif self.stage == "rl":
                loss = self._rl_step(batch)
            elif self.stage == "sft":
                loss = self._sft_step(batch)
            else:
                raise ValueError(f"Unknown stage: {self.stage}")
            
            # 反向传播
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.max_grad_norm
                )
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            if scheduler:
                scheduler.step()
            
            # 记录
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                'avg_loss': total_loss / num_batches,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # 日志
            if self.global_step % self.log_interval == 0:
                log_entry = {
                    'step': self.global_step,
                    'epoch': self.epoch,
                    'loss': loss.item(),
                    'avg_loss': total_loss / num_batches,
                    'lr': self.optimizer.param_groups[0]['lr']
                }
                self.train_logs.append(log_entry)
            
            # 保存检查点
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}")
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def _pretrain_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """重建任务预训练步骤"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # 使用查询部分作为重建目标
        query_len = input_ids.shape[1] // 4  # 简化处理：使用前1/4作为query
        query_ids = input_ids[:, :query_len]
        query_mask = attention_mask[:, :query_len]
        context_ids = input_ids[:, query_len:]
        context_mask = attention_mask[:, query_len:]
        
        # 前向传播
        outputs = self.model(
            query_tokens=query_ids,
            context_tokens=context_ids,
            query_attention_mask=query_mask,
            context_attention_mask=context_mask,
            labels=labels
        )
        
        return outputs['loss']
    
    def _cpt_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """持续预训练步骤（下一段预测）"""
        context_ids = batch['context_ids']
        context_mask = batch['context_mask']
        input_ids = batch['input_ids']
        labels = batch['labels']
        
        # 构建查询（使用特殊token或简化处理）
        query_ids = input_ids[:, :1]  # 使用第一个token作为query
        query_mask = torch.ones_like(query_ids)
        
        # 前向传播
        outputs = self.model(
            query_tokens=query_ids,
            context_tokens=context_ids,
            query_attention_mask=query_mask,
            context_attention_mask=context_mask,
            labels=labels
        )
        
        return outputs['loss']
    
    def _rl_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """强化学习步骤"""
        context_ids = batch['context_ids']
        context_mask = batch['context_mask']
        query_ids = batch['query_ids']
        query_mask = batch['query_mask']
        
        # 编码上下文块
        block_embeddings = self.model.encode_blocks(context_ids, context_mask)
        
        # 获取旧策略的log概率
        with torch.no_grad():
            old_logits, _ = self.model.rl_policy(block_embeddings)
            old_probs = torch.sigmoid(old_logits)
            old_log_probs = torch.log(old_probs + 1e-10)
        
        # 计算奖励（使用困惑度的负数）
        with torch.no_grad():
            outputs = self.model(
                query_tokens=query_ids,
                context_tokens=context_ids,
                query_attention_mask=query_mask,
                context_attention_mask=context_mask
            )
            rewards = -outputs['loss'].unsqueeze(0)  # 简化为使用loss的负数
        
        # 计算GRPO损失
        loss, metrics = self.model.rl_policy.compute_grpo_loss(
            block_embeddings,
            rewards,
            old_log_probs
        )
        
        return loss
    
    def _sft_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """监督微调步骤"""
        # 根据数据集类型处理
        if 'answer_ids' in batch:
            # RAG数据集
            context_ids = batch['context_ids']
            context_mask = batch['context_mask']
            query_ids = batch['query_ids']
            query_mask = batch['query_mask']
            answer_ids = batch['answer_ids']
            
            # 构建labels
            batch_size = query_ids.shape[0]
            query_len = query_ids.shape[1]
            answer_len = answer_ids.shape[1]
            
            labels = torch.full(
                (batch_size, query_len + answer_len),
                -100,
                dtype=torch.long,
                device=self.device
            )
            labels[:, query_len:] = answer_ids
            
            # 合并query和answer
            combined_ids = torch.cat([query_ids, answer_ids], dim=1)
            combined_mask = torch.cat([
                query_mask,
                torch.ones_like(answer_ids)
            ], dim=1)
            
            outputs = self.model(
                query_tokens=combined_ids,
                context_tokens=context_ids,
                query_attention_mask=combined_mask,
                context_attention_mask=context_mask,
                labels=labels
            )
        else:
            # 其他数据集类型
            outputs = self.model(**batch)
        
        return outputs['loss']
    
    def evaluate(self) -> Dict[str, float]:
        """评估模型"""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                if self.stage in ["pretrain", "curriculum"]:
                    loss = self._pretrain_step(batch)
                elif self.stage == "cpt":
                    loss = self._cpt_step(batch)
                elif self.stage == "rl":
                    loss = self._rl_step(batch)
                elif self.stage == "sft":
                    loss = self._sft_step(batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            resume_from: 从检查点恢复训练
        """
        # 恢复训练
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # 计算总训练步数
        total_steps = len(self.train_dataloader) * num_epochs
        scheduler = self._create_scheduler(total_steps)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Stage: {self.stage}")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch(scheduler)
            
            # 评估
            val_metrics = self.evaluate()
            
            # 合并指标
            metrics = {**train_metrics, **val_metrics}
            
            # 日志
            logger.info(f"Epoch {epoch}: {metrics}")
            
            # 保存最佳模型
            if 'val_loss' in metrics and metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = metrics['val_loss']
                self.save_checkpoint("best_model")
                logger.info(f"New best model saved with val_loss: {self.best_val_loss:.4f}")
            
            # 保存epoch检查点
            self.save_checkpoint(f"epoch_{epoch}")
        
        # 保存最终模型
        self.save_checkpoint("final_model")
        logger.info("Training completed!")
    
    def save_checkpoint(self, name: str):
        """保存检查点"""
        checkpoint_dir = os.path.join(self.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(checkpoint_dir)
        
        # 保存训练状态
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'optimizer_state': self.optimizer.state_dict(),
            'stage': self.stage
        }
        torch.save(state, os.path.join(checkpoint_dir, 'trainer_state.pt'))
        
        # 保存训练日志
        with open(os.path.join(checkpoint_dir, 'train_logs.json'), 'w') as f:
            json.dump(self.train_logs, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_dir}")
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        # 加载模型
        self.model.load_pretrained(path)
        
        # 加载训练状态
        state_path = os.path.join(path, 'trainer_state.pt')
        if os.path.exists(state_path):
            state = torch.load(state_path)
            self.global_step = state['global_step']
            self.epoch = state['epoch']
            self.best_val_loss = state['best_val_loss']
            self.optimizer.load_state_dict(state['optimizer_state'])
        
        # 加载训练日志
        logs_path = os.path.join(path, 'train_logs.json')
        if os.path.exists(logs_path):
            with open(logs_path, 'r') as f:
                self.train_logs = json.load(f)
        
        logger.info(f"Checkpoint loaded: {path}")


class CurriculumScheduler:
    """
    课程学习调度器
    动态调整课程阶段
    """
    
    def __init__(
        self,
        dataset: CurriculumDataset,
        stages: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
        stage_epochs: int = 2,
        improvement_threshold: float = 0.01
    ):
        self.dataset = dataset
        self.stages = stages
        self.stage_epochs = stage_epochs
        self.improvement_threshold = improvement_threshold
        self.current_stage_idx = 0
        self.best_loss = float('inf')
        self.epochs_at_stage = 0
    
    def step(self, current_loss: float) -> bool:
        """
        更新课程阶段
        
        Returns:
            advanced: 是否进入下一阶段
        """
        self.epochs_at_stage += 1
        
        # 检查是否应该进入下一阶段
        improved = self.best_loss - current_loss > self.improvement_threshold
        
        if improved:
            self.best_loss = current_loss
        
        # 如果达到每个阶段的最小epoch数，且性能提升不足，则进入下一阶段
        if self.epochs_at_stage >= self.stage_epochs:
            if not improved or self.epochs_at_stage >= self.stage_epochs * 2:
                if self.current_stage_idx < len(self.stages) - 1:
                    self.current_stage_idx += 1
                    self.dataset.set_stage(self.current_stage_idx)
                    self.epochs_at_stage = 0
                    self.best_loss = float('inf')
                    logger.info(f"Advanced to curriculum stage {self.current_stage_idx}: "
                              f"{self.stages[self.current_stage_idx]} blocks")
                    return True
        
        return False
    
    def get_current_stage(self) -> int:
        """获取当前课程阶段"""
        return self.current_stage_idx
