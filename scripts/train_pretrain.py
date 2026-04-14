#!/usr/bin/env python3
"""
REFRAG重建任务预训练脚本
"""

import argparse
import yaml
import torch
from transformers import AutoTokenizer

import sys
sys.path.append('/mnt/PRO6000_disk/swd/servo_0/refrag/src')

from models import RefragModel
from data import ReconstructionDataset, get_dataloader
from training import RefragTrainer


def main():
    parser = argparse.ArgumentParser(description="REFRAG Pretraining")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/pretrain")
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['decoder_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建数据集
    train_dataset = ReconstructionDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=config['training']['pretrain']['max_length'],
        block_size=config['model']['block_size']
    )
    
    train_dataloader = get_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    # 创建模型
    model = RefragModel(
        encoder_name=config['model']['encoder_name'],
        decoder_name=config['model']['decoder_name'],
        block_size=config['model']['block_size'],
        use_rl_policy=config['model']['use_rl_policy'],
        encoder_config=config['model']['encoder_config'],
        projection_config=config['model']['projection_config'],
        decoder_config=config['model']['decoder_config'],
        rl_config=config['model']['rl_config']
    )
    
    # 创建训练器
    trainer = RefragTrainer(
        model=model,
        train_dataloader=train_dataloader,
        stage="pretrain",
        learning_rate=config['training']['pretrain']['learning_rate'],
        weight_decay=config['training']['pretrain']['weight_decay'],
        warmup_ratio=config['training']['pretrain']['warmup_ratio'],
        max_grad_norm=config['training']['max_grad_norm'],
        output_dir=args.output_dir,
        log_interval=config['output']['log_interval'],
        save_interval=config['output']['save_interval'],
        device=device,
        mixed_precision=config['training']['mixed_precision']
    )
    
    # 训练
    trainer.train(
        num_epochs=config['training']['pretrain']['num_epochs'],
        resume_from=args.resume_from
    )
    
    print("Pretraining completed!")


if __name__ == "__main__":
    main()
