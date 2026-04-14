#!/usr/bin/env python3
"""
使用预训练模型进行微调的脚本
"""

import argparse
import yaml
import torch
from transformers import AutoTokenizer
import os

import sys
sys.path.append('/mnt/PRO6000_disk/swd/servo_0/refrag/src')

from models import RefragModel
from data import RAGDataset, get_dataloader
from training import RefragTrainer


def main():
    parser = argparse.ArgumentParser(description="Finetune Pre-trained REFRAG Model")
    parser.add_argument("--config", type=str, default="configs/pretrained_config.yaml")
    parser.add_argument("--encoder_path", type=str, default=None, 
                       help="Path to pre-trained encoder model")
    parser.add_argument("--decoder_path", type=str, default=None, 
                       help="Path to pre-trained decoder model")
    parser.add_argument("--data_path", type=str, required=True, 
                       help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="outputs/finetune")
    parser.add_argument("--stage", type=str, default="sft", 
                       choices=["rl", "sft"],
                       help="Training stage: rl or sft")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 加载tokenizer
    decoder_model = args.decoder_path or config['model']['decoder_name']
    tokenizer = AutoTokenizer.from_pretrained(decoder_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建模型
    encoder_model = args.encoder_path or config['model']['encoder_name']
    
    model = RefragModel(
        encoder_name=encoder_model,
        decoder_name=decoder_model,
        block_size=config['model']['block_size'],
        use_rl_policy=config['model']['use_rl_policy'],
        encoder_config=config['model']['encoder_config'],
        projection_config=config['model']['projection_config'],
        decoder_config=config['model']['decoder_config'],
        rl_config=config['model']['rl_config']
    )
    
    # 冻结编码器（使用预训练模型时）
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # 冻结解码器（如果只训练RL策略）
    if args.stage == "rl":
        for param in model.decoder.parameters():
            param.requires_grad = False
    
    # 创建数据集
    train_dataset = RAGDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_context_length=config['data']['rag']['max_context_length'],
        max_query_length=config['data']['rag']['max_query_length'],
        max_answer_length=config['data']['rag']['max_answer_length'],
        block_size=config['model']['block_size']
    )
    
    train_dataloader = get_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    # 创建训练器
    if args.stage == "rl":
        trainer = RefragTrainer(
            model=model,
            train_dataloader=train_dataloader,
            stage="rl",
            learning_rate=config['training']['rl']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            warmup_ratio=config['training']['warmup_ratio'],
            max_grad_norm=config['training']['max_grad_norm'],
            output_dir=os.path.join(args.output_dir, "rl"),
            log_interval=config['output']['log_interval'],
            save_interval=config['output']['save_interval'],
            device=device,
            mixed_precision=config['training']['mixed_precision']
        )
        num_epochs = config['training']['rl']['num_epochs']
    else:  # sft
        trainer = RefragTrainer(
            model=model,
            train_dataloader=train_dataloader,
            stage="sft",
            learning_rate=config['training']['sft']['learning_rate'],
            weight_decay=config['training']['sft']['weight_decay'],
            warmup_ratio=config['training']['sft']['warmup_ratio'],
            max_grad_norm=config['training']['max_grad_norm'],
            output_dir=os.path.join(args.output_dir, "sft"),
            log_interval=config['output']['log_interval'],
            save_interval=config['output']['save_interval'],
            device=device,
            mixed_precision=config['training']['mixed_precision']
        )
        num_epochs = config['training']['sft']['num_epochs']
    
    # 训练
    print(f"\nStarting {args.stage.upper()} finetuning...")
    trainer.train(num_epochs=num_epochs)
    
    # 保存模型
    output_path = os.path.join(args.output_dir, f"{args.stage}_model")
    os.makedirs(output_path, exist_ok=True)
    
    # 保存模型
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"\nFinetuning completed!")
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    main()
