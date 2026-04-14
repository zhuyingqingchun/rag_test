#!/usr/bin/env python3
"""
REFRAG完整训练流程脚本
依次执行：预训练 -> 课程学习 -> CPT -> RL -> SFT
"""

import argparse
import yaml
import torch
from transformers import AutoTokenizer
import os

import sys
sys.path.append('/mnt/PRO6000_disk/swd/servo_0/refrag/src')

from models import RefragModel
from data import (
    ReconstructionDataset, CurriculumDataset, CPTDataset,
    RLDataset, RAGDataset, get_dataloader
)
from training import RefragTrainer, CurriculumScheduler


def train_stage(trainer, num_epochs, stage_name):
    """训练单个阶段"""
    print(f"\n{'='*50}")
    print(f"Starting {stage_name}")
    print(f"{'='*50}\n")
    trainer.train(num_epochs=num_epochs)
    print(f"\n{stage_name} completed!")


def main():
    parser = argparse.ArgumentParser(description="REFRAG Full Training Pipeline")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing all data files")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--stages", type=str, nargs="+", 
                       default=["pretrain", "curriculum", "cpt", "rl", "sft"],
                       help="Stages to run")
    parser.add_argument("--skip_stages", type=str, nargs="+", default=[],
                       help="Stages to skip")
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
    
    # 阶段1: 预训练
    if "pretrain" in args.stages and "pretrain" not in args.skip_stages:
        train_data_path = os.path.join(args.data_dir, "pretrain.jsonl")
        
        train_dataset = ReconstructionDataset(
            data_path=train_data_path,
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
        
        trainer = RefragTrainer(
            model=model,
            train_dataloader=train_dataloader,
            stage="pretrain",
            learning_rate=config['training']['pretrain']['learning_rate'],
            weight_decay=config['training']['pretrain']['weight_decay'],
            warmup_ratio=config['training']['pretrain']['warmup_ratio'],
            max_grad_norm=config['training']['max_grad_norm'],
            output_dir=os.path.join(args.output_dir, "pretrain"),
            log_interval=config['output']['log_interval'],
            save_interval=config['output']['save_interval'],
            device=device,
            mixed_precision=config['training']['mixed_precision']
        )
        
        train_stage(trainer, config['training']['pretrain']['num_epochs'], "Pretraining")
        
        # 加载最佳模型
        model.load_pretrained(os.path.join(args.output_dir, "pretrain", "best_model"))
    
    # 阶段2: 课程学习
    if "curriculum" in args.stages and "curriculum" not in args.skip_stages:
        train_data_path = os.path.join(args.data_dir, "curriculum.jsonl")
        
        train_dataset = CurriculumDataset(
            data_path=train_data_path,
            tokenizer=tokenizer,
            max_length=config['training']['pretrain']['max_length'],
            block_size=config['model']['block_size'],
            curriculum_stages=config['training']['curriculum']['stages']
        )
        
        train_dataloader = get_dataloader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers']
        )
        
        trainer = RefragTrainer(
            model=model,
            train_dataloader=train_dataloader,
            stage="pretrain",  # 课程学习使用pretrain阶段设置
            learning_rate=config['training']['curriculum']['learning_rate'],
            weight_decay=config['training']['curriculum']['weight_decay'],
            warmup_ratio=config['training']['curriculum']['warmup_ratio'],
            max_grad_norm=config['training']['max_grad_norm'],
            output_dir=os.path.join(args.output_dir, "curriculum"),
            log_interval=config['output']['log_interval'],
            save_interval=config['output']['save_interval'],
            device=device,
            mixed_precision=config['training']['mixed_precision']
        )
        
        # 课程学习调度器
        scheduler = CurriculumScheduler(
            dataset=train_dataset,
            stages=config['training']['curriculum']['stages'],
            stage_epochs=config['training']['curriculum']['stage_epochs'],
            improvement_threshold=config['training']['curriculum']['improvement_threshold']
        )
        
        # 训练直到完成所有课程阶段
        current_stage = 0
        epochs_at_stage = 0
        best_loss = float('inf')
        
        while current_stage < len(config['training']['curriculum']['stages']):
            print(f"\nCurriculum Stage {current_stage}: {config['training']['curriculum']['stages'][current_stage]} blocks")
            
            metrics = trainer.train_epoch()
            current_loss = metrics['train_loss']
            
            # 检查是否应该进入下一阶段
            improved = best_loss - current_loss > config['training']['curriculum']['improvement_threshold']
            if improved:
                best_loss = current_loss
            
            epochs_at_stage += 1
            
            if epochs_at_stage >= config['training']['curriculum']['stage_epochs']:
                if not improved or epochs_at_stage >= config['training']['curriculum']['stage_epochs'] * 2:
                    current_stage += 1
                    if current_stage < len(config['training']['curriculum']['stages']):
                        train_dataset.set_stage(current_stage)
                        epochs_at_stage = 0
                        best_loss = float('inf')
                        print(f"Advanced to stage {current_stage}")
        
        print("\nCurriculum Learning completed!")
        
        # 加载最佳模型
        model.load_pretrained(os.path.join(args.output_dir, "curriculum", "best_model"))
    
    # 阶段3: CPT
    if "cpt" in args.stages and "cpt" not in args.skip_stages:
        train_data_path = os.path.join(args.data_dir, "cpt.jsonl")
        
        train_dataset = CPTDataset(
            data_path=train_data_path,
            tokenizer=tokenizer,
            context_length=config['training']['cpt']['context_length'],
            prediction_length=config['training']['cpt']['prediction_length'],
            block_size=config['model']['block_size']
        )
        
        train_dataloader = get_dataloader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers']
        )
        
        trainer = RefragTrainer(
            model=model,
            train_dataloader=train_dataloader,
            stage="cpt",
            learning_rate=config['training']['cpt']['learning_rate'],
            weight_decay=config['training']['cpt']['weight_decay'],
            warmup_ratio=config['training']['cpt']['warmup_ratio'],
            max_grad_norm=config['training']['max_grad_norm'],
            output_dir=os.path.join(args.output_dir, "cpt"),
            log_interval=config['output']['log_interval'],
            save_interval=config['output']['save_interval'],
            device=device,
            mixed_precision=config['training']['mixed_precision']
        )
        
        train_stage(trainer, config['training']['cpt']['num_epochs'], "CPT")
        
        # 加载最佳模型
        model.load_pretrained(os.path.join(args.output_dir, "cpt", "best_model"))
    
    # 阶段4: RL
    if "rl" in args.stages and "rl" not in args.skip_stages:
        train_data_path = os.path.join(args.data_dir, "rl.jsonl")
        
        train_dataset = RLDataset(
            data_path=train_data_path,
            tokenizer=tokenizer,
            max_length=config['training']['pretrain']['max_length'],
            block_size=config['model']['block_size'],
            num_samples_per_example=config['training']['rl']['num_samples_per_example']
        )
        
        train_dataloader = get_dataloader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers']
        )
        
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
        
        train_stage(trainer, config['training']['rl']['num_epochs'], "RL Training")
        
        # 加载最佳模型
        model.load_pretrained(os.path.join(args.output_dir, "rl", "best_model"))
    
    # 阶段5: SFT
    if "sft" in args.stages and "sft" not in args.skip_stages:
        train_data_path = os.path.join(args.data_dir, "sft.jsonl")
        
        train_dataset = RAGDataset(
            data_path=train_data_path,
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
        
        train_stage(trainer, config['training']['sft']['num_epochs'], "SFT")
        
        # 加载最佳模型
        model.load_pretrained(os.path.join(args.output_dir, "sft", "best_model"))
    
    print("\n" + "="*50)
    print("All training stages completed!")
    print(f"Final model saved to: {args.output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
