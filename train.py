"""
模型微调训练脚本
支持 LoRA 和 QLoRA 微调
"""

import yaml
import argparse
from datasets import load_dataset
from typing import Dict

# 从 src 模块导入功能
from src.model import load_base_model, setup_lora
from src.data_loader import MedicalQADataset
from src.trainer import create_training_arguments, create_trainer


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="大模型微调训练")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='配置文件路径，如 configs/lora_config.yaml'
    )
    parser.add_argument(
        '--resume_from_checkpoint',
        type=str,
        default=None,
        help='从检查点恢复训练'
    )
    args = parser.parse_args()
    
    print("=" * 50)
    print("大模型微调训练")
    print("=" * 50)
    
    # 1. 加载配置
    print(f"\n1. 加载配置文件: {args.config}")
    config = load_config(args.config)
    
    # 2. 加载模型和分词器
    print("\n2. 加载基础模型...")
    quantization_config = config.get('quantization_config', None)
    model, tokenizer = load_base_model(
        config['model_name_or_path'],
        quantization_config
    )
    
    # 3. 配置 LoRA
    print("\n3. 配置 LoRA...")
    model = setup_lora(model, config['lora_config'])
    
    # 4. 加载数据
    print("\n4. 加载和预处理数据...")
    data_config = config['data_config']
    
    # 加载训练集
    train_loader = MedicalQADataset(
        data_config['train_file'],
        tokenizer,
        data_config['max_source_length'],
        data_config['max_target_length']
    )
    train_dataset = train_loader.get_dataset()
    
    # 加载验证集
    val_loader = MedicalQADataset(
        data_config['validation_file'],
        tokenizer,
        data_config['max_source_length'],
        data_config['max_target_length']
    )
    val_dataset = val_loader.get_dataset()
    
    print(f"   训练样本数: {len(train_dataset)}")
    print(f"   验证样本数: {len(val_dataset)}")
    
    # 5. 创建训练参数
    print("\n5. 配置训练参数...")
    training_args = create_training_arguments(config)
    
    # 6. 创建训练器
    print("\n6. 创建训练器...")
    trainer = create_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )
    
    # 7. 开始训练
    print("\n7. 开始训练...")
    print("=" * 50)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # 8. 保存模型
    print("\n8. 保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained(config['training_args']['output_dir'])
    
    print("\n" + "=" * 50)
    print("✓ 训练完成！")
    print(f"模型保存在: {config['training_args']['output_dir']}")
    print("=" * 50)


if __name__ == "__main__":
    main()
