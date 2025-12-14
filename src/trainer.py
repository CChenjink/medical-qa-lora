"""
训练器模块
"""

from transformers import Trainer, TrainingArguments
from typing import Dict


def create_training_arguments(config: Dict) -> TrainingArguments:
    """创建训练参数"""
    
    training_config = config['training_args']
    
    # 支持 warmup_steps 或 warmup_ratio
    warmup_args = {}
    if 'warmup_steps' in training_config:
        warmup_args['warmup_steps'] = training_config['warmup_steps']
    elif 'warmup_ratio' in training_config:
        warmup_args['warmup_ratio'] = training_config['warmup_ratio']
    
    return TrainingArguments(
        output_dir=training_config['output_dir'],
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 
                                                        training_config['per_device_train_batch_size']),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
        learning_rate=training_config['learning_rate'],
        logging_steps=training_config.get('logging_steps', 10),
        save_steps=training_config.get('save_steps', 500),
        eval_steps=training_config.get('eval_steps', 500),
        save_total_limit=training_config.get('save_total_limit', 3),
        fp16=training_config.get('fp16', True),
        evaluation_strategy=training_config.get('evaluation_strategy', 'steps'),
        load_best_model_at_end=training_config.get('load_best_model_at_end', True),
        metric_for_best_model=training_config.get('metric_for_best_model', 'loss'),
        greater_is_better=training_config.get('greater_is_better', False),
        report_to='tensorboard',
        logging_dir=f"{training_config['output_dir']}/logs",
        **warmup_args
    )


def create_trainer(
    model,
    training_args: TrainingArguments,
    train_dataset,
    eval_dataset,
    tokenizer
) -> Trainer:
    """创建训练器"""
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )
    
    return trainer
