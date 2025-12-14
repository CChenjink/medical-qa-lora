"""
训练器模块
"""

from transformers import Trainer, TrainingArguments
from typing import Dict


def create_training_arguments(config: Dict) -> TrainingArguments:
    """创建训练参数"""
    
    training_config = config['training_args']
    
    return TrainingArguments(
        output_dir=training_config['output_dir'],
        num_train_epochs=training_config['num_train_epochs'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        warmup_steps=training_config['warmup_steps'],
        logging_steps=training_config['logging_steps'],
        save_steps=training_config['save_steps'],
        eval_steps=training_config['eval_steps'],
        save_total_limit=training_config['save_total_limit'],
        fp16=training_config['fp16'],
        evaluation_strategy=training_config['evaluation_strategy'],
        load_best_model_at_end=training_config['load_best_model_at_end'],
        metric_for_best_model=training_config.get('metric_for_best_model', 'loss'),
        greater_is_better=training_config.get('greater_is_better', False),
        report_to='tensorboard',
        logging_dir=f"{training_config['output_dir']}/logs"
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
