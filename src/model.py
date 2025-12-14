"""
模型加载和配置模块
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from typing import Dict, Optional, Tuple


def load_base_model(
    model_name_or_path: str,
    quantization_config: Optional[Dict] = None
):
    """加载基础模型"""
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    
    # 配置量化（如果使用 QLoRA）
    bnb_config = None
    if quantization_config:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quantization_config.get('load_in_4bit', False),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=quantization_config.get('bnb_4bit_use_double_quant', True),
            bnb_4bit_quant_type=quantization_config.get('bnb_4bit_quant_type', 'nf4')
        )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map='auto',
        torch_dtype=torch.float16
    )
    
    return model, tokenizer


def setup_lora(model, lora_config: Dict):
    """配置 LoRA"""
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        target_modules=lora_config['target_modules'],
        bias=lora_config.get('bias', 'none')  # 默认值 'none'
    )
    
    model = get_peft_model(model, peft_config)
    
    # 打印可训练参数
    model.print_trainable_parameters()
    
    return model


def load_trained_model(
    model_path: str,
    base_model_path: Optional[str] = None
) -> Tuple:
    """加载训练好的模型"""
    
    if base_model_path:
        # 加载 LoRA 模型
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # 加载完整模型
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=torch.float16
        )
    
    model.eval()
    return model, tokenizer
