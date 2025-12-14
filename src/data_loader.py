"""
数据加载模块
"""

import json
from typing import Dict, List
from datasets import Dataset
from transformers import PreTrainedTokenizer


class MedicalQADataset:
    """医疗问答数据集类"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_source_length: int = 512,
        max_target_length: int = 512
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        # 加载数据
        self.data = self.load_data()
    
    def load_data(self) -> List[Dict]:
        """加载 JSON 数据"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def format_prompt(self, instruction: str, input_text: str) -> str:
        """格式化提示"""
        return f"{instruction}\n问题：{input_text}\n回答："
    
    def preprocess_function(self, examples: Dict) -> Dict:
        """预处理函数"""
        inputs = []
        targets = []
        
        for instruction, input_text, output_text in zip(
            examples['instruction'],
            examples['input'],
            examples['output']
        ):
            prompt = self.format_prompt(instruction, input_text)
            inputs.append(prompt)
            targets.append(output_text)
        
        # 编码输入
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.max_source_length,
            truncation=True,
            padding='max_length',
            return_tensors=None
        )
        
        # 编码标签
        labels = self.tokenizer(
            targets,
            max_length=self.max_target_length,
            truncation=True,
            padding='max_length',
            return_tensors=None
        )
        
        model_inputs['labels'] = labels['input_ids']
        
        return model_inputs
    
    def get_dataset(self) -> Dataset:
        """获取 Hugging Face Dataset 对象"""
        dataset = Dataset.from_list(self.data)
        tokenized_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        return tokenized_dataset


def load_datasets(
    train_file: str,
    val_file: str,
    test_file: str,
    tokenizer: PreTrainedTokenizer,
    max_source_length: int = 512,
    max_target_length: int = 512
):
    """加载训练、验证、测试数据集"""
    
    train_dataset = MedicalQADataset(
        train_file, tokenizer, max_source_length, max_target_length
    ).get_dataset()
    
    val_dataset = MedicalQADataset(
        val_file, tokenizer, max_source_length, max_target_length
    ).get_dataset()
    
    test_dataset = MedicalQADataset(
        test_file, tokenizer, max_source_length, max_target_length
    ).get_dataset()
    
    return train_dataset, val_dataset, test_dataset
