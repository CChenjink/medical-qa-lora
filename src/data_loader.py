"""
数据加载模块
"""

import json
from typing import Dict, List
from datasets import Dataset
from transformers import PreTrainedTokenizer
import torch

IGNORE_TOKEN_ID = -100

class MedicalQADataset:
    """医疗问答数据集类"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 加载数据
        self.data = self.load_data()
        # self.data = self.data[:2000]  # 仅用于调试，取前2000条数据

        self.system1 = "问题："
        self.system2 = "回答："
    
    def load_data(self) -> List[Dict]:
        """加载 JSON 数据"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def format_prompt(self, instruction: str, input_text: str) -> str:
        """格式化提示"""
        return f"{instruction}\n{self.system1}{input_text}\n{self.system2}"
    
    def preprocess_function(self, examples: Dict) -> Dict:
        """预处理函数"""
        input_ids = []
        targets = []
        
        for instruction, input_text, output_text in zip(
            examples['instruction'],
            examples['input'],
            examples['output']
        ):
            prompt = self.format_prompt(instruction, input_text)
            prompt_id = self.tokenizer(prompt).input_ids
            response_id = self.tokenizer(output_text + "\n<|im_end|>").input_ids

            input_id = prompt_id + response_id
            target_id = [IGNORE_TOKEN_ID] * len(prompt_id) + response_id

            # pad
            assert len(input_id) == len(target_id)
            input_id += [self.tokenizer.pad_token_id] * (self.max_length - len(input_id))
            target_id += [IGNORE_TOKEN_ID] * (self.max_length - len(target_id))
            input_ids.append(input_id[:self.max_length])
            targets.append(target_id[:self.max_length])

        
        input_ids = torch.tensor(input_ids)
        targets = torch.tensor(targets)

        return dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    
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
    max_length: int = 512,
):
    """加载训练、验证、测试数据集"""
    
    train_dataset = MedicalQADataset(
        train_file, tokenizer, max_length
    ).get_dataset()
    
    val_dataset = MedicalQADataset(
        val_file, tokenizer, max_length
    ).get_dataset()
    
    test_dataset = MedicalQADataset(
        test_file, tokenizer, max_length
    ).get_dataset()
    
    return train_dataset, val_dataset, test_dataset
