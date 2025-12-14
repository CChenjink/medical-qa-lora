"""
评估模块
"""

import torch
import jieba
from rouge_chinese import Rouge
from typing import List, Dict
from tqdm import tqdm


class MedicalQAEvaluator:
    """医疗问答评估器"""
    
    def __init__(self, model, tokenizer, batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.rouge = Rouge()
        self.batch_size = batch_size
        
        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 对于解码器模型，批量生成时使用左填充
        self.tokenizer.padding_side = 'left'
    
    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """生成单个回答"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,  # 只生成新的 token，不包括输入
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取回答部分
        if "回答：" in response:
            response = response.split("回答：")[-1].strip()
        
        return response
    
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> List[str]:
        """批量生成回答（更快）"""
        
        # 批量编码
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # 批量解码
        responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # 提取回答部分
        cleaned_responses = []
        for response in responses:
            if "回答：" in response:
                response = response.split("回答：")[-1].strip()
            cleaned_responses.append(response)
        
        return cleaned_responses
    
    def calculate_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict:
        """计算 ROUGE 分数"""
        
        # 分词
        predictions_seg = [' '.join(jieba.cut(p)) for p in predictions]
        references_seg = [' '.join(jieba.cut(r)) for r in references]
        
        # 计算 ROUGE
        scores = self.rouge.get_scores(predictions_seg, references_seg, avg=True)
        
        return scores
    
    def evaluate(self, test_data: List[Dict], use_batch=True) -> Dict:
        """评估模型"""
        
        predictions = []
        references = []
        
        print("生成回答...")
        
        if use_batch:
            # 批量生成（更快）
            for i in tqdm(range(0, len(test_data), self.batch_size)):
                batch = test_data[i:i + self.batch_size]
                
                # 准备批量提示
                prompts = [
                    f"{item['instruction']}\n问题：{item['input']}\n回答："
                    for item in batch
                ]
                
                # 批量生成
                responses = self.generate_batch(prompts)
                
                predictions.extend(responses)
                references.extend([item['output'] for item in batch])
        else:
            # 单个生成（慢但更稳定）
            for item in tqdm(test_data):
                prompt = f"{item['instruction']}\n问题：{item['input']}\n回答："
                response = self.generate_response(prompt)
                
                predictions.append(response)
                references.append(item['output'])
        
        # 计算指标
        print("计算评估指标...")
        rouge_scores = self.calculate_rouge(predictions, references)
        
        results = {
            'rouge_scores': rouge_scores,
            'num_samples': len(test_data),
            'predictions': predictions,
            'references': references
        }
        
        return results
    
    def print_results(self, results: Dict):
        """打印评估结果"""
        
        print("\n" + "=" * 50)
        print("评估结果")
        print("=" * 50)
        print(f"样本数量: {results['num_samples']}")
        print(f"\nROUGE-1: {results['rouge_scores']['rouge-1']['f']:.4f}")
        print(f"ROUGE-2: {results['rouge_scores']['rouge-2']['f']:.4f}")
        print(f"ROUGE-L: {results['rouge_scores']['rouge-l']['f']:.4f}")
        print("=" * 50)
