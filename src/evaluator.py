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
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.rouge = Rouge()
    
    def generate_response(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.8
    ) -> str:
        """生成回答"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                top_p=top_p,
                temperature=temperature
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
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
    
    def evaluate(self, test_data: List[Dict]) -> Dict:
        """评估模型"""
        
        predictions = []
        references = []
        
        print("生成回答...")
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
