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
    
    def __init__(self, model, tokenizer, batch_size=16):
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
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        min_new_tokens: int = 10
    ) -> str:
        """生成单个回答"""
        
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取回答部分
        if "回答：" in response:
            response = response.split("回答：")[-1].strip()
        
        # 确保不为空
        if not response:
            response = "无法生成回答"
        
        return response
    
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        min_new_tokens: int = 10  # 强制至少生成10个token
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
                min_new_tokens=min_new_tokens,  # 强制最小生成长度
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # 只解码新生成的部分（不包括输入）
        input_lengths = inputs['input_ids'].shape[1]
        
        # 批量解码
        responses = []
        for i, output in enumerate(outputs):
            # 只取生成的新token
            generated_tokens = output[input_lengths:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # 清理回答
            response = response.strip()
            
            # 如果回答为空，使用完整解码
            if not response:
                response = self.tokenizer.decode(output, skip_special_tokens=True)
                if "回答：" in response:
                    response = response.split("回答：")[-1].strip()
            
            # 确保不为空
            if not response:
                response = "无法生成回答"
            
            responses.append(response)
        
        return responses
    
    def calculate_rouge(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict:
        """计算 ROUGE 分数"""
        
        # 过滤空预测（用参考答案替代，避免计算错误）
        filtered_predictions = []
        filtered_references = []
        
        for pred, ref in zip(predictions, references):
            # 如果预测为空或只有空白字符，使用一个占位符
            if not pred or not pred.strip():
                pred = "无法生成回答"
            filtered_predictions.append(pred)
            filtered_references.append(ref)
        
        # 分词
        predictions_seg = [' '.join(jieba.cut(p)) for p in filtered_predictions]
        references_seg = [' '.join(jieba.cut(r)) for r in filtered_references]
        
        # 计算 ROUGE
        scores = self.rouge.get_scores(predictions_seg, references_seg, avg=True)
        
        return scores
    
    def evaluate(self, test_data: List[Dict], use_batch=True) -> Dict:
        """评估模型"""
        
        predictions = []
        references = []
        empty_count = 0
        
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
                
                # 统计空回答
                for resp in responses:
                    if not resp or resp == "无法生成回答":
                        empty_count += 1
                
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
        
        if empty_count > 0:
            print(f"⚠️  警告: {empty_count}/{len(test_data)} 个样本生成为空")
        
        rouge_scores = self.calculate_rouge(predictions, references)
        
        results = {
            'rouge_scores': rouge_scores,
            'num_samples': len(test_data),
            'empty_count': empty_count,
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
