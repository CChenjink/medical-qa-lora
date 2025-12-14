"""
增强版评估模块 - 支持多种评估指标
"""

import torch
import jieba
from rouge_chinese import Rouge
from typing import List, Dict
from tqdm import tqdm


class EnhancedMedicalQAEvaluator:
    """增强版医疗问答评估器"""
    
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
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取回答部分
        if "回答：" in response:
            response = response.split("回答：")[-1].strip()
        
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
    
    def calculate_bleu(
        self,
        predictions: List[str],
        references: List[str]
    ) -> float:
        """计算 BLEU 分数"""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            
            smooth = SmoothingFunction()
            bleu_scores = []
            
            for pred, ref in zip(predictions, references):
                pred_tokens = list(jieba.cut(pred))
                ref_tokens = [list(jieba.cut(ref))]
                
                score = sentence_bleu(
                    ref_tokens, 
                    pred_tokens,
                    smoothing_function=smooth.method1
                )
                bleu_scores.append(score)
            
            return sum(bleu_scores) / len(bleu_scores)
        except ImportError:
            print("⚠️  NLTK 未安装，跳过 BLEU 计算")
            print("   安装命令: pip install nltk")
            return None
        except Exception as e:
            print(f"⚠️  BLEU 计算失败: {e}")
            return None
    
    def calculate_bertscore(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict:
        """计算 BERTScore"""
        try:
            from bert_score import score
            
            print("  计算 BERTScore (可能需要几分钟)...")
            
            # 计算 BERTScore
            # model_type: 使用中文 BERT 模型
            P, R, F1 = score(
                predictions, 
                references,
                lang='zh',  # 中文
                verbose=False,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            return {
                'precision': P.mean().item(),
                'recall': R.mean().item(),
                'f1': F1.mean().item()
            }
        except ImportError:
            print("⚠️  bert-score 未安装，跳过 BERTScore 计算")
            print("   安装命令: pip install bert-score")
            return None
        except Exception as e:
            print(f"⚠️  BERTScore 计算失败: {e}")
            return None
    
    def calculate_length_stats(
        self,
        predictions: List[str],
        references: List[str]
    ) -> Dict:
        """计算长度统计"""
        pred_lengths = [len(p) for p in predictions]
        ref_lengths = [len(r) for r in references]
        
        return {
            'avg_pred_length': sum(pred_lengths) / len(pred_lengths),
            'avg_ref_length': sum(ref_lengths) / len(ref_lengths),
            'length_ratio': sum(pred_lengths) / sum(ref_lengths)
        }
    
    def evaluate(self, test_data: List[Dict], verbose: bool = True) -> Dict:
        """评估模型"""
        
        predictions = []
        references = []
        
        if verbose:
            print("生成回答...")
        
        iterator = tqdm(test_data) if verbose else test_data
        
        for item in iterator:
            prompt = f"{item['instruction']}\n问题：{item['input']}\n回答："
            response = self.generate_response(prompt)
            
            predictions.append(response)
            references.append(item['output'])
        
        # 计算指标
        if verbose:
            print("计算评估指标...")
        
        rouge_scores = self.calculate_rouge(predictions, references)
        bleu_score = self.calculate_bleu(predictions, references)
        bert_score = self.calculate_bertscore(predictions, references)
        length_stats = self.calculate_length_stats(predictions, references)
        
        results = {
            'rouge_scores': rouge_scores,
            'bleu_score': bleu_score,
            'bert_score': bert_score,
            'length_stats': length_stats,
            'num_samples': len(test_data),
            'predictions': predictions,
            'references': references
        }
        
        return results
    
    def print_results(self, results: Dict):
        """打印评估结果"""
        
        print("\n" + "=" * 60)
        print("评估结果")
        print("=" * 60)
        print(f"样本数量: {results['num_samples']}")
        
        print("\n【文本重叠度 (ROUGE)】")
        print(f"  ROUGE-1: {results['rouge_scores']['rouge-1']['f']:.4f}")
        print(f"  ROUGE-2: {results['rouge_scores']['rouge-2']['f']:.4f}")
        print(f"  ROUGE-L: {results['rouge_scores']['rouge-l']['f']:.4f}")
        
        if results['bleu_score'] is not None:
            print("\n【N-gram 精确度 (BLEU)】")
            print(f"  BLEU: {results['bleu_score']:.4f}")
        
        if results['bert_score'] is not None:
            print("\n【语义相似度 (BERTScore)】")
            print(f"  Precision: {results['bert_score']['precision']:.4f}")
            print(f"  Recall:    {results['bert_score']['recall']:.4f}")
            print(f"  F1:        {results['bert_score']['f1']:.4f}")
        
        print("\n【长度统计】")
        print(f"  平均预测长度: {results['length_stats']['avg_pred_length']:.1f} 字")
        print(f"  平均参考长度: {results['length_stats']['avg_ref_length']:.1f} 字")
        print(f"  长度比率: {results['length_stats']['length_ratio']:.2f}")
        
        print("=" * 60)
    
    def print_samples(self, results: Dict, num_samples: int = 3):
        """打印示例"""
        
        print("\n" + "=" * 60)
        print(f"示例展示（前 {num_samples} 个）")
        print("=" * 60)
        
        for i in range(min(num_samples, len(results['predictions']))):
            print(f"\n【示例 {i+1}】")
            print(f"参考答案: {results['references'][i][:100]}...")
            print(f"模型回答: {results['predictions'][i][:100]}...")
            print("-" * 60)
