"""
增强版模型评估脚本 - 支持 ROUGE + BLEU + BERTScore
"""

import json
import argparse
from pathlib import Path

# 从 src 模块导入功能
from src.model import load_trained_model
from src.evaluator_enhanced import EnhancedMedicalQAEvaluator


def main():
    parser = argparse.ArgumentParser(description="增强版模型评估")
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='模型路径，如 outputs/lora_medical/checkpoint-best'
    )
    parser.add_argument(
        '--base_model_path',
        type=str,
        default=None,
        help='基础模型路径（如果是 LoRA 模型需要提供）'
    )
    parser.add_argument(
        '--test_file',
        type=str,
        default='./data/processed/test.json',
        help='测试数据文件'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='结果保存路径'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='最大评估样本数（用于快速测试）'
    )
    parser.add_argument(
        '--show_samples',
        type=int,
        default=3,
        help='显示示例数量'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='批量生成大小（越大越快，但需要更多显存）'
    )
    parser.add_argument(
        '--max_new_tokens',
        type=int,
        default=128,
        help='最大生成长度（减少可加快速度）'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("增强版模型评估 (ROUGE + BLEU + BERTScore)")
    print("=" * 60)
    
    # 1. 加载模型
    print(f"\n1. 加载模型: {args.model_path}")
    if args.base_model_path:
        print(f"   基础模型: {args.base_model_path}")
    
    model, tokenizer = load_trained_model(args.model_path, args.base_model_path)
    
    # 2. 加载测试数据
    print(f"\n2. 加载测试数据: {args.test_file}")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    if args.max_samples and len(test_data) > args.max_samples:
        test_data = test_data[:args.max_samples]
        print(f"   限制样本数: {args.max_samples}")
    
    print(f"   测试样本数: {len(test_data)}")
    
    # 3. 创建评估器
    print("\n3. 创建增强版评估器...")
    print(f"   批量大小: {args.batch_size}")
    print(f"   最大生成长度: {args.max_new_tokens} tokens")
    evaluator = EnhancedMedicalQAEvaluator(model, tokenizer, batch_size=args.batch_size)
    
    # 4. 开始评估
    print("\n4. 开始评估...")
    print(f"   预计时间: ~{len(test_data) * 3 / args.batch_size / 60:.1f} 分钟")
    print("-" * 60)
    results = evaluator.evaluate(test_data, verbose=True, use_batch=True)
    
    # 5. 打印结果
    print("\n5. 评估结果:")
    evaluator.print_results(results)
    
    # 6. 显示示例
    if args.show_samples > 0:
        evaluator.print_samples(results, num_samples=args.show_samples)
    
    # 7. 保存结果
    if args.output_file:
        print(f"\n7. 保存结果到: {args.output_file}")
        
        # 创建输出目录
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        save_results = {
            'rouge_scores': results['rouge_scores'],
            'bleu_score': results['bleu_score'],
            'bert_score': results['bert_score'],
            'length_stats': results['length_stats'],
            'num_samples': results['num_samples'],
            'samples': [
                {
                    'input': test_data[i]['input'],
                    'reference': results['references'][i],
                    'prediction': results['predictions'][i]
                }
                for i in range(min(10, len(test_data)))
            ]
        }
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(save_results, f, ensure_ascii=False, indent=2)
        
        print(f"   ✓ 结果已保存")
    
    print("\n" + "=" * 60)
    print("✓ 评估完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
