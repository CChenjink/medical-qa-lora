"""
汇总所有实验结果
生成对比表格和图表
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def load_results(result_file):
    """加载评估结果"""
    if not os.path.exists(result_file):
        return None
    
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['rouge_scores']


def main():
    print("=" * 50)
    print("实验结果汇总")
    print("=" * 50)
    
    # 实验配置
    experiments = [
        ('Baseline', 'outputs/baseline/eval_results.json', '-'),
        ('LoRA-1k', 'outputs/lora_1k/eval_results.json', '1k'),
        ('LoRA-5k', 'outputs/lora_5k/eval_results.json', '5k'),
        ('LoRA-10k', 'outputs/lora_10k/eval_results.json', '10k'),
        ('QLoRA-1k', 'outputs/qlora_1k/eval_results.json', '1k'),
        ('QLoRA-5k', 'outputs/qlora_5k/eval_results.json', '5k'),
        ('QLoRA-10k', 'outputs/qlora_10k/eval_results.json', '10k'),
    ]
    
    # 收集结果
    results = []
    for name, file_path, data_size in experiments:
        scores = load_results(file_path)
        if scores:
            results.append({
                '实验': name,
                '数据量': data_size,
                'ROUGE-1': scores['rouge-1']['f'],
                'ROUGE-2': scores['rouge-2']['f'],
                'ROUGE-L': scores['rouge-l']['f']
            })
            print(f"✓ {name:15s} ROUGE-L: {scores['rouge-l']['f']:.4f}")
        else:
            print(f"✗ {name:15s} 结果文件不存在")
    
    # 创建 DataFrame
    df = pd.DataFrame(results)
    
    # 保存为 CSV
    output_dir = Path('outputs/summary')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = output_dir / 'results_summary.csv'
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 结果已保存到: {csv_file}")
    
    # 打印表格
    print("\n" + "=" * 50)
    print("实验结果对比表")
    print("=" * 50)
    print(df.to_string(index=False))
    
    # 绘制图表（如果有 matplotlib）
    try:
        # 数据规模影响图
        lora_data = df[df['实验'].str.contains('LoRA') & ~df['实验'].str.contains('Q')]
        qlora_data = df[df['实验'].str.contains('QLoRA')]
        
        if len(lora_data) > 0 and len(qlora_data) > 0:
            plt.figure(figsize=(10, 6))
            
            data_sizes = ['1k', '5k', '10k']
            lora_scores = [lora_data[lora_data['数据量'] == size]['ROUGE-L'].values[0] 
                          for size in data_sizes if size in lora_data['数据量'].values]
            qlora_scores = [qlora_data[qlora_data['数据量'] == size]['ROUGE-L'].values[0] 
                           for size in data_sizes if size in qlora_data['数据量'].values]
            
            x = range(len(data_sizes))
            plt.plot(x, lora_scores, marker='o', label='LoRA', linewidth=2)
            plt.plot(x, qlora_scores, marker='s', label='QLoRA', linewidth=2)
            
            plt.xlabel('数据规模', fontsize=12)
            plt.ylabel('ROUGE-L', fontsize=12)
            plt.title('数据规模对模型性能的影响', fontsize=14)
            plt.xticks(x, data_sizes)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            chart_file = output_dir / 'data_scale_comparison.png'
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            print(f"✓ 图表已保存到: {chart_file}")
            
    except Exception as e:
        print(f"⚠ 绘图失败: {e}")
    
    print("\n" + "=" * 50)
    print("✓ 汇总完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
