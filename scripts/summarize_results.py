"""
汇总所有实验结果
生成对比表格和图表
支持 ROUGE 和 BERTScore 指标
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_results(result_file):
    """加载评估结果"""
    if not os.path.exists(result_file):
        return None
    
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {
        'rouge_scores': data.get('rouge_scores'),
        'bleu_score': data.get('bleu_score'),
        'bert_score': data.get('bert_score'),
        'num_samples': data.get('num_samples', 0)
    }
    
    return results


def main():
    print("=" * 60)
    print("实验结果汇总 (ROUGE + BLEU + BERTScore)")
    print("=" * 60)
    
    # 实验配置 - 更新为新的数据规模
    experiments = [
        ('Baseline', 'outputs/baseline/eval_results.json', 0, '-'),
        ('LoRA-10k', 'outputs/lora_10k/eval_results.json', 10000, 'LoRA'),
        ('LoRA-20k', 'outputs/lora_20k/eval_results.json', 20000, 'LoRA'),
        ('LoRA-40k', 'outputs/lora_40k/eval_results.json', 40000, 'LoRA'),
        ('LoRA-60k', 'outputs/lora_60k/eval_results.json', 60000, 'LoRA'),
        ('QLoRA-10k', 'outputs/qlora_10k/eval_results.json', 10000, 'QLoRA'),
        ('QLoRA-20k', 'outputs/qlora_20k/eval_results.json', 20000, 'QLoRA'),
        ('QLoRA-40k', 'outputs/qlora_40k/eval_results.json', 40000, 'QLoRA'),
        ('QLoRA-60k', 'outputs/qlora_60k/eval_results.json', 60000, 'QLoRA'),
    ]
    
    # 收集结果
    results = []
    for name, file_path, data_size, method in experiments:
        data = load_results(file_path)
        if data and data['rouge_scores']:
            rouge = data['rouge_scores']
            bleu = data['bleu_score']
            bert = data['bert_score']
            
            result = {
                '实验': name,
                '方法': method,
                '数据量': data_size,
                'ROUGE-1': rouge['rouge-1']['f'],
                'ROUGE-2': rouge['rouge-2']['f'],
                'ROUGE-L': rouge['rouge-l']['f'],
            }
            
            # 添加 BLEU（如果有）
            if bleu is not None:
                result['BLEU'] = bleu
            
            # 添加 BERTScore（如果有）
            if bert:
                result['BERTScore-F1'] = bert['f1']
            
            results.append(result)
            
            # 打印加载状态
            bleu_info = f"BLEU: {bleu:.4f}" if bleu is not None else ""
            bert_info = f"BERTScore: {bert['f1']:.4f}" if bert else ""
            extra_info = "  ".join(filter(None, [bleu_info, bert_info]))
            print(f"✓ {name:15s} ROUGE-L: {rouge['rouge-l']['f']:.4f}  {extra_info}")
        else:
            print(f"✗ {name:15s} 结果文件不存在")
    
    if not results:
        print("\n⚠️  没有找到任何评估结果")
        print("请先运行评估脚本生成结果文件")
        return
    
    # 创建 DataFrame
    df = pd.DataFrame(results)
    
    # 保存为 CSV
    output_dir = Path('outputs/summary')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = output_dir / 'results_summary.csv'
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 结果已保存到: {csv_file}")
    
    # 打印表格
    print("\n" + "=" * 60)
    print("实验结果对比表")
    print("=" * 60)
    print(df.to_string(index=False))
    
    # 计算改进幅度（相对于 baseline）
    if 'Baseline' in df['实验'].values:
        baseline_rouge_l = df[df['实验'] == 'Baseline']['ROUGE-L'].values[0]
        print(f"\n【相对 Baseline 的改进】")
        print(f"Baseline ROUGE-L: {baseline_rouge_l:.4f}")
        print("-" * 60)
        
        for _, row in df[df['实验'] != 'Baseline'].iterrows():
            improvement = (row['ROUGE-L'] - baseline_rouge_l) / baseline_rouge_l * 100
            print(f"{row['实验']:15s} ROUGE-L: {row['ROUGE-L']:.4f}  "
                  f"提升: {improvement:+.1f}%")
    
    
    # 绘制图表
    try:
        plot_comparison_charts(df, output_dir)
    except Exception as e:
        print(f"\n⚠️  绘图失败: {e}")
    
    print("\n" + "=" * 60)
    print("✓ 汇总完成！")
    print("=" * 60)


def plot_comparison_charts(df, output_dir):
    """绘制对比图表"""
    
    # 过滤出有效数据
    lora_data = df[df['方法'] == 'LoRA'].sort_values('数据量')
    qlora_data = df[df['方法'] == 'QLoRA'].sort_values('数据量')
    baseline_data = df[df['方法'] == '-']
    
    if len(lora_data) == 0 and len(qlora_data) == 0:
        print("\n⚠️  没有足够的数据用于绘图")
        return
    
    # 图1: 数据规模对 ROUGE-L 的影响
    plt.figure(figsize=(12, 6))
    
    if len(lora_data) > 0:
        plt.plot(lora_data['数据量']/1000, lora_data['ROUGE-L'], 
                marker='o', label='LoRA', linewidth=2, markersize=8)
    
    if len(qlora_data) > 0:
        plt.plot(qlora_data['数据量']/1000, qlora_data['ROUGE-L'], 
                marker='s', label='QLoRA', linewidth=2, markersize=8)
    
    # 添加 baseline 水平线
    if len(baseline_data) > 0:
        baseline_score = baseline_data['ROUGE-L'].values[0]
        plt.axhline(y=baseline_score, color='r', linestyle='--', 
                   label=f'Baseline ({baseline_score:.4f})', linewidth=2)
    
    plt.xlabel('训练数据量 (k)', fontsize=12)
    plt.ylabel('ROUGE-L F1 Score', fontsize=12)
    plt.title('数据规模对模型性能的影响 (ROUGE-L)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    chart_file = output_dir / 'rouge_l_comparison.png'
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ ROUGE-L 对比图已保存: {chart_file}")
    plt.close()
    
    # 图2: 所有 ROUGE 指标对比（如果有多个实验）
    if len(df) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            if len(lora_data) > 0:
                ax.plot(lora_data['数据量']/1000, lora_data[metric], 
                       marker='o', label='LoRA', linewidth=2, markersize=8)
            
            if len(qlora_data) > 0:
                ax.plot(qlora_data['数据量']/1000, qlora_data[metric], 
                       marker='s', label='QLoRA', linewidth=2, markersize=8)
            
            # Baseline
            if len(baseline_data) > 0:
                baseline_score = baseline_data[metric].values[0]
                ax.axhline(y=baseline_score, color='r', linestyle='--', 
                          label='Baseline', linewidth=2)
            
            ax.set_xlabel('训练数据量 (k)', fontsize=11)
            ax.set_ylabel(f'{metric} F1 Score', fontsize=11)
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_file = output_dir / 'all_rouge_comparison.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"✓ 所有 ROUGE 指标对比图已保存: {chart_file}")
        plt.close()
    
    # 图3: BERTScore 对比（如果有）
    if 'BERTScore-F1' in df.columns:
        lora_bert = lora_data[lora_data['BERTScore-F1'].notna()]
        qlora_bert = qlora_data[qlora_data['BERTScore-F1'].notna()]
        
        if len(lora_bert) > 0 or len(qlora_bert) > 0:
            plt.figure(figsize=(12, 6))
            
            if len(lora_bert) > 0:
                plt.plot(lora_bert['数据量']/1000, lora_bert['BERTScore-F1'], 
                        marker='o', label='LoRA', linewidth=2, markersize=8)
            
            if len(qlora_bert) > 0:
                plt.plot(qlora_bert['数据量']/1000, qlora_bert['BERTScore-F1'], 
                        marker='s', label='QLoRA', linewidth=2, markersize=8)
            
            # Baseline
            if len(baseline_data) > 0 and 'BERTScore-F1' in baseline_data.columns:
                if baseline_data['BERTScore-F1'].notna().any():
                    baseline_score = baseline_data['BERTScore-F1'].values[0]
                    plt.axhline(y=baseline_score, color='r', linestyle='--', 
                               label=f'Baseline ({baseline_score:.4f})', linewidth=2)
            
            plt.xlabel('训练数据量 (k)', fontsize=12)
            plt.ylabel('BERTScore F1', fontsize=12)
            plt.title('数据规模对模型性能的影响 (BERTScore)', fontsize=14, fontweight='bold')
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            
            chart_file = output_dir / 'bertscore_comparison.png'
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            print(f"✓ BERTScore 对比图已保存: {chart_file}")
            plt.close()


if __name__ == "__main__":
    main()
