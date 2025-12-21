"""
汇总所有实验结果
生成对比表格和图表
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
    
    # 实验配置 - 数据规模: 2k, 5k, 10k, 20k, 40k
    experiments = [
        ('Baseline', 'outputs/baseline/eval_results.json', 0, '-'),
        ('LoRA-2k', 'outputs/lora_2k/eval_results.json', 2000, 'LoRA'),
        ('LoRA-5k', 'outputs/lora_5k/eval_results.json', 5000, 'LoRA'),
        ('LoRA-10k', 'outputs/lora_10k/eval_results.json', 10000, 'LoRA'),
        ('LoRA-20k', 'outputs/lora_20k/eval_results.json', 20000, 'LoRA'),
        ('LoRA-40k', 'outputs/lora_40k/eval_results.json', 40000, 'LoRA'),
        ('QLoRA-2k', 'outputs/qlora_2k/eval_results.json', 2000, 'QLoRA'),
        ('QLoRA-5k', 'outputs/qlora_5k/eval_results.json', 5000, 'QLoRA'),
        ('QLoRA-10k', 'outputs/qlora_10k/eval_results.json', 10000, 'QLoRA'),
        ('QLoRA-20k', 'outputs/qlora_20k/eval_results.json', 20000, 'QLoRA'),
        ('QLoRA-40k', 'outputs/qlora_40k/eval_results.json', 40000, 'QLoRA'),
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
            
            # 打印加载状态 - 显示所有关键指标
            print(f"✓ {name:15s} R1: {rouge['rouge-1']['f']:.4f}  "
                  f"R2: {rouge['rouge-2']['f']:.4f}  "
                  f"RL: {rouge['rouge-l']['f']:.4f}", end="")
            
            if bleu is not None:
                print(f"  BLEU: {bleu:.4f}", end="")
            
            if bert:
                print(f"  BERT: {bert['f1']:.4f}", end="")
            
            print()  # 换行
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
    
    # 详细分析所有指标
    print("\n" + "=" * 60)
    print("【详细指标分析】")
    print("=" * 60)
    
    # 1. 相对 Baseline 的改进
    if 'Baseline' in df['实验'].values:
        baseline_row = df[df['实验'] == 'Baseline'].iloc[0]
        print(f"\n1. 相对 Baseline 的改进幅度")
        print("-" * 60)
        print(f"{'实验':<15} {'ROUGE-1':>10} {'ROUGE-2':>10} {'ROUGE-L':>10} {'BLEU':>10} {'BERTScore':>10}")
        print("-" * 60)
        
        for _, row in df[df['实验'] != 'Baseline'].iterrows():
            r1_imp = (row['ROUGE-1'] - baseline_row['ROUGE-1']) / baseline_row['ROUGE-1'] * 100
            r2_imp = (row['ROUGE-2'] - baseline_row['ROUGE-2']) / baseline_row['ROUGE-2'] * 100
            rl_imp = (row['ROUGE-L'] - baseline_row['ROUGE-L']) / baseline_row['ROUGE-L'] * 100
            
            bleu_str = ""
            if 'BLEU' in row and pd.notna(row['BLEU']) and 'BLEU' in baseline_row and pd.notna(baseline_row['BLEU']):
                bleu_imp = (row['BLEU'] - baseline_row['BLEU']) / baseline_row['BLEU'] * 100
                bleu_str = f"{bleu_imp:+.1f}%"
            else:
                bleu_str = "N/A"
            
            bert_str = ""
            if 'BERTScore-F1' in row and pd.notna(row['BERTScore-F1']) and 'BERTScore-F1' in baseline_row and pd.notna(baseline_row['BERTScore-F1']):
                bert_imp = (row['BERTScore-F1'] - baseline_row['BERTScore-F1']) / baseline_row['BERTScore-F1'] * 100
                bert_str = f"{bert_imp:+.1f}%"
            else:
                bert_str = "N/A"
            
            print(f"{row['实验']:<15} {r1_imp:>9.1f}% {r2_imp:>9.1f}% {rl_imp:>9.1f}% {bleu_str:>10} {bert_str:>10}")
    
    # 2. LoRA vs QLoRA 对比
    lora_df = df[df['方法'] == 'LoRA'].sort_values('数据量')
    qlora_df = df[df['方法'] == 'QLoRA'].sort_values('数据量')
    
    if len(lora_df) > 0 and len(qlora_df) > 0:
        print(f"\n2. LoRA vs QLoRA 性能差距")
        print("-" * 60)
        print(f"{'数据量':<10} {'ROUGE-1':>10} {'ROUGE-2':>10} {'ROUGE-L':>10} {'BLEU':>10} {'BERTScore':>10}")
        print("-" * 60)
        
        for size in sorted(set(lora_df['数据量']) & set(qlora_df['数据量'])):
            lora_row = lora_df[lora_df['数据量'] == size].iloc[0]
            qlora_row = qlora_df[qlora_df['数据量'] == size].iloc[0]
            
            r1_diff = (lora_row['ROUGE-1'] - qlora_row['ROUGE-1']) / qlora_row['ROUGE-1'] * 100
            r2_diff = (lora_row['ROUGE-2'] - qlora_row['ROUGE-2']) / qlora_row['ROUGE-2'] * 100
            rl_diff = (lora_row['ROUGE-L'] - qlora_row['ROUGE-L']) / qlora_row['ROUGE-L'] * 100
            
            bleu_str = ""
            if 'BLEU' in lora_row and pd.notna(lora_row['BLEU']) and 'BLEU' in qlora_row and pd.notna(qlora_row['BLEU']):
                bleu_diff = (lora_row['BLEU'] - qlora_row['BLEU']) / qlora_row['BLEU'] * 100
                bleu_str = f"{bleu_diff:+.1f}%"
            else:
                bleu_str = "N/A"
            
            bert_str = ""
            if 'BERTScore-F1' in lora_row and pd.notna(lora_row['BERTScore-F1']) and 'BERTScore-F1' in qlora_row and pd.notna(qlora_row['BERTScore-F1']):
                bert_diff = (lora_row['BERTScore-F1'] - qlora_row['BERTScore-F1']) / qlora_row['BERTScore-F1'] * 100
                bert_str = f"{bert_diff:+.1f}%"
            else:
                bert_str = "N/A"
            
            print(f"{int(size/1000):>3}k      {r1_diff:>9.1f}% {r2_diff:>9.1f}% {rl_diff:>9.1f}% {bleu_str:>10} {bert_str:>10}")
        
        print("\n注: 正值表示 LoRA 优于 QLoRA，负值表示 QLoRA 优于 LoRA")
    
    # 3. 数据规模边际效应分析
    if len(lora_df) > 1:
        print(f"\n3. 数据规模边际效应 (LoRA)")
        print("-" * 60)
        print(f"{'数据增量':<15} {'ROUGE-1':>10} {'ROUGE-2':>10} {'ROUGE-L':>10} {'BLEU':>10} {'BERTScore':>10}")
        print("-" * 60)
        
        for i in range(1, len(lora_df)):
            prev_row = lora_df.iloc[i-1]
            curr_row = lora_df.iloc[i]
            
            size_label = f"{int(prev_row['数据量']/1000)}k→{int(curr_row['数据量']/1000)}k"
            
            r1_gain = (curr_row['ROUGE-1'] - prev_row['ROUGE-1']) / prev_row['ROUGE-1'] * 100
            r2_gain = (curr_row['ROUGE-2'] - prev_row['ROUGE-2']) / prev_row['ROUGE-2'] * 100
            rl_gain = (curr_row['ROUGE-L'] - prev_row['ROUGE-L']) / prev_row['ROUGE-L'] * 100
            
            bleu_str = ""
            if 'BLEU' in curr_row and pd.notna(curr_row['BLEU']) and 'BLEU' in prev_row and pd.notna(prev_row['BLEU']):
                bleu_gain = (curr_row['BLEU'] - prev_row['BLEU']) / prev_row['BLEU'] * 100
                bleu_str = f"{bleu_gain:+.1f}%"
            else:
                bleu_str = "N/A"
            
            bert_str = ""
            if 'BERTScore-F1' in curr_row and pd.notna(curr_row['BERTScore-F1']) and 'BERTScore-F1' in prev_row and pd.notna(prev_row['BERTScore-F1']):
                bert_gain = (curr_row['BERTScore-F1'] - prev_row['BERTScore-F1']) / prev_row['BERTScore-F1'] * 100
                bert_str = f"{bert_gain:+.1f}%"
            else:
                bert_str = "N/A"
            
            print(f"{size_label:<15} {r1_gain:>9.1f}% {r2_gain:>9.1f}% {rl_gain:>9.1f}% {bleu_str:>10} {bert_str:>10}")
    
    if len(qlora_df) > 1:
        print(f"\n   数据规模边际效应 (QLoRA)")
        print("-" * 60)
        print(f"{'数据增量':<15} {'ROUGE-1':>10} {'ROUGE-2':>10} {'ROUGE-L':>10} {'BLEU':>10} {'BERTScore':>10}")
        print("-" * 60)
        
        for i in range(1, len(qlora_df)):
            prev_row = qlora_df.iloc[i-1]
            curr_row = qlora_df.iloc[i]
            
            size_label = f"{int(prev_row['数据量']/1000)}k→{int(curr_row['数据量']/1000)}k"
            
            r1_gain = (curr_row['ROUGE-1'] - prev_row['ROUGE-1']) / prev_row['ROUGE-1'] * 100
            r2_gain = (curr_row['ROUGE-2'] - prev_row['ROUGE-2']) / prev_row['ROUGE-2'] * 100
            rl_gain = (curr_row['ROUGE-L'] - prev_row['ROUGE-L']) / prev_row['ROUGE-L'] * 100
            
            bleu_str = ""
            if 'BLEU' in curr_row and pd.notna(curr_row['BLEU']) and 'BLEU' in prev_row and pd.notna(prev_row['BLEU']):
                bleu_gain = (curr_row['BLEU'] - prev_row['BLEU']) / prev_row['BLEU'] * 100
                bleu_str = f"{bleu_gain:+.1f}%"
            else:
                bleu_str = "N/A"
            
            bert_str = ""
            if 'BERTScore-F1' in curr_row and pd.notna(curr_row['BERTScore-F1']) and 'BERTScore-F1' in prev_row and pd.notna(prev_row['BERTScore-F1']):
                bert_gain = (curr_row['BERTScore-F1'] - prev_row['BERTScore-F1']) / prev_row['BERTScore-F1'] * 100
                bert_str = f"{bert_gain:+.1f}%"
            else:
                bert_str = "N/A"
            
            print(f"{size_label:<15} {r1_gain:>9.1f}% {r2_gain:>9.1f}% {rl_gain:>9.1f}% {bleu_str:>10} {bert_str:>10}")
    
    # 4. 最佳配置推荐
    print(f"\n4. 最佳配置推荐")
    print("-" * 60)
    
    # 找出各指标的最高分
    best_configs = {}
    for metric in ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BLEU', 'BERTScore-F1']:
        if metric in df.columns:
            valid_df = df[df[metric].notna()]
            if len(valid_df) > 0:
                best_row = valid_df.loc[valid_df[metric].idxmax()]
                best_configs[metric] = {
                    'config': best_row['实验'],
                    'score': best_row[metric],
                    'method': best_row['方法'],
                    'data_size': best_row['数据量']
                }
    
    for metric, info in best_configs.items():
        print(f"{metric:<15}: {info['config']:<15} (分数: {info['score']:.4f})")
    
    
    # 绘制图表
    try:
        plot_comparison_charts(df, output_dir)
    except Exception as e:
        print(f"\n⚠️  绘图失败: {e}")
    
    print("\n" + "=" * 60)
    print("✓ 汇总完成！")
    print("=" * 60)


def plot_comparison_charts(df, output_dir):
    """绘制对比图表 - 全面分析所有指标"""
    
    # 过滤出有效数据
    lora_data = df[df['方法'] == 'LoRA'].sort_values('数据量')
    qlora_data = df[df['方法'] == 'QLoRA'].sort_values('数据量')
    baseline_data = df[df['方法'] == '-']
    
    if len(lora_data) == 0 and len(qlora_data) == 0:
        print("\n⚠️  没有足够的数据用于绘图")
        return
    
    print("\n" + "=" * 60)
    print("生成对比图表...")
    print("=" * 60)
    
    # 图1: 所有 ROUGE 指标对比（3个子图）
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        if len(lora_data) > 0:
            ax.plot(lora_data['数据量']/1000, lora_data[metric], 
                   marker='o', label='LoRA', linewidth=2.5, markersize=8, color='#2E86AB')
        
        if len(qlora_data) > 0:
            ax.plot(qlora_data['数据量']/1000, qlora_data[metric], 
                   marker='s', label='QLoRA', linewidth=2.5, markersize=8, color='#A23B72')
        
        # Baseline
        if len(baseline_data) > 0:
            baseline_score = baseline_data[metric].values[0]
            ax.axhline(y=baseline_score, color='#F18F01', linestyle='--', 
                      label=f'Baseline ({baseline_score:.4f})', linewidth=2)
        
        ax.set_xlabel('训练数据量 (k)', fontsize=12, fontweight='bold')
        # 只在最左边的子图显示 Y 轴标签
        if idx == 0:
            ax.set_ylabel('ROUGE F1 Score', fontsize=12, fontweight='bold')
        ax.set_title(metric, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    chart_file = output_dir / '1_rouge_all_metrics.png'
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"✓ [1/5] ROUGE 全指标对比图: {chart_file.name}")
    plt.close()
    
    # 图2: BLEU 对比
    if 'BLEU' in df.columns:
        lora_bleu = lora_data[lora_data['BLEU'].notna()]
        qlora_bleu = qlora_data[qlora_data['BLEU'].notna()]
        
        if len(lora_bleu) > 0 or len(qlora_bleu) > 0:
            plt.figure(figsize=(10, 6))
            
            if len(lora_bleu) > 0:
                plt.plot(lora_bleu['数据量']/1000, lora_bleu['BLEU'], 
                        marker='o', label='LoRA', linewidth=2.5, markersize=8, color='#2E86AB')
            
            if len(qlora_bleu) > 0:
                plt.plot(qlora_bleu['数据量']/1000, qlora_bleu['BLEU'], 
                        marker='s', label='QLoRA', linewidth=2.5, markersize=8, color='#A23B72')
            
            # Baseline
            if len(baseline_data) > 0 and 'BLEU' in baseline_data.columns:
                if baseline_data['BLEU'].notna().any():
                    baseline_score = baseline_data['BLEU'].values[0]
                    plt.axhline(y=baseline_score, color='#F18F01', linestyle='--', 
                               label=f'Baseline ({baseline_score:.4f})', linewidth=2)
            
            plt.xlabel('训练数据量 (k)', fontsize=12, fontweight='bold')
            plt.ylabel('BLEU Score', fontsize=12, fontweight='bold')
            plt.title('数据规模对 BLEU 的影响', fontsize=14, fontweight='bold')
            plt.legend(fontsize=11, loc='best')
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.ylim(bottom=0)
            
            chart_file = output_dir / '2_bleu_comparison.png'
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            print(f"✓ [2/5] BLEU 对比图: {chart_file.name}")
            plt.close()
    
    # 图3: BERTScore 对比
    if 'BERTScore-F1' in df.columns:
        lora_bert = lora_data[lora_data['BERTScore-F1'].notna()]
        qlora_bert = qlora_data[qlora_data['BERTScore-F1'].notna()]
        
        if len(lora_bert) > 0 or len(qlora_bert) > 0:
            plt.figure(figsize=(10, 6))
            
            if len(lora_bert) > 0:
                plt.plot(lora_bert['数据量']/1000, lora_bert['BERTScore-F1'], 
                        marker='o', label='LoRA', linewidth=2.5, markersize=8, color='#2E86AB')
            
            if len(qlora_bert) > 0:
                plt.plot(qlora_bert['数据量']/1000, qlora_bert['BERTScore-F1'], 
                        marker='s', label='QLoRA', linewidth=2.5, markersize=8, color='#A23B72')
            
            # Baseline
            if len(baseline_data) > 0 and 'BERTScore-F1' in baseline_data.columns:
                if baseline_data['BERTScore-F1'].notna().any():
                    baseline_score = baseline_data['BERTScore-F1'].values[0]
                    plt.axhline(y=baseline_score, color='#F18F01', linestyle='--', 
                               label=f'Baseline ({baseline_score:.4f})', linewidth=2)
            
            plt.xlabel('训练数据量 (k)', fontsize=12, fontweight='bold')
            plt.ylabel('BERTScore F1', fontsize=12, fontweight='bold')
            plt.title('数据规模对 BERTScore 的影响', fontsize=14, fontweight='bold')
            plt.legend(fontsize=11, loc='best')
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.ylim(bottom=0)
            
            chart_file = output_dir / '3_bertscore_comparison.png'
            plt.savefig(chart_file, dpi=300, bbox_inches='tight')
            print(f"✓ [3/5] BERTScore 对比图: {chart_file.name}")
            plt.close()
    
    # 图4: 所有指标综合对比（LoRA）
    if len(lora_data) > 0:
        plt.figure(figsize=(12, 7))
        
        plt.plot(lora_data['数据量']/1000, lora_data['ROUGE-1'], 
                marker='o', label='ROUGE-1', linewidth=2, markersize=7)
        plt.plot(lora_data['数据量']/1000, lora_data['ROUGE-2'], 
                marker='s', label='ROUGE-2', linewidth=2, markersize=7)
        plt.plot(lora_data['数据量']/1000, lora_data['ROUGE-L'], 
                marker='^', label='ROUGE-L', linewidth=2, markersize=7)
        
        if 'BLEU' in lora_data.columns and lora_data['BLEU'].notna().any():
            plt.plot(lora_data['数据量']/1000, lora_data['BLEU'], 
                    marker='d', label='BLEU', linewidth=2, markersize=7)
        
        if 'BERTScore-F1' in lora_data.columns and lora_data['BERTScore-F1'].notna().any():
            plt.plot(lora_data['数据量']/1000, lora_data['BERTScore-F1'], 
                    marker='*', label='BERTScore', linewidth=2, markersize=10)
        
        plt.xlabel('训练数据量 (k)', fontsize=12, fontweight='bold')
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.title('LoRA: 所有评估指标综合对比', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best', ncol=2)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.ylim(bottom=0)
        
        chart_file = output_dir / '4_lora_all_metrics.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"✓ [4/5] LoRA 全指标综合图: {chart_file.name}")
        plt.close()
    
    # 图5: 所有指标综合对比（QLoRA）
    if len(qlora_data) > 0:
        plt.figure(figsize=(12, 7))
        
        plt.plot(qlora_data['数据量']/1000, qlora_data['ROUGE-1'], 
                marker='o', label='ROUGE-1', linewidth=2, markersize=7)
        plt.plot(qlora_data['数据量']/1000, qlora_data['ROUGE-2'], 
                marker='s', label='ROUGE-2', linewidth=2, markersize=7)
        plt.plot(qlora_data['数据量']/1000, qlora_data['ROUGE-L'], 
                marker='^', label='ROUGE-L', linewidth=2, markersize=7)
        
        if 'BLEU' in qlora_data.columns and qlora_data['BLEU'].notna().any():
            plt.plot(qlora_data['数据量']/1000, qlora_data['BLEU'], 
                    marker='d', label='BLEU', linewidth=2, markersize=7)
        
        if 'BERTScore-F1' in qlora_data.columns and qlora_data['BERTScore-F1'].notna().any():
            plt.plot(qlora_data['数据量']/1000, qlora_data['BERTScore-F1'], 
                    marker='*', label='BERTScore', linewidth=2, markersize=10)
        
        plt.xlabel('训练数据量 (k)', fontsize=12, fontweight='bold')
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.title('QLoRA: 所有评估指标综合对比', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best', ncol=2)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.ylim(bottom=0)
        
        chart_file = output_dir / '5_qlora_all_metrics.png'
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        print(f"✓ [5/5] QLoRA 全指标综合图: {chart_file.name}")
        plt.close()
    
    print("=" * 60)

if __name__ == "__main__":
    main()
