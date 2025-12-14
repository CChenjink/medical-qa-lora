"""
准备不同规模的数据集
从完整训练集中采样 20k, 50k, 100k 数据
用于研究训练数据规模对模型性能的影响
"""

import json
import random
import argparse
from pathlib import Path


def load_data(file_path):
    """加载数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_data(data, file_path):
    """保存数据"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def create_data_splits(train_file, output_dir, seed=42):
    """创建不同规模的数据集"""
    
    # 设置随机种子
    random.seed(seed)
    
    # 加载完整训练集
    print(f"加载训练数据: {train_file}")
    train_data = load_data(train_file)
    print(f"完整训练集大小: {len(train_data)}")
    
    # 打乱数据
    random.shuffle(train_data)
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建不同规模的数据集
    splits = {
        '20k': 20000,
        '50k': 50000,
        '100k': 100000
    }
    
    for name, size in splits.items():
        if len(train_data) >= size:
            split_data = train_data[:size]
            output_file = Path(output_dir) / f"train_{name}.json"
            save_data(split_data, output_file)
            print(f"✓ 创建 {name} 数据集: {len(split_data)} 样本 -> {output_file}")
        else:
            print(f"⚠ 警告: 数据不足，无法创建 {name} 数据集（需要 {size}，实际 {len(train_data)}）")


def main():
    parser = argparse.ArgumentParser(description="准备不同规模的数据集")
    parser.add_argument(
        '--train_file',
        type=str,
        default='./data/processed/train.json',
        help='完整训练集路径'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/processed',
        help='输出目录'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("准备不同规模的数据集")
    print("=" * 50)
    
    create_data_splits(args.train_file, args.output_dir, args.seed)
    
    print("\n" + "=" * 50)
    print("✓ 数据准备完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()
