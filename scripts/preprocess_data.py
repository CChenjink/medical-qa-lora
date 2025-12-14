"""
数据预处理脚本
将原始数据转换为训练格式，并划分训练集、验证集、测试集
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict


def load_raw_data(data_path: str) -> List[Dict]:
    """加载原始数据"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def format_data(raw_data: List[Dict]) -> List[Dict]:
    """
    格式化数据为指令微调格式
    输入格式可能各异，需要统一转换为:
    {
        "instruction": "任务描述",
        "input": "问题",
        "output": "答案"
    }
    """
    formatted_data = []
    
    for item in raw_data:
        # 根据实际数据格式调整
        if "instruction" in item and "input" in item and "output" in item:
            # 已经是标准格式
            formatted_data.append(item)
        elif "question" in item and "answer" in item:
            # question-answer 格式
            formatted_data.append({
                "instruction": "回答医疗健康问题",
                "input": item["question"],
                "output": item["answer"]
            })
        elif "query" in item and "response" in item:
            # query-response 格式
            formatted_data.append({
                "instruction": "回答医疗健康问题",
                "input": item["query"],
                "output": item["response"]
            })
    
    return formatted_data


def clean_data(data: List[Dict], min_length: int = 10, max_length: int = 512) -> List[Dict]:
    """清洗数据"""
    cleaned_data = []
    
    for item in data:
        # 检查必要字段
        if not all(key in item for key in ["instruction", "input", "output"]):
            continue
        
        # 检查长度
        if len(item["input"]) < min_length or len(item["output"]) < min_length:
            continue
        
        if len(item["input"]) > max_length or len(item["output"]) > max_length:
            continue
        
        # 去除空白
        item["input"] = item["input"].strip()
        item["output"] = item["output"].strip()
        
        if item["input"] and item["output"]:
            cleaned_data.append(item)
    
    return cleaned_data


def split_data(data: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1):
    """划分数据集"""
    random.shuffle(data)
    
    total = len(data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    return train_data, val_data, test_data


def save_data(data: List[Dict], save_path: str):
    """保存数据"""
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="数据预处理")
    parser.add_argument(
        "--raw_data",
        type=str,
        default="./data/raw/medical_qa.json",
        help="原始数据路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="输出目录"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="最大样本数（用于快速测试）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("数据预处理")
    print("=" * 50)
    
    # 加载原始数据
    print(f"\n1. 加载原始数据: {args.raw_data}")
    raw_data = load_raw_data(args.raw_data)
    print(f"   原始样本数: {len(raw_data)}")
    
    # 格式化数据
    print("\n2. 格式化数据...")
    formatted_data = format_data(raw_data)
    print(f"   格式化后样本数: {len(formatted_data)}")
    
    # 清洗数据
    print("\n3. 清洗数据...")
    cleaned_data = clean_data(formatted_data)
    print(f"   清洗后样本数: {len(cleaned_data)}")
    
    # 限制样本数（如果指定）
    if args.max_samples and len(cleaned_data) > args.max_samples:
        cleaned_data = random.sample(cleaned_data, args.max_samples)
        print(f"   限制样本数: {len(cleaned_data)}")
    
    # 划分数据集
    print("\n4. 划分数据集...")
    train_data, val_data, test_data = split_data(cleaned_data)
    print(f"   训练集: {len(train_data)}")
    print(f"   验证集: {len(val_data)}")
    print(f"   测试集: {len(test_data)}")
    
    # 保存数据
    print("\n5. 保存数据...")
    save_data(train_data, os.path.join(args.output_dir, "train.json"))
    save_data(val_data, os.path.join(args.output_dir, "dev.json"))
    save_data(test_data, os.path.join(args.output_dir, "test.json"))
    
    print(f"\n✓ 数据预处理完成！")
    print(f"  输出目录: {args.output_dir}")
    
    # 显示示例
    print("\n数据示例:")
    print(json.dumps(train_data[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
