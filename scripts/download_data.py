"""
下载医疗问答数据集
"""

import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset


def download_medical_dataset(save_dir):
    """下载医疗问答数据集"""
    
    print("正在下载医疗问答数据集...")
    
    try:
        # 方案1: 使用 trust_remote_code=True 参数
        print("尝试方案1: 使用 trust_remote_code 参数...")
        dataset = load_dataset("shibing624/medical", split="train", trust_remote_code=True)
        
        # 保存原始数据
        raw_dir = os.path.join(save_dir, "raw")
        Path(raw_dir).mkdir(parents=True, exist_ok=True)
        
        dataset.to_json(os.path.join(raw_dir, "medical_qa.json"))
        
        print(f"✓ 数据集已下载到: {raw_dir}")
        print(f"  总样本数: {len(dataset)}")
        
        return dataset
        
    except Exception as e:
        print(f"方案1失败: {e}")
        
        # 方案2: 直接从数据文件加载
        try:
            print("\n尝试方案2: 直接加载数据文件...")
            dataset = load_dataset(
                "shibing624/medical",
                data_files="medical.jsonl",
                split="train"
            )
            
            raw_dir = os.path.join(save_dir, "raw")
            Path(raw_dir).mkdir(parents=True, exist_ok=True)
            
            dataset.to_json(os.path.join(raw_dir, "medical_qa.json"))
            
            print(f"✓ 数据集已下载到: {raw_dir}")
            print(f"  总样本数: {len(dataset)}")
            
            return dataset
            
        except Exception as e2:
            print(f"方案2失败: {e2}")
            print("\n备选方案：使用其他医疗数据集")
            print("1. cMedQA2: https://github.com/zhangsheng93/cMedQA2")
            print("2. webMedQA: https://github.com/hejunqing/webMedQA")
            print("3. 或使用示例数据: python download_data.py --create_sample")
            return None


def create_sample_data(save_dir):
    """创建示例数据（用于测试）"""
    
    print("创建示例数据...")
    
    sample_data = [
        {
            "instruction": "回答医疗健康问题",
            "input": "感冒了应该吃什么药？",
            "output": "感冒的治疗需要根据症状选择合适的药物。如果是普通感冒，可以选择对症治疗的药物，如退烧药（对乙酰氨基酚）、止咳药等。建议多休息、多喝水，如果症状严重或持续时间长，请及时就医。"
        },
        {
            "instruction": "回答医疗健康问题",
            "input": "高血压患者饮食需要注意什么？",
            "output": "高血压患者饮食应注意：1. 低盐饮食，每天盐摄入量不超过6克；2. 多吃新鲜蔬菜水果；3. 控制脂肪摄入，少吃油腻食物；4. 适量摄入优质蛋白；5. 戒烟限酒；6. 保持规律饮食。"
        },
        {
            "instruction": "回答医疗健康问题",
            "input": "如何预防糖尿病？",
            "output": "预防糖尿病的方法包括：1. 保持健康体重，避免肥胖；2. 坚持规律运动，每周至少150分钟中等强度运动；3. 健康饮食，控制糖分和精制碳水化合物摄入；4. 定期体检，监测血糖；5. 保持良好作息，避免熬夜；6. 减轻压力，保持心情愉悦。"
        }
    ]
    
    raw_dir = os.path.join(save_dir, "raw")
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    
    with open(os.path.join(raw_dir, "sample_data.json"), "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 示例数据已创建: {raw_dir}/sample_data.json")


def main():
    parser = argparse.ArgumentParser(description="下载医疗问答数据集")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./data",
        help="保存目录"
    )
    parser.add_argument(
        "--create_sample",
        action="store_true",
        help="创建示例数据用于测试"
    )
    
    args = parser.parse_args()
    
    # 创建保存目录
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    if args.create_sample:
        create_sample_data(args.save_dir)
    else:
        dataset = download_medical_dataset(args.save_dir)
        if dataset is None:
            print("\n由于下载失败，创建示例数据供测试使用...")
            create_sample_data(args.save_dir)


if __name__ == "__main__":
    main()
