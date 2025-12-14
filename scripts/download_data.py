"""
下载 shibing624/medical 中文医疗问答数据集
"""

import os
import json
import argparse
from pathlib import Path


def download_medical_dataset(save_dir, max_samples=None):
    """下载 shibing624/medical 中文医疗问答数据集
    
    Args:
        save_dir: 保存目录
        max_samples: 最大样本数，None表示下载全部数据
    """
    
    print("正在下载 shibing624/medical 数据集...")
    
    try:
        from huggingface_hub import hf_hub_download
        import json
        import random
        
        # 设置镜像
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        print("从 finetune/train_zh_0.json 下载数据...")
        if max_samples:
            print(f"将采样 {max_samples:,} 条数据（原始数据约195万条）")
        else:
            print("将下载全部数据（约195万条）")
        
        raw_dir = os.path.join(save_dir, "raw")
        Path(raw_dir).mkdir(parents=True, exist_ok=True)
        
        # 下载 train_zh_0.json
        print(f"\n下载: finetune/train_zh_0.json")
        file_path = hf_hub_download(
            repo_id="shibing624/medical",
            filename="finetune/train_zh_0.json",
            repo_type="dataset"
        )
        
        # 读取数据（JSONL 格式：每行一个 JSON 对象）
        print("读取数据（JSONL 格式）...")
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        
        original_count = len(data)
        print(f"原始数据: {original_count:,} 条")
        
        # 如果设置了最大样本数，进行采样
        if max_samples and len(data) > max_samples:
            print(f"随机采样 {max_samples:,} 条...")
            random.seed(42)  # 设置随机种子以保证可复现
            data = random.sample(data, max_samples)
        
        # 保存到目标位置
        output_file = os.path.join(raw_dir, "medical_qa.json")
        print(f"保存数据到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 数据集下载完成!")
        print(f"  文件位置: {output_file}")
        print(f"  样本数量: {len(data):,} 条")
        print(f"\n数据集字段: {list(data[0].keys())}")
        print(f"示例数据: {data[0]}")
        print(f"\n提示: 使用 prepare_data_splits.py 脚本划分训练集、验证集、测试集")
        
        return data
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n备选方案：")
        print("1. 使用示例数据: python download_data.py --create_sample")
        print("2. 手动下载:")
        print("   - cMedQA2: https://github.com/zhangsheng93/cMedQA2")
        print("   - webMedQA: https://github.com/hejunqing/webMedQA")
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
    parser = argparse.ArgumentParser(description="下载 shibing624/medical 医疗问答数据集")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./data",
        help="保存目录"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200000,
        help="最大样本数 (默认: 200000, 设为0表示下载全部数据)"
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
        max_samples = None if args.max_samples == 0 else args.max_samples
        dataset = download_medical_dataset(args.save_dir, max_samples)
        if dataset is None:
            print("\n由于下载失败，创建示例数据供测试使用...")
            create_sample_data(args.save_dir)


if __name__ == "__main__":
    main()
