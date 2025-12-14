"""
从 Hugging Face 下载模型脚本
"""

import os
import argparse
from pathlib import Path


def download_model(model_name, save_dir):
    """从 Hugging Face 下载模型"""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # 自动使用镜像加速
        if 'HF_ENDPOINT' not in os.environ:
            os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            print(f"使用 Hugging Face 镜像: https://hf-mirror.com\n")
        
        print(f"正在下载模型: {model_name}")
        print(f"保存目录: {save_dir}")
        print("-" * 50)
        
        # 下载 tokenizer
        print("\n1. 下载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=save_dir
        )
        
        # 下载模型（不加载到内存，只下载文件）
        print("\n2. 下载模型文件...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=save_dir,
            low_cpu_mem_usage=True
        )
        
        # 保存到本地目录
        local_path = os.path.join(save_dir, model_name.replace('/', '_'))
        print(f"\n3. 保存到本地: {local_path}")
        
        tokenizer.save_pretrained(local_path)
        model.save_pretrained(local_path)
        
        print(f"\n✓ 模型已下载并保存到: {local_path}")
        return local_path
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="从 Hugging Face 下载预训练模型")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-4B-Instruct",
        help="模型名称，如 Qwen/Qwen2.5-4B-Instruct"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./models",
        help="保存目录"
    )
    
    args = parser.parse_args()
    
    # 创建保存目录
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    
    download_model(args.model_name, args.save_dir)


if __name__ == "__main__":
    main()
