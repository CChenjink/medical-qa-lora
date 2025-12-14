"""
下载模型脚本
支持从 ModelScope 或 Hugging Face 下载
"""

import os
import argparse
from pathlib import Path


def download_from_modelscope(model_name, save_dir):
    """从 ModelScope 下载模型（国内推荐）"""
    try:
        from modelscope import snapshot_download
        
        model_dir = snapshot_download(
            model_name,
            cache_dir=save_dir,
            revision='master'
        )
        print(f"✓ 模型已下载到: {model_dir}")
        return model_dir
    except ImportError:
        print("请先安装 modelscope: pip install modelscope")
        return None


def download_from_huggingface(model_name, save_dir):
    """从 Hugging Face 下载模型"""
    try:
        from transformers import AutoTokenizer, AutoModel
        
        print(f"正在下载模型: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=save_dir
        )
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=save_dir
        )
        
        # 保存到本地
        local_path = os.path.join(save_dir, model_name.split('/')[-1])
        tokenizer.save_pretrained(local_path)
        model.save_pretrained(local_path)
        
        print(f"✓ 模型已下载到: {local_path}")
        return local_path
    except Exception as e:
        print(f"下载失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="下载预训练模型")
    parser.add_argument(
        "--model_name",
        type=str,
        default="ZhipuAI/chatglm3-6b",
        help="模型名称，如 ZhipuAI/chatglm3-6b 或 qwen/Qwen-7B-Chat"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="modelscope",
        choices=["modelscope", "huggingface"],
        help="下载源"
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
    
    print(f"开始下载模型: {args.model_name}")
    print(f"下载源: {args.source}")
    print(f"保存目录: {args.save_dir}")
    print("-" * 50)
    
    if args.source == "modelscope":
        download_from_modelscope(args.model_name, args.save_dir)
    else:
        download_from_huggingface(args.model_name, args.save_dir)


if __name__ == "__main__":
    main()
