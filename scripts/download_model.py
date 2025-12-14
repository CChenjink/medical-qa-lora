"""
从 Hugging Face 下载模型脚本
"""

import os
import argparse
from pathlib import Path


def download_model(model_name, save_dir):
    """从 Hugging Face 下载模型"""
    try:
        from huggingface_hub import snapshot_download
        
        # 设置镜像
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        print(f"使用 Hugging Face 镜像: https://hf-mirror.com\n")
        
        print(f"正在下载模型: {model_name}")
        print(f"保存目录: {save_dir}")
        print("-" * 50)
        
        # 使用 snapshot_download 下载整个模型仓库
        print("\n下载模型文件...")
        local_path = snapshot_download(
            repo_id=model_name,
            cache_dir=save_dir,
            resume_download=True,
            local_dir=os.path.join(save_dir, model_name.replace('/', '_')),
            local_dir_use_symlinks=False
        )
        
        print(f"\n✓ 模型已下载到: {local_path}")
        return local_path
    except ImportError:
        print("\n❌ 请先安装依赖: pip install huggingface_hub")
        return None
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n提示：")
        print("1. 检查网络连接")
        print("2. 确认模型名称正确")
        print("3. 如果是私有模型，需要登录: huggingface-cli login")
        return None


def main():
    parser = argparse.ArgumentParser(description="从 Hugging Face 下载预训练模型")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="模型名称，如 Qwen/Qwen2.5-3B-Instruct"
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
