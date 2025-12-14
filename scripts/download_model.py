"""
从 ModelScope 下载模型脚本
"""

import argparse
from pathlib import Path


def download_model(model_name, save_dir):
    """从 ModelScope 下载模型"""
    try:
        from modelscope import snapshot_download
        
        print(f"从 ModelScope 下载模型: {model_name}")
        
        model_path = snapshot_download(
            model_name,
            cache_dir=save_dir,
            local_dir=f"{save_dir}/{model_name.replace('/', '_')}"
        )
        
        print(f"✓ 模型下载完成: {model_path}")
        return model_path
        
    except ImportError:
        print("❌ 请先安装依赖: pip install modelscope")
        return None
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="从 ModelScope 下载预训练模型")
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="模型名称"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./models",
        help="保存目录"
    )
    
    args = parser.parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    download_model(args.model_name, args.save_dir)


if __name__ == "__main__":
    main()
