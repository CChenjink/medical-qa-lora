"""
交互式推理脚本
用于测试微调后的模型
"""

import argparse
import torch

# 从 src 模块导入功能
from src.model import load_trained_model


def chat(model, tokenizer, instruction="回答医疗健康问题"):
    """交互式对话"""
    print("=" * 50)
    print("🏥 医疗问答助手")
    print("=" * 50)
    print("输入问题开始对话")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'clear' 清屏")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n💬 问题: ").strip()
            
            # 退出命令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 再见！")
                break
            
            # 清屏命令
            if user_input.lower() == 'clear':
                print("\033[2J\033[H")  # 清屏
                continue
            
            # 空输入
            if not user_input:
                continue
            
            # 构建提示
            prompt = f"{instruction}\n问题：{user_input}\n回答："
            
            # 生成回答
            inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
            
            print("\n🤔 思考中...")
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    do_sample=True,
                    top_p=0.8,
                    temperature=0.8,
                    repetition_penalty=1.1
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取回答部分（去掉提示）
            if "回答：" in response:
                response = response.split("回答：")[-1].strip()
            
            print(f"\n🏥 回答: {response}")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description="交互式推理测试")
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='模型路径，如 outputs/lora_medical/checkpoint-best'
    )
    parser.add_argument(
        '--base_model_path',
        type=str,
        default=None,
        help='基础模型路径（如果是 LoRA 模型需要提供）'
    )
    parser.add_argument(
        '--instruction',
        type=str,
        default="回答医疗健康问题",
        help='指令提示'
    )
    
    args = parser.parse_args()
    
    print("\n🚀 加载模型...")
    print(f"   模型路径: {args.model_path}")
    if args.base_model_path:
        print(f"   基础模型: {args.base_model_path}")
    
    # 加载模型
    model, tokenizer = load_trained_model(args.model_path, args.base_model_path)
    
    print("✓ 模型加载完成！\n")
    
    # 开始对话
    chat(model, tokenizer, args.instruction)


if __name__ == "__main__":
    main()
