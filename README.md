# 大模型微调项目：中文医疗问答系统

基于 **Qwen2.5-4B-Instruct** 的医疗问答系统微调项目，使用 LoRA/QLoRA 参数高效微调方法。

## 📋 项目概述

- **基座模型**：Qwen2.5-4B-Instruct（4B参数）
- **任务**：中文医疗健康问答
- **方法**：LoRA / QLoRA 参数高效微调
- **核心实验**：Baseline + LoRA(1k/5k/10k) + QLoRA(1k/5k/10k)
- **算力**：免费 GPU（Google Colab / Kaggle）

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 下载 Qwen2.5-4B 模型
python scripts/download_model.py \
    --model_name Qwen/Qwen2.5-4B-Instruct \
    --source modelscope

# 下载和预处理数据
python scripts/download_data.py
python scripts/preprocess_data.py

# 准备不同规模数据集（1k/5k/10k）
python scripts/prepare_data_splits.py
```

### 2. 运行实验

根据你的分工选择对应的实验：

```bash
# 成员1：Baseline 评估
python evaluate.py \
    --model_path ./models/qwen2.5-4b \
    --test_file ./data/processed/test.json

# 成员2：LoRA 1k + 5k
python train.py --config configs/lora_1k.yaml
python train.py --config configs/lora_5k.yaml

# 成员3：LoRA 10k
python train.py --config configs/lora_10k.yaml

# 成员4：QLoRA 1k + 5k
python train.py --config configs/qlora_1k.yaml
python train.py --config configs/qlora_5k.yaml

# 成员5：QLoRA 10k
python train.py --config configs/qlora_10k.yaml
```

### 3. 评估和测试

```bash
# 评估模型
python evaluate.py \
    --model_path outputs/lora_1k/checkpoint-best \
    --base_model_path models/qwen2.5-4b

# 交互测试
python inference.py \
    --model_path outputs/lora_1k/checkpoint-best \
    --base_model_path models/qwen2.5-4b
```

## 👥 小组分工

| 成员 | 实验 | 主要工作 | 预计时间 |
|------|------|----------|----------|
| 成员1 | Baseline | 评估框架 + 原始模型测试 + 报告整合 | 轻 |
| 成员2 | LoRA 1k/5k | LoRA 小规模实验（2个） | 中 |
| 成员3 | LoRA 10k | LoRA 大规模实验 + 数据规模分析 | 重 |
| 成员4 | QLoRA 1k/5k | QLoRA 小规模实验（2个） | 中 |
| 成员5 | QLoRA 10k | QLoRA 大规模实验 + 方法对比总结 | 重 |

**详细分工**：查看 [docs/小组分工说明.md](docs/小组分工说明.md)

## 📊 实验矩阵

| 实验编号 | 方法 | 数据量 | 配置文件 | 输出目录 | 负责人 |
|---------|------|--------|----------|----------|--------|
| EXP-00 | Baseline | - | - | outputs/baseline | 成员1 |
| EXP-01 | LoRA | 1k | lora_1k.yaml | outputs/lora_1k | 成员2 |
| EXP-02 | LoRA | 5k | lora_5k.yaml | outputs/lora_5k | 成员2 |
| EXP-03 | LoRA | 10k | lora_10k.yaml | outputs/lora_10k | 成员3 |
| EXP-04 | QLoRA | 1k | qlora_1k.yaml | outputs/qlora_1k | 成员4 |
| EXP-05 | QLoRA | 5k | qlora_5k.yaml | outputs/qlora_5k | 成员4 |
| EXP-06 | QLoRA | 10k | qlora_10k.yaml | outputs/qlora_10k | 成员5 |

## 📁 项目结构

```
project/
├── README.md                # 本文件
├── requirements.txt         # 依赖列表
│
├── configs/                 # 配置文件（8个）
│   ├── lora_config.yaml    # LoRA 配置模板（仅供参考）
│   ├── lora_1k.yaml        # LoRA 1k 实验配置
│   ├── lora_5k.yaml        # LoRA 5k 实验配置
│   ├── lora_10k.yaml       # LoRA 10k 实验配置
│   ├── qlora_config.yaml   # QLoRA 配置模板（仅供参考）
│   ├── qlora_1k.yaml       # QLoRA 1k 实验配置
│   ├── qlora_5k.yaml       # QLoRA 5k 实验配置
│   └── qlora_10k.yaml      # QLoRA 10k 实验配置
│
├── scripts/                 # 工具脚本（5个）
│   ├── download_model.py
│   ├── download_data.py
│   ├── preprocess_data.py
│   ├── prepare_data_splits.py
│   └── summarize_results.py
│
├── src/                     # 源代码模块（4个）
│   ├── model.py
│   ├── data_loader.py
│   ├── trainer.py
│   └── evaluator.py
│
├── train.py                 # 训练入口
├── evaluate.py              # 评估入口
├── inference.py             # 推理入口
│
├── docs/                    # 文档
│   ├── 小组分工说明.md
│   ├── 使用指南.md
│   └── 免费算力平台说明.md
│
├── data/                    # 数据目录（运行后生成）
├── models/                  # 模型目录（运行后生成）
└── outputs/                 # 输出目录（训练后生成）
```

## 📚 文档说明

### 必读文档
- **README.md** - 本文件，项目说明和快速开始
- **docs/小组分工说明.md** - 详细的5人分工方案和协作流程
- **docs/使用指南.md** - 详细的操作步骤和命令说明

### 参考文档
- **docs/免费算力平台说明.md** - Google Colab、Kaggle 等平台使用方法

### 配置文件说明
- **lora_config.yaml / qlora_config.yaml** - 配置模板，仅供参考，不直接用于训练
- **lora_1k/5k/10k.yaml** - LoRA 实验的实际配置文件
- **qlora_1k/5k/10k.yaml** - QLoRA 实验的实际配置文件

## 💻 环境要求

### 硬件
- **最低**：Google Colab 免费 T4 GPU（15GB）
- **推荐**：Kaggle P100 GPU（16GB）
- **Qwen2.5-4B**：显存占用 ~8GB，训练更快

### 软件
- Python 3.8+
- PyTorch 2.0+
- 完整依赖见 [requirements.txt](requirements.txt)

### 免费算力平台
1. **Google Colab** - 免费 T4 GPU，每次 12 小时
2. **Kaggle Notebooks** - 免费 P100 GPU，每周 30 小时
3. **AutoDL** - 新用户有免费额度

详见：[docs/免费算力平台说明.md](docs/免费算力平台说明.md)

## ❓ 常见问题

**Q: 显存不够怎么办？**  
A: 使用 QLoRA（4-bit 量化），只需 ~6GB 显存

**Q: 训练需要多长时间？**  
A: 1k数据 ~0.5h，5k数据 ~1h，10k数据 ~2h

**Q: 如何选择 LoRA 还是 QLoRA？**  
A: LoRA 效果更好（~10GB显存），QLoRA 显存更少（~6GB）

## 📖 参考资源

- [Qwen2.5 官方文档](https://github.com/QwenLM/Qwen2.5)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [PEFT 库文档](https://huggingface.co/docs/peft)

---

**开始使用**：按照"快速开始"部分的步骤操作，遇到问题查看 docs/ 目录下的文档 🚀
