# 大模型微调项目：中文医疗问答系统

基于 **Qwen2.5-3B-Instruct** 的医疗问答系统微调项目，使用 LoRA/QLoRA 参数高效微调方法。

## 📋 项目概述

- **基座模型**：Qwen2.5-3B-Instruct（3B参数）
- **任务**：中文医疗健康问答
- **方法**：LoRA / QLoRA 参数高效微调
- **核心实验**：研究训练数据规模对模型性能的影响（10k/20k/40k/60k）
- **数据来源**：从10万原始数据清洗得到7-8万高质量数据
- **算力**：免费 GPU（Google Colab / Kaggle）

## 🚀 快速开始

### 1. 克隆仓库

```bash
# 安装 Git LFS（用于下载大文件）
git lfs install

# 克隆仓库（自动下载数据集）
git clone https://github.com/CChenjink/medical-qa-lora.git
cd medical-qa-lora/project
```

**注意**：本项目使用 Git LFS 管理数据集文件。如果克隆后 `data/raw/medical_qa.json` 只有几百字节，说明 LFS 文件未下载，运行：
```bash
git lfs pull
```

### 2. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 下载 Qwen2.5-3B 模型
python scripts/download_model.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --save_dir ./models

# 预处理数据
python scripts/preprocess_data.py

# 准备不同规模数据集（10k/20k/40k/60k）
python scripts/prepare_data_splits.py
```

### 3. 运行实验

根据不同数据规模进行训练：

```bash
# LoRA 训练
python train.py --config configs/lora_10k.yaml   # 10k 数据
python train.py --config configs/lora_20k.yaml   # 20k 数据
python train.py --config configs/lora_40k.yaml   # 40k 数据
python train.py --config configs/lora_60k.yaml   # 60k 数据

# QLoRA 训练（显存更少）
python train.py --config configs/qlora_10k.yaml   # 10k 数据
python train.py --config configs/qlora_20k.yaml   # 20k 数据
python train.py --config configs/qlora_40k.yaml   # 40k 数据
python train.py --config configs/qlora_60k.yaml   # 60k 数据
```

### 4. 评估和测试

```bash
# 评估模型
python evaluate.py \
    --model_path outputs/lora_20k/checkpoint-best \
    --base_model_path models/Qwen_Qwen2.5-3B-Instruct

# 交互测试
python inference.py \
    --model_path outputs/lora_20k/checkpoint-best \
    --base_model_path models/Qwen_Qwen2.5-3B-Instruct
```

## 📊 实验设计

本项目研究训练数据规模对模型性能的影响，设计了以下实验：

| 实验编号 | 方法 | 数据量 | 配置文件 | 输出目录 | 预计训练时间 (T4) |
|---------|------|--------|----------|----------|-------------------|
| EXP-01 | LoRA | 10k | lora_10k.yaml | outputs/lora_10k | ~0.5-1h |
| EXP-02 | LoRA | 20k | lora_20k.yaml | outputs/lora_20k | ~1-1.5h |
| EXP-03 | LoRA | 40k | lora_40k.yaml | outputs/lora_40k | ~2-3h |
| EXP-04 | LoRA | 60k | lora_60k.yaml | outputs/lora_60k | ~3-4.5h |
| EXP-05 | QLoRA | 10k | qlora_10k.yaml | outputs/qlora_10k | ~0.7-1.2h |
| EXP-06 | QLoRA | 20k | qlora_20k.yaml | outputs/qlora_20k | ~1.5-2h |
| EXP-07 | QLoRA | 40k | qlora_40k.yaml | outputs/qlora_40k | ~3-4h |
| EXP-08 | QLoRA | 60k | qlora_60k.yaml | outputs/qlora_60k | ~4.5-6h |

**实验目标**：
- 对比不同数据规模（10k vs 20k vs 40k vs 60k）对模型性能的影响
- 对比不同微调方法（LoRA vs QLoRA）的效果差异
- 分析数据规模与模型性能的关系曲线

## 📁 项目结构

```
project/
├── README.md                # 本文件
├── requirements.txt         # 依赖列表
│
├── configs/                 # 配置文件
│   ├── README.md           # 配置文件详细说明
│   ├── lora_config.yaml    # LoRA 默认配置
│   ├── lora_10k.yaml       # LoRA 10k 实验配置
│   ├── lora_20k.yaml       # LoRA 20k 实验配置
│   ├── lora_40k.yaml       # LoRA 40k 实验配置
│   ├── lora_60k.yaml       # LoRA 60k 实验配置
│   ├── qlora_config.yaml   # QLoRA 默认配置
│   ├── qlora_10k.yaml      # QLoRA 10k 实验配置
│   ├── qlora_20k.yaml      # QLoRA 20k 实验配置
│   ├── qlora_40k.yaml      # QLoRA 40k 实验配置
│   └── qlora_60k.yaml      # QLoRA 60k 实验配置
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
- **lora_config.yaml / qlora_config.yaml** - 默认配置（使用10k数据）
- **lora_10k/20k/40k/60k.yaml** - LoRA 不同数据规模的配置文件
- **qlora_10k/20k/40k/60k.yaml** - QLoRA 不同数据规模的配置文件
- **configs/README.md** - 配置文件的详细说明文档

## 💻 环境要求

### 硬件
- **最低**：Google Colab 免费 T4 GPU（15GB）
- **推荐**：Kaggle P100 GPU（16GB）
- **Qwen2.5-3B**：显存占用 ~6GB，训练更快

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
A: 使用 QLoRA（4-bit 量化），只需 ~4GB 显存

**Q: 训练需要多长时间？**  
A: 使用T4 GPU，LoRA方法：10k约0.5-1h，20k约1-1.5h，40k约2-3h，60k约3-4.5h；QLoRA稍慢约1.3-1.5倍

**Q: 如何选择 LoRA 还是 QLoRA？**  
A: LoRA 效果更好（~8GB显存），QLoRA 显存更少（~4GB）

## 📖 参考资源

- [Qwen2.5 官方文档](https://github.com/QwenLM/Qwen2.5)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314)
- [PEFT 库文档](https://huggingface.co/docs/peft)

---

**开始使用**：按照"快速开始"部分的步骤操作，遇到问题查看 docs/ 目录下的文档 🚀
