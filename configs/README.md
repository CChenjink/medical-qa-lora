# 配置文件说明

本目录包含不同数据规模和训练方法的配置文件。

## 数据规模

根据 `scripts/prepare_data_splits.py` 脚本，我们准备了以下数据规模：

- **20k**: 20,000 条训练样本
- **50k**: 50,000 条训练样本
- **100k**: 100,000 条训练样本

## 配置文件列表

### LoRA 配置

| 配置文件 | 数据规模 | 训练步数 | Warmup | Logging | Save/Eval |
|---------|---------|---------|--------|---------|-----------|
| `lora_20k.yaml` | 20k | ~3,750 steps/epoch | 10% | 100 | 1200 |
| `lora_50k.yaml` | 50k | ~9,375 steps/epoch | 10% | 200 | 3000 |
| `lora_100k.yaml` | 100k | ~18,750 steps/epoch | 10% | 400 | 6000 |
| `lora_config.yaml` | 20k (默认) | - | 10% | 100 | 1200 |

### QLoRA 配置

| 配置文件 | 数据规模 | 训练步数 | Warmup | Logging | Save/Eval |
|---------|---------|---------|--------|---------|-----------|
| `qlora_20k.yaml` | 20k | ~1,250 steps/epoch | 10% | 100 | 1200 |
| `qlora_50k.yaml` | 50k | ~3,125 steps/epoch | 10% | 200 | 3000 |
| `qlora_100k.yaml` | 100k | ~6,250 steps/epoch | 10% | 400 | 6000 |
| `qlora_config.yaml` | 20k (默认) | - | 10% | 100 | 1200 |

## 配置参数说明

### 模型配置
- **model_name_or_path**: `Qwen/Qwen2.5-3B-Instruct` - 统一使用 3B 模型
- **model_type**: `qwen2`

### LoRA 参数
- **r**: 8 - LoRA rank
- **lora_alpha**: 32 - LoRA alpha 参数
- **target_modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj` - 目标注意力层
- **lora_dropout**: 0.1
- **task_type**: `CAUSAL_LM`

### 训练参数

#### LoRA
- **batch_size**: 4 (per device)
- **gradient_accumulation_steps**: 4
- **effective_batch_size**: 16

#### QLoRA
- **batch_size**: 8 (per device) - 由于量化，可以使用更大的 batch size
- **gradient_accumulation_steps**: 2
- **effective_batch_size**: 16

#### 通用参数
- **num_train_epochs**: 3
- **learning_rate**: 1e-4
- **warmup_ratio**: 0.1 (10% 的训练步数用于 warmup)
- **fp16**: true
- **evaluation_strategy**: "steps"
- **metric_for_best_model**: "loss"

### 数据参数
- **max_source_length**: 512
- **max_target_length**: 512
- **validation_file**: `./data/processed/dev.json`
- **test_file**: `./data/processed/test.json`

## 使用方法

### 训练

```bash
# 使用 LoRA 训练 20k 数据
python train.py --config configs/lora_20k.yaml

# 使用 QLoRA 训练 50k 数据
python train.py --config configs/qlora_50k.yaml

# 使用 100k 数据集
python train.py --config configs/lora_100k.yaml
```

### 评估

```bash
# 评估模型
python evaluate.py --config configs/lora_20k.yaml --checkpoint ./outputs/lora_20k/checkpoint-best
```

### 推理

```bash
# 使用训练好的模型进行推理
python inference.py --config configs/lora_20k.yaml --checkpoint ./outputs/lora_20k/checkpoint-best
```

## 训练步数计算

训练步数 = (数据集大小 / effective_batch_size) × num_epochs

### LoRA (effective_batch_size = 16)
- 20k: (20000 / 16) × 3 ≈ 3,750 steps
- 50k: (50000 / 16) × 3 ≈ 9,375 steps
- 100k: (100000 / 16) × 3 ≈ 18,750 steps

### QLoRA (effective_batch_size = 16)
- 20k: (20000 / 16) × 3 ≈ 3,750 steps
- 50k: (50000 / 16) × 3 ≈ 9,375 steps
- 100k: (100000 / 16) × 3 ≈ 18,750 steps

## 注意事项

1. **数据准备**: 使用前需要先运行 `scripts/prepare_data_splits.py` 生成不同规模的数据集
2. **显存要求**: 
   - LoRA: 需要约 12-16GB 显存
   - QLoRA: 需要约 8-12GB 显存（4-bit 量化）
3. **训练时间**: 根据数据规模和硬件配置，训练时间从几小时到几天不等
4. **模型选择**: 所有配置统一使用 `Qwen2.5-3B-Instruct` 模型，确保实验的一致性
