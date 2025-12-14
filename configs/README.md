# 配置文件说明

本目录包含不同数据规模和训练方法的配置文件。

## 数据规模

根据 `scripts/prepare_data_splits.py` 脚本，我们准备了以下数据规模：

- **10k**: 10,000 条训练样本
- **20k**: 20,000 条训练样本
- **40k**: 40,000 条训练样本（完整训练集）

**数据来源**：从原始20万条数据清洗后得到约5万条高质量数据，按8:1:1划分为训练集(40k)、验证集(5k)、测试集(5k)

## 配置文件列表

### LoRA 配置

| 配置文件 | 数据规模 | 训练步数 | Warmup | Logging | Save/Eval | 预计时间(T4) |
|---------|---------|---------|--------|---------|-----------|-------------|
| `lora_10k.yaml` | 10k | ~1,875 steps/epoch | 10% | 50 | 600 | ~0.5-1h |
| `lora_20k.yaml` | 20k | ~3,750 steps/epoch | 10% | 100 | 1200 | ~1-1.5h |
| `lora_40k.yaml` | 40k | ~7,500 steps/epoch | 10% | 200 | 2400 | ~2-3h |
| `lora_config.yaml` | 10k (默认) | - | 10% | 50 | 600 | ~0.5-1h |

### QLoRA 配置

| 配置文件 | 数据规模 | 训练步数 | Warmup | Logging | Save/Eval | 预计时间(T4) |
|---------|---------|---------|--------|---------|-----------|-------------|
| `qlora_10k.yaml` | 10k | ~1,875 steps/epoch | 10% | 50 | 600 | ~0.7-1.2h |
| `qlora_20k.yaml` | 20k | ~3,750 steps/epoch | 10% | 100 | 1200 | ~1.5-2h |
| `qlora_40k.yaml` | 40k | ~7,500 steps/epoch | 10% | 200 | 2400 | ~3-4h |
| `qlora_config.yaml` | 10k (默认) | - | 10% | 50 | 600 | ~0.7-1.2h |

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
- **validation_file**: `./data/processed/dev.json` (5,000条)
- **test_file**: `./data/processed/test.json` (5,000条)

## 使用方法

### 训练

```bash
# 使用 LoRA 训练 10k 数据
python train.py --config configs/lora_10k.yaml

# 使用 QLoRA 训练 20k 数据
python train.py --config configs/qlora_20k.yaml

# 使用完整训练集 (40k)
python train.py --config configs/lora_40k.yaml
```

### 评估

```bash
# 评估模型
python evaluate.py --config configs/lora_10k.yaml --checkpoint ./outputs/lora_10k/checkpoint-best
```

### 推理

```bash
# 使用训练好的模型进行推理
python inference.py --config configs/lora_10k.yaml --checkpoint ./outputs/lora_10k/checkpoint-best
```

## 训练步数计算

训练步数 = (数据集大小 / effective_batch_size) × num_epochs

### LoRA & QLoRA (effective_batch_size = 16)
- 10k: (10000 / 16) × 3 ≈ 1,875 steps
- 20k: (20000 / 16) × 3 ≈ 3,750 steps
- 40k: (40000 / 16) × 3 ≈ 7,500 steps

## 训练时间估算

基于 T4 GPU 的实际测试：

| 数据规模 | LoRA | QLoRA |
|---------|------|-------|
| 10k | 0.5-1h | 0.7-1.2h |
| 20k | 1-1.5h | 1.5-2h |
| 40k | 2-3h | 3-4h |

**说明**：
- QLoRA 由于 4-bit 量化的开销，训练时间约为 LoRA 的 1.3-1.5 倍
- 实际时间会根据 GPU 型号、数据复杂度等因素有所波动
- 使用更好的 GPU (如 V100, A100) 可以显著缩短训练时间

## 注意事项

1. **数据准备**：使用前需要先运行 `scripts/prepare_data_splits.py` 生成不同规模的数据集
2. **显存要求**: 
   - LoRA: 需要约 12-14GB 显存
   - QLoRA: 需要约 8-10GB 显存（4-bit 量化）
3. **训练时间**: 根据数据规模，训练时间从0.5小时到4小时不等
4. **模型选择**: 所有配置统一使用 `Qwen2.5-3B-Instruct` 模型，确保实验的一致性
5. **数据质量**: 训练数据经过严格清洗，质量高于原始数据

## 实验建议

1. **快速验证**：先用 10k 数据快速验证流程和超参数
2. **性能对比**：依次运行 10k、20k、40k 实验，观察数据规模对性能的影响
3. **方法选择**：
   - 显存充足：优先使用 LoRA（效果略好）
   - 显存受限：使用 QLoRA（显存需求低）
4. **时间规划**：
   - 10k 实验适合快速迭代和调参
   - 40k 实验用于最终性能评估
