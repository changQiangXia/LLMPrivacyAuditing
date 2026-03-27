# 实验矩阵说明

更新时间：2026-03-27

实验矩阵主文件为 `configs/experiment_matrix.yaml`。当前执行策略如下。

## baseline

- **base_model**：基础模型直接审计，不做 LoRA 微调
- **lora_standard**：标准 LoRA 训练，使用插入 canary 的训练集
- **lora_no_canary**：不插入 canary 的 LoRA 训练
- **lora_dedup**：开启 dedup 的 LoRA 训练
- **decode_safe**：推理阶段防护，仅作为补充对比项

当前已完成 baseline：

- **base_model**
- **lora_standard**：seed `0/1/2`
- **lora_no_canary**：seed `0`
- **lora_dedup**：seed `0`

## 主消融

- **num_canaries**
- **canary_repeats**
- **epochs**
- **lr**
- **lora_r**

当前已完成的主消融起点：

- **num_canaries=50, canary_repeats=5, seed=0**
  - 训练目录：`outputs/experiments/train/ablation_num_canaries50_seed0`
  - 审计目录：`outputs/experiments/attack_extract/ablation_num_canaries50_seed0_full`
- **num_canaries=100, canary_repeats=5, seed=0**
  - 训练目录：`outputs/experiments/train/ablation_num_canaries100_seed0`
  - 审计目录：`outputs/experiments/attack_extract/ablation_num_canaries100_seed0_full`
- **num_canaries=200, canary_repeats=5, seed=0**
  - 对应标准 baseline seed0

## 补充消融

- **max_len**
- **lora_alpha**
- **lora_dropout**
- **target_modules**
- **dedup**
- **num_samples_per_prefix**
- **temperature**
- **top_p**
- **num_reference**
- **neigh_k**
- **word_drop**

## 执行顺序约束

- 先完成最小 baseline 集合
- 再执行主消融
- 补充消融在主结果稳定后扩展
- 每一轮正式实验执行前，需锁定配置与 run_id
