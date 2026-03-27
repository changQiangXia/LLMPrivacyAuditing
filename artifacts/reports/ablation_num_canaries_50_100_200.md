# 主消融记录：`num_canaries=50/100/200`

更新时间：2026-03-27

## 1. 实验目的

在其余训练与评估配置保持一致的前提下，观察 canary 数量变化对训练损失、抽取攻击、approximate exposure 与 baseline MIA 的影响。

## 2. 固定条件

- 训练 seed：`0`
- `canary_repeats=5`
- 基础模型：`Qwen/Qwen2.5-0.5B-Instruct`
- LoRA 配置：沿用 `configs/train_lora.yaml`
- 攻击与评估配置：沿用 `configs/experiment.yaml`
- MIA 成员集：`data/processed/train_no_canary.jsonl`
- MIA 非成员集：`data/raw/repaired/nonmember.jsonl`

## 3. 结果对照

| 指标 | `num_canaries=50` | `num_canaries=100` | `num_canaries=200` |
| --- | --- | --- | --- |
| 训练集规模 | `1456` | `1706` | `2206` |
| `train_loss` | `2.3736726687504697` | `2.4773752464438386` | `2.6381317194360885` |
| extraction success rate | `0.0`，`0/2500` | `0.0`，`0/5000` | `0.0`，`0/10000` |
| avg exposure | `2.0695333058228114` | `1.9008463671733251` | `1.6609825235608724` |
| MIA AUC loss | `0.5101775281180013` | `0.511355280040146` | `0.512527471610932` |
| MIA AUC neighbourhood | `0.5165945211077888` | `0.5165114633593695` | `0.5155175505540195` |

## 4. 当前观察

- 当前单次运行下，avg exposure 呈现 `50 > 100 > 200` 的顺序
- 当前单次运行下，`train_loss` 随 `num_canaries` 增大而上升
- 三组配置的完整 canary 抽取攻击均为 `0`
- MIA 指标整体变化较小，维持在弱信号区间

## 5. 解释边界

- 当前结果仅覆盖 **1 个训练 seed**
- `num_canaries` 改变后，训练集规模、抽取攻击总尝试数与 exposure 汇总规模同步变化
- 当前 exposure 口径为 **approximate exposure**
- 当前 MIA 口径为 **baseline MIA**
- 因此该结果可作为 **主消融首轮趋势记录**，当前不适合表述为稳定结论

## 6. 结果定位

- `50`
  - 训练：`outputs/experiments/train/ablation_num_canaries50_seed0`
  - attack：`outputs/experiments/attack_extract/ablation_num_canaries50_seed0_full`
  - exposure：`outputs/experiments/exposure/ablation_num_canaries50_seed0_full`
  - mia：`outputs/experiments/mia/ablation_num_canaries50_seed0_full`
- `100`
  - 训练：`outputs/experiments/train/ablation_num_canaries100_seed0`
  - attack：`outputs/experiments/attack_extract/ablation_num_canaries100_seed0_full`
  - exposure：`outputs/experiments/exposure/ablation_num_canaries100_seed0_full`
  - mia：`outputs/experiments/mia/ablation_num_canaries100_seed0_full`
- `200`
  - 对应标准 baseline seed0：
  - 训练：`outputs/experiments/train/baseline_standard_seed0`
  - attack：`outputs/experiments/attack_extract/baseline_standard_seed0_full`
  - exposure：`outputs/experiments/exposure/baseline_standard_seed0_full`
  - mia：`outputs/experiments/mia/baseline_standard_seed0_full`
