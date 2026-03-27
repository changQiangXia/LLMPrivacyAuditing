# 主消融记录：`num_canaries=50` 对比 `num_canaries=200`

更新时间：2026-03-27

## 1. 实验目的

评估在其余训练与评估配置保持不变时，降低 canary 数量是否会改变隐私风险信号。

## 2. 对照设置

| 条目 | 标准 baseline seed0 | 主消融 seed0 |
| --- | --- | --- |
| 训练集 | `train_with_canary.jsonl` | `train_with_canary_num50.jsonl` |
| canary 数量 | `200` | `50` |
| repeats | `5` | `5` |
| 训练 seed | `0` | `0` |
| LoRA 配置 | 与 `configs/train_lora.yaml` 一致 | 与 `configs/train_lora.yaml` 一致 |
| 攻击配置 | 与 `configs/experiment.yaml` 一致 | 与 `configs/experiment.yaml` 一致 |

## 3. 结果对照

| 指标 | `num_canaries=200` | `num_canaries=50` |
| --- | --- | --- |
| `train_loss` | `2.6381317194360885` | `2.3736726687504697` |
| extraction success rate | `0.0`，`0/10000` | `0.0`，`0/2500` |
| avg exposure | `1.6609825235608724` | `2.0695333058228114` |
| MIA AUC loss | `0.512527471610932` | `0.5101775281180013` |
| MIA AUC neighbourhood | `0.5155175505540195` | `0.5165945211077888` |

## 4. 当前观察

- 当前单次运行下，`num_canaries=50` 的 avg exposure 高于 `num_canaries=200`
- 抽取攻击在两组配置下均未命中完整 canary
- MIA 指标变化幅度较小，仍处于弱信号区间

## 5. 解释边界

- 当前对照仅覆盖 **1 个训练 seed**
- `num_canaries` 改变后，攻击总尝试数与 exposure 汇总规模同步变化
- 当前实现口径仍为 **approximate exposure** 与 **baseline MIA**
- 因此当前结果仅可作为 **主消融的首轮观察**，暂不形成稳定趋势结论

## 6. 结果定位

- 训练：`outputs/experiments/train/ablation_num_canaries50_seed0`
- 抽取攻击：`outputs/experiments/attack_extract/ablation_num_canaries50_seed0_full`
- Exposure：`outputs/experiments/exposure/ablation_num_canaries50_seed0_full`
- MIA：`outputs/experiments/mia/ablation_num_canaries50_seed0_full`
- 图表：`outputs/experiments/plots/ablation_num_canaries50_seed0_full`
