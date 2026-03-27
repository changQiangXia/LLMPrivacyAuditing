# 最终隐私审计报告

更新时间：2026-03-27

## 1. 审计范围

本轮审计覆盖以下对象与流程：

- 基础模型：**Qwen2.5-0.5B-Instruct**
- 模型来源：**ModelScope**
- 训练方式：LoRA 微调
- 风险评估：
  - canary 抽取攻击
  - **approximate exposure**
  - baseline MIA
- baseline：
  - `base_model`
  - `lora_standard`
  - `lora_no_canary`
  - `lora_dedup`
  - `decode_safe`
- ablation：
  - `num_canaries`
  - `canary_repeats`
  - `epochs`
  - `lr`
  - `max_len`
  - `lora_r`
  - `lora_alpha`
  - `lora_dropout`
  - `target_modules`
  - `dedup`
  - `num_samples_per_prefix`
  - `temperature/top_p`
  - `num_reference`
  - `neigh_k/word_drop`

## 2. 数据、模型与威胁模型

- 正式成员输入：`data/raw/repaired/train.jsonl`
- 正式非成员输入：`data/raw/repaired/nonmember.jsonl`
- 标准训练集：`data/processed/train_with_canary.jsonl`
- 无 canary 训练集：`data/processed/train_no_canary.jsonl`
- 基础模型本地目录：`models/Qwen2.5-0.5B-Instruct`
- 威胁模型：
  - 攻击者可访问生成接口
  - 攻击者可控制采样参数
  - 攻击者不能访问梯度或优化器状态
- 指标口径：
  - 抽取攻击成功定义为**完整 canary 命中**
  - exposure 为 **approximate exposure based on full-canary NLL ranking**
  - MIA 为 baseline MIA，包含 `loss_threshold` 与 `neighbourhood`
  - 低 FPR 阈值由独立验证集选择，再应用到报告集

方法细节见：`artifacts/reports/method_notes.md`

## 3. Baseline 结果

完整总表见：

- `artifacts/reports/final_summary.md`
- `artifacts/reports/final_summary.csv`

当前最关键的 baseline 结果如下：

- `lora_standard`，3 seeds：
  - `train_loss = 2.638820 ± 0.003240`
  - `avg_exposure = 1.651849 ± 0.019638`
  - `mia_auc_loss = 0.510857 ± 0.000201`
  - `mia_auc_neighbourhood = 0.521102 ± 0.001191`
- `lora_no_canary`，3 seeds：
  - `train_loss = 2.221614 ± 0.004310`
  - `avg_exposure = 1.261029 ± 0.008580`
  - `mia_auc_loss = 0.508279 ± 0.000815`
  - `mia_auc_neighbourhood = 0.521339 ± 0.000056`
- `lora_dedup`，1 seed：
  - `avg_exposure = 1.664285`
  - `mia_auc_neighbourhood = 0.520696`
- `base_model`，1 seed：
  - `avg_exposure = 1.259471`
  - `mia_auc_neighbourhood = 0.501257`

当前观察：

- 全部 baseline 的**完整 canary 抽取成功率均为 0**
- 与 `lora_no_canary` 相比，`lora_standard` 的 **avg exposure** 显著更高
- `lora_dedup` 与 `lora_standard` 结果接近，符合当前训练集精确重复极少的现状
- `decode_safe` 作为推理阶段防护 baseline，同样保持 `0` 抽取成功
- 在独立验证集选阈的口径下，当前多数 baseline 的 `TPR@FPR=1e-3` 接近 `0`

## 4. Ablation 结果

因子总表见：

- `artifacts/reports/final_factor_summary.md`
- `artifacts/reports/final_factor_summary.csv`

### 4.1 主消融

- `num_canaries`
  - `50 -> 100 -> 200` 时，`avg_exposure` 呈 `2.069533 -> 1.900846 -> 1.660983`
- `canary_repeats`
  - `1 -> 3 -> 5` 时，`avg_exposure` 呈 `1.399037 -> 1.561922 -> 1.660983`
- `epochs`
  - `1 -> 2 -> 3` 时，`avg_exposure` 呈明显上升：`1.660983 -> 2.039426 -> 3.055287`
  - `mia_auc_neighbourhood` 同步抬升到 `0.538428`
- `lr`
  - `1e-4 -> 2e-4 -> 5e-4` 时，`avg_exposure` 呈 `1.570356 -> 1.660983 -> 1.786848`
- `lora_r`
  - `8/16/32` 三值间差异较小，当前信号强度有限

### 4.2 补充消融

- `max_len`
  - `256/512/1024` 三值差异很小，当前未观察到显著风险变化
- `lora_alpha`
  - 较大 `alpha` 对 exposure 与 MIA 有抬升趋势
- `lora_dropout`
  - `0.0/0.05/0.1` 差异较小
- `target_modules`
  - `qv` 相比 `qkvo` 显示出更低的 exposure 与 MIA
- `dedup`
  - `false/true` 差异极小
- `num_samples_per_prefix`
  - `20/50/100` 当前抽取成功率均为 `0`
- `temperature/top_p`
  - 当前三组采样配置下抽取成功率仍为 `0`
- `num_reference`
  - `500/1000/2000` 时，approx exposure 汇总值随参考规模变化
- `neigh_k/word_drop`
  - 更激进的邻域扰动设置抬高了 `mia_auc_neighbourhood`

## 5. 可视化与关键图表

总图与总表目录：

- `artifacts/reports/final_plots/final_overview.png`
- `artifacts/reports/final_plots/mia_tpr_compare.png`
- `artifacts/reports/final_plots/ablations/`

当前建议优先查看：

- `artifacts/reports/final_plots/final_overview.png`
  - baseline 与主要 aggregate label 的关键指标总览
- `artifacts/reports/final_plots/mia_tpr_compare.png`
  - 独立验证集选阈后的 `TPR@FPR` 对比
- `artifacts/reports/final_plots/ablations/epochs.png`
  - 训练 epoch 对 exposure 与 MIA 的抬升最明显
- `artifacts/reports/final_plots/ablations/num_canaries.png`
  - canary 数量对 exposure 的影响趋势清晰
- `artifacts/reports/final_plots/ablations/canary_repeats.png`
  - canary 重复次数对 exposure 的影响趋势清晰

## 6. 统计稳健性说明

- `lora_standard` 已完成 `3` 个 seed
- `lora_no_canary` 已完成 `3` 个 seed
- 其余 ablation 当前统一标注为**单次运行结果**
- 所有单次运行结果均保留原始实验目录，后续可继续扩展多 seed

## 7. 局限与解释边界

- 抽取攻击在当前设置下始终为 `0`，当前最强风险信号主要来自 exposure 与 neighbourhood MIA
- exposure 仍为 **approximate exposure**
- MIA 仍为 baseline MIA，未覆盖更强攻击族
- 单次 ablation 当前用于趋势审计与敏感性定位，不表述为稳定统计规律
- `decode_safe` 属于推理阶段防护，不表述为训练阶段隐私防护

## 8. 结果追溯

本轮全部正式结果可追溯到：

- registry：`configs/final_registry.yaml`
- 逐运行总表：`artifacts/reports/final_runs.csv`
- 聚合总表：`artifacts/reports/final_summary.csv`
- 因子总表：`artifacts/reports/final_factor_summary.csv`
- 方法说明：`artifacts/reports/method_notes.md`
- 图表目录：`artifacts/reports/final_plots/`
