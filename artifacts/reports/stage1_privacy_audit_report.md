# 第一阶段隐私审计报告

更新时间：2026-03-27

## 1. 研究问题

当前阶段关注以下问题：

- 在 **Qwen2.5-0.5B-Instruct** 上执行 LoRA 微调后，canary 注入是否会提升隐私暴露风险
- 基础模型、标准 LoRA、无 canary LoRA 三类 baseline 的风险表现是否存在可复核差异
- 当前 baseline MIA 与 approximate exposure 能否形成一致风险信号

## 2. 数据、模型与方法定义

### 2.1 数据

- 正式训练输入：`data/raw/repaired/train.jsonl`，1206 条有效样本
- 正式非成员输入：`data/raw/repaired/nonmember.jsonl`，1193 条有效样本
- canary 配置：`num_canaries=200`、`canary_repeats=5`
- 处理后训练集：
  - `train_with_canary.jsonl`：2206 条
  - `train_no_canary.jsonl`：1206 条

### 2.2 模型

- 来源：**ModelScope**
- 仓库：`Qwen/Qwen2.5-0.5B-Instruct`
- 本地目录：`models/Qwen2.5-0.5B-Instruct`
- 下载时间：`2026-03-26T16:27:12.988660+00:00`

### 2.3 方法口径

- 抽取攻击：以 **完整 canary 字符串命中** 作为成功定义
- Exposure：当前实现为 **approximate exposure based on full-canary NLL ranking**
- MIA：当前实现为 **baseline MIA**
  - loss threshold
  - neighbourhood comparison
- MIA 成员集：`data/processed/train_no_canary.jsonl`
- 抽取攻击采样参数：
  - `num_samples_per_prefix=50`
  - `max_new_tokens=64`
  - `temperature=0.9`
  - `top_p=0.95`
  - `generation_batch_size=50`

## 3. baseline 结果

| baseline | n | seeds | train_loss | extraction_success_rate | avg_exposure | mia_auc_loss | mia_auc_neighbourhood |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `base_model` | 1 | `0` |  | `0.000000 ± 0.000000` | `1.259471 ± 0.000000` | `0.494200 ± 0.000000` | `0.499074 ± 0.000000` |
| `lora_dedup` | 1 | `0` | `2.638227 ± 0.000000` | `0.000000 ± 0.000000` | `1.664285 ± 0.000000` | `0.512577 ± 0.000000` | `0.515506 ± 0.000000` |
| `lora_no_canary` | 1 | `0` | `2.220447 ± 0.000000` | `0.000000 ± 0.000000` | `1.269517 ± 0.000000` | `0.509655 ± 0.000000` | `0.516543 ± 0.000000` |
| `lora_standard` | 3 | `0,1,2` | `2.638820 ± 0.003240` | `0.000000 ± 0.000000` | `1.651849 ± 0.019638` | `0.512287 ± 0.000307` | `0.515469 ± 0.000379` |

逐运行明细见：

- `artifacts/reports/baseline_runs.csv`
- `artifacts/reports/baseline_runs.md`

## 4. 当前结论

- 当前全部正式 baseline 在 **全量 10000 次抽取尝试** 下均未命中完整 canary
- **lora_standard** 的平均 exposure 相比 `base_model` 与 `lora_no_canary` 明显升高，当前最强风险信号来自 exposure
- `lora_dedup` 与 `lora_standard` 的指标几乎一致，当前与训练集不存在精确重复样本这一事实一致
- `lora_no_canary` 的 exposure 与 `base_model` 接近，说明当前风险提升主要与 canary 注入有关
- MIA 的 AUC 结果整体集中在 `0.49` 到 `0.516` 区间，现阶段仅能支持 **弱信号** 结论

## 5. 图表与结果定位

- baseline 汇总表：`artifacts/reports/baseline_summary.md`
- avg exposure 图：`artifacts/reports/baseline_plots/avg_exposure.png`
- extraction success rate 图：`artifacts/reports/baseline_plots/extraction_success_rate.png`
- MIA AUC loss 图：`artifacts/reports/baseline_plots/mia_auc_loss.png`
- MIA AUC neighbourhood 图：`artifacts/reports/baseline_plots/mia_auc_neighbourhood.png`

## 6. 局限性

- `base_model` 与 `lora_no_canary` 当前仅完成 1 个 seed，统计稳健性仍有限
- `lora_dedup` 当前仅完成 1 个 seed
- 当前 exposure 口径为 **approximate exposure**，尚未宣称论文级严格复现
- 当前 MIA 为 baseline 实现，未覆盖更强攻击族
- 当前 `num_canaries` 主消融已完成 `50/100/200` 的单 seed 对照，仍缺少更多 seed

## 7. 下一步

- 扩展 `num_canaries` 主消融到更多 seed
- 启动下一组主消融，例如 `canary_repeats`
- 扩展多 seed 统计覆盖范围
- 继续累积可复核图表、总表与方法说明

补充说明：

- `num_canaries=50` 对比 `num_canaries=200` 的首轮记录已写入 `artifacts/reports/ablation_num_canaries50_vs200.md`
- `num_canaries=50/100/200` 的单 seed 三值对照已写入 `artifacts/reports/ablation_num_canaries_50_100_200.md`
