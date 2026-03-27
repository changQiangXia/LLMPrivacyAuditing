# 项目进度记录

更新时间：2026-03-27

## 2026-03-27 基础环境与工程闭环

### 已确认事实

- 工作空间为 `/root/autodl-tmp`
- 基础环境为 Python 3.10.8、torch 2.1.2+cu121、CUDA 12.1
- 计算资源为单卡 **NVIDIA GeForce RTX 4090 24GB**
- 基础模型已统一为 **ModelScope** 来源：
  - 仓库：`Qwen/Qwen2.5-0.5B-Instruct`
  - 本地目录：`models/Qwen2.5-0.5B-Instruct`
  - 下载时间：`2026-03-26T16:27:12.988660+00:00`
- 环境快照已保存到：
  - `artifacts/env/pip_freeze.txt`
  - `artifacts/env/env_check.json`
  - `docs/environment.md`

### 依赖与兼容性处理

- 初始问题：`transformers 4.57.6` 与 `torch 2.1.2+cu121` 组合不兼容，`pytest` 收集阶段触发 `torch.utils._pytree` 相关错误
- 已固定版本：
  - `transformers 4.41.2`
  - `peft 0.11.1`
  - `datasets 2.19.2`
  - `accelerate 0.31.0`
- 当前依赖安装说明与实际命令已写入 `docs/dependency_install.md`

### 工程基础设施改造

- 已写入 `instructions.md`，固化项目目标、写作要求、资源约束与执行原则
- 已新增与重构以下核心模块：
  - `src/config_utils.py`
  - `src/experiment.py`
  - `src/metrics.py`
  - `src/validation.py`
  - `src/data_prep.py`
  - `src/train_lora.py`
  - `src/attack_extract.py`
  - `src/exposure.py`
  - `src/mia.py`
  - `src/download_model.py`
  - `src/env_check.py`
  - `src/visualize_results.py`
  - `src/repair_jsonl.py`
  - `src/aggregate_results.py`
- 已新增测试：
  - `tests/test_config_utils.py`
  - `tests/test_metrics.py`
  - `tests/test_validation.py`
  - `tests/test_data_prep.py`
  - `tests/test_pipeline_smoke.py`
  - `tests/test_aggregate_results.py`
  - `tests/test_cli_config.py`
  - `tests/test_exposure.py`
  - `tests/test_utils.py`

## 2026-03-27 数据治理与一致性修复

### 原始数据修复

- `data/raw/train.jsonl`、`data/raw/nonmember.jsonl`、`data/raw/dolly.jsonl`、`data/raw/all_text.jsonl` 的最后一行均存在截断问题
- 修复逻辑已脚本化，正式输入切换到 `data/raw/repaired/`
- 当前修复结果：
  - `data/raw/repaired/train.jsonl`：1206 行，有效行 1206，损坏行 1
  - `data/raw/repaired/nonmember.jsonl`：1193 行，有效行 1193，损坏行 1
  - `data/raw/repaired/dolly.jsonl`：1502 行，有效行 1502，损坏行 1
  - `data/raw/repaired/all_text.jsonl`：1559 行，有效行 1559，损坏行 1
- 每份修复结果均已保留：
  - `*.repair.log`
  - `*.repair_report.json`

### 处理后数据重建

- `data/processed/` 已由修复后的训练集重新生成
- 当前正式文件与 manifest 如下：
  - `data/processed/train_with_canary.jsonl`：2206 行
  - `data/processed/train_no_canary.jsonl`：1206 行
  - `data/processed/train_dedup_with_canary.jsonl`：2206 行
  - `data/processed/canaries.json`：`num_canaries=200`、`repeats=5`、`positions=1000`
- `data/processed/canaries.json` 中插入位置范围已验证为 `[0, 2205]`
- manifest 与 validation 文件已齐备：
  - `data/processed/train_with_canary.manifest.json`
  - `data/processed/train_with_canary.validation.json`
  - `data/processed/train_no_canary.manifest.json`
  - `data/processed/train_no_canary.validation.json`
  - `data/processed/train_dedup_with_canary.manifest.json`
  - `data/processed/train_dedup_with_canary.validation.json`

## 2026-03-27 配置链路与本地模型加载修复

### 配置链路问题

- 新发现问题：`argparse` 默认值覆盖了 YAML 默认值，导致 `configs/train_lora.yaml` 与 `configs/experiment.yaml` 中的部分配置未真正生效
- 影响范围：
  - `src/train_lora.py`
  - `src/attack_extract.py`
  - `src/exposure.py`
  - `src/mia.py`
- 处理结果：
  - 改为显式读取配置默认值并回填到 CLI 参数
  - 对关键字段改为解析后校验，允许 `--config` 单独提供必填项
  - 新增 `tests/test_cli_config.py` 覆盖配置默认值接入

### 本地模型加载验证

- 历史训练日志曾出现 Hugging Face HEAD 重试告警
- 已在 `src/utils.py` 与 `src/train_lora.py` 接入 `local_files_only`
- 已进一步完成两项修复：
  - 本地模型路径统一规范为绝对路径
  - 本地模型场景强制开启 `HF_HUB_OFFLINE=1` 与 `TRANSFORMERS_OFFLINE=1`
- 已将 `train_lora` 的 adapter 保存逻辑改为显式控制 `save_embedding_layers`
- 已新增 `LocalAwareTrainer`，将中途 checkpoint 保存也切换到本地离线路径
- 已对以下场景做本地加载验证：
  - 仅基础模型加载
  - 基础模型 + LoRA 目录加载
  - 微型训练 + 保存 + checkpoint 可加载性验证
  - 微型训练 + `save_steps=10` 的中途 checkpoint 保存验证
- 当前验证中已不再出现外部 HEAD 重试告警与本地配置探测告警

## 2026-03-27 baseline 结果收敛

### 已完成正式实验

- `base_model`
  - attack：`outputs/experiments/attack_extract/base_model_full`
  - exposure：`outputs/experiments/exposure/base_model_full`
  - mia：`outputs/experiments/mia/base_model_full`
- `lora_standard seed0`
  - train：`outputs/experiments/train/baseline_standard_seed0`
  - attack：`outputs/experiments/attack_extract/baseline_standard_seed0_full`
  - exposure：`outputs/experiments/exposure/baseline_standard_seed0_full`
  - mia：`outputs/experiments/mia/baseline_standard_seed0_full`
- `lora_standard seed1`
  - train：`outputs/experiments/train/baseline_standard_seed1`
  - attack：`outputs/experiments/attack_extract/baseline_standard_seed1_full`
  - exposure：`outputs/experiments/exposure/baseline_standard_seed1_full`
  - mia：`outputs/experiments/mia/baseline_standard_seed1_full`
- `lora_standard seed2`
  - train：`outputs/experiments/train/baseline_standard_seed2`
  - attack：`outputs/experiments/attack_extract/baseline_standard_seed2_full`
  - exposure：`outputs/experiments/exposure/baseline_standard_seed2_full`
  - mia：`outputs/experiments/mia/baseline_standard_seed2_full`
- `lora_no_canary seed0`
  - train：`outputs/experiments/train/baseline_no_canary_seed0`
  - attack：`outputs/experiments/attack_extract/baseline_no_canary_seed0_full`
  - exposure：`outputs/experiments/exposure/baseline_no_canary_seed0_full`
  - mia：`outputs/experiments/mia/baseline_no_canary_seed0_full`
- `lora_dedup seed0`
  - train：`outputs/experiments/train/baseline_dedup_seed0`
  - attack：`outputs/experiments/attack_extract/baseline_dedup_seed0_full`
  - exposure：`outputs/experiments/exposure/baseline_dedup_seed0_full`
  - mia：`outputs/experiments/mia/baseline_dedup_seed0_full`

### 运行中风险与处理

- 抽取攻击初始默认 `generation_batch_size=10`，全量 200 个 canary 运行时间约为半小时级别
- 已做 1 个 canary 的提速基准测试，`generation_batch_size=50` 可稳定运行
- `configs/experiment.yaml` 已更新为 `generation_batch_size=50`
- 该调整后，`lora_standard seed1` 与 `seed2` 的全量抽取攻击均已完成
- 超时遗留目录已保留为：
  - `outputs/experiments/attack_extract/baseline_standard_seed1_full_timeout_partial`
  - 用途：记录默认 batch 设置下的失败尝试

### 当前正式结果

- `base_model`
  - extraction success rate：`0.0`
  - avg exposure：`1.2594712206491605`
  - MIA AUC loss：`0.49419985848905834`
  - MIA AUC neighbourhood：`0.49907420149879267`
- `lora_no_canary seed0`
  - train_loss：`2.2204467264811196`
  - extraction success rate：`0.0`
  - avg exposure：`1.2695170316113356`
  - MIA AUC loss：`0.5096548550902931`
  - MIA AUC neighbourhood：`0.5165427403357616`
- `lora_dedup seed0`
  - train_loss：`2.638227421001796`
  - extraction success rate：`0.0`
  - avg exposure：`1.6642854344629334`
  - MIA AUC loss：`0.5125768197292385`
  - MIA AUC neighbourhood：`0.5155064298513027`
- `lora_standard` 三个训练 seed 统计
  - train_loss：`2.638820 ± 0.003240`
  - extraction success rate：`0.000000 ± 0.000000`
  - avg exposure：`1.651849 ± 0.019638`
  - MIA AUC loss：`0.512287 ± 0.000307`
  - MIA AUC neighbourhood：`0.515469 ± 0.000379`

### 阶段性结论

- 当前所有正式 baseline 的全量抽取攻击均为 `0/10000`
- **lora_standard** 的平均 exposure 明显高于 `base_model` 与 `lora_no_canary`
- `lora_dedup seed0` 与 `lora_standard` 的各项指标几乎重合，和 `train_dedup_with_canary` 中未发现精确重复样本这一事实一致
- MIA 指标整体接近随机猜测区间，当前证据强度偏弱

### 已生成报告与图表

- 分组汇总：
  - `artifacts/reports/baseline_summary.csv`
  - `artifacts/reports/baseline_summary.md`
- 逐运行明细：
  - `artifacts/reports/baseline_runs.csv`
  - `artifacts/reports/baseline_runs.md`
- 图表：
  - `artifacts/reports/baseline_plots/avg_exposure.png`
  - `artifacts/reports/baseline_plots/extraction_success_rate.png`
  - `artifacts/reports/baseline_plots/mia_auc_loss.png`
  - `artifacts/reports/baseline_plots/mia_auc_neighbourhood.png`
- 阶段性说明报告：
  - `artifacts/reports/stage1_privacy_audit_report.md`
- 主消融对照：
  - `artifacts/reports/ablation_num_canaries50_vs200.md`
  - `artifacts/reports/ablation_num_canaries_50_100_200.md`

## 2026-03-27 主消融进展：`num_canaries=50/100/200`

### 已完成实验

- `num_canaries=50`
  - 数据：`data/processed/train_with_canary_num50.jsonl`
  - canary：`data/processed/canaries_num50.json`
  - 训练：`outputs/experiments/train/ablation_num_canaries50_seed0`
  - attack：`outputs/experiments/attack_extract/ablation_num_canaries50_seed0_full`
  - exposure：`outputs/experiments/exposure/ablation_num_canaries50_seed0_full`
  - mia：`outputs/experiments/mia/ablation_num_canaries50_seed0_full`
- `num_canaries=100`
  - 数据：`data/processed/train_with_canary_num100.jsonl`
  - canary：`data/processed/canaries_num100.json`
  - 训练：`outputs/experiments/train/ablation_num_canaries100_seed0`
  - attack：`outputs/experiments/attack_extract/ablation_num_canaries100_seed0_full`
  - exposure：`outputs/experiments/exposure/ablation_num_canaries100_seed0_full`
  - mia：`outputs/experiments/mia/ablation_num_canaries100_seed0_full`
- `num_canaries=200`
  - 数据：`data/processed/train_with_canary.jsonl`
  - canary：`data/processed/canaries.json`
  - 训练：`outputs/experiments/train/baseline_standard_seed0`
  - attack：`outputs/experiments/attack_extract/baseline_standard_seed0_full`
  - exposure：`outputs/experiments/exposure/baseline_standard_seed0_full`
  - mia：`outputs/experiments/mia/baseline_standard_seed0_full`

### 三值对照结果

- `num_canaries=50`
  - train_loss：`2.3736726687504697`
  - extraction success rate：`0.0`，对应 `0/2500`
  - avg exposure：`2.0695333058228114`
  - MIA AUC loss：`0.5101775281180013`
  - MIA AUC neighbourhood：`0.5165945211077888`
- `num_canaries=100`
  - train_loss：`2.4773752464438386`
  - extraction success rate：`0.0`，对应 `0/5000`
  - avg exposure：`1.9008463671733251`
  - MIA AUC loss：`0.511355280040146`
  - MIA AUC neighbourhood：`0.5165114633593695`
- `num_canaries=200`
  - train_loss：`2.6381317194360885`
  - extraction success rate：`0.0`，对应 `0/10000`
  - avg exposure：`1.6609825235608724`
  - MIA AUC loss：`0.512527471610932`
  - MIA AUC neighbourhood：`0.5155175505540195`

### 当前观察

- 当前单次运行下，avg exposure 呈现 `50 > 100 > 200` 的顺序
- 三组配置的抽取攻击均未命中完整 canary
- MIA 指标变化幅度较小，仍处于弱信号区间
- 上述现象当前仅覆盖 **单个训练 seed**，当前仅作为主消融趋势记录

### 已生成记录

- `artifacts/reports/ablation_num_canaries50_vs200.md`
- `artifacts/reports/ablation_num_canaries_50_100_200.md`

## 2026-03-27 completion suite、总表总图与方法文档收口

### 已完成批量实验与汇总

- `run_completion_suite.sh` 已完整结束
  - 日志：`logs/completion_suite_resume_20260327_020540.log`
  - 结束标记：`========== completion suite done ==========`
- `run_final_report.sh` 已完成最终汇总生成
  - registry：`configs/final_registry.yaml`
  - 总表明细：`artifacts/reports/final_runs.csv`
  - 总表聚合：`artifacts/reports/final_summary.csv`
  - 因子总表：`artifacts/reports/final_factor_summary.csv`
  - 方法说明：`artifacts/reports/method_notes.md`
  - 最终审计报告：`artifacts/reports/final_audit_report.md`

### 本轮补齐的 baseline 与 ablation

- baseline 已覆盖：
  - `base_model`
  - `lora_standard`
  - `lora_no_canary`
  - `lora_dedup`
  - `decode_safe`
- ablation 已覆盖：
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

### MIA 口径修正与统计稳健性

- MIA 新增独立验证集选阈流程：
  - `validation_fraction=0.2`
  - `max_validation_samples=500`
- 低 FPR 阈值统一在验证集上选取，再在报告集汇报 `TPR/FPR`
- `tests/test_metrics.py` 已覆盖：
  - ROC 端点
  - AUC
  - `TPR@FPR`
  - 阈值选择
  - 阈值应用后的混淆矩阵统计
- 关键实验当前已满足多 seed 汇总：
  - `lora_standard`：`3` 个 seed
  - `lora_no_canary`：`3` 个 seed

### 关键结果摘要

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
- `decode_safe`：
  - 抽取成功率仍为 `0`
- 当前主要趋势：
  - `canary_repeats` 增大时，`avg_exposure` 单调上升
  - `epochs` 从 `1` 增加到 `3` 时，exposure 与 neighbourhood MIA 明显抬升
  - `lr` 增大时，exposure 与 MIA 呈上升趋势
  - `target_modules=qv` 的风险低于 `qkvo`
  - `num_samples_per_prefix` 与 `temperature/top_p` 在当前范围内未带来完整抽取成功

### 最终图表归档

- 总图：
  - `artifacts/reports/final_plots/final_overview.png`
  - `artifacts/reports/final_plots/mia_tpr_compare.png`
- 图表原始数据同目录归档：
  - `artifacts/reports/final_plots/group_metrics.csv`
  - `artifacts/reports/final_plots/factor_metrics.csv`
  - `artifacts/reports/final_plots/ablations/*.csv`
- 当前已按实际产物同步更新 `Checklist.md` 中与 **baseline 设计**、**ablation 设计**、**MIA**、**统计稳健性**、**可视化**、**最低可接受交付物** 相关的完成状态

### 当前验证状态

- `pytest -q` 已通过：**20 passed**
- `ruff check .` 已通过
- `python -m compileall src` 已通过
