# 方法说明文档

更新时间：2026-03-27

## 1. 文档目的

该文档用于固定当前 **LLM Privacy Audit** 项目的方法口径、实验边界与结果追溯规则。所有表格、图像与结论均以本文件、`configs/final_registry.yaml`、对应实验目录中的 `effective_config.yaml`、`metadata.json`、`metrics.json`、`results.json` 为准。

## 2. 研究问题

当前阶段关注以下问题：

- 在 **Qwen2.5-0.5B-Instruct** 上进行 LoRA 微调后，训练语料中注入 canary 是否会提升隐私暴露风险
- 不同 baseline 设计下，canary 抽取攻击、approximate exposure 与 baseline MIA 是否呈现一致信号
- 训练超参数、canary 注入策略、生成参数、exposure 参考规模、MIA 邻域参数变化时，风险指标如何变化
- 推理阶段的 **decode-safe** 防护能否降低直接抽取风险

## 3. 威胁模型与评估边界

- 审计对象为基础模型与其 LoRA 适配器
- 攻击者可访问模型生成接口，并可设置采样参数
- 攻击者不能访问训练梯度、优化器状态或完整训练语料
- 当前抽取攻击以 **prefix completion** 形式进行
- 当前 exposure 为 **approximate exposure based on full-canary NLL ranking**
- 当前 MIA 为 **baseline MIA**，包含 loss threshold 与 neighbourhood comparison 两类攻击
- 当前结论适用于本地单卡 **RTX 4090 24GB**、当前数据规模、当前 LoRA 配置与当前评估实现，不外推到更大模型或更强攻击族

## 4. 数据与可追溯性

### 4.1 正式原始输入

- 成员原始输入：`data/raw/repaired/train.jsonl`
- 非成员原始输入：`data/raw/repaired/nonmember.jsonl`
- 原始 JSONL 的末行截断问题已通过 `src/repair_jsonl.py` 脚本化修复

### 4.2 正式处理后数据

- 标准训练集：`data/processed/train_with_canary.jsonl`
- 无 canary 训练集：`data/processed/train_no_canary.jsonl`
- dedup 训练集：`data/processed/train_dedup_with_canary.jsonl`
- canary 元数据：`data/processed/canaries.json`

### 4.3 数据处理规则

- 处理入口：`src/data_prep.py`
- 每次数据处理均保存：
  - `*.manifest.json`
  - `*.validation.json`
  - `*.log`
- manifest 至少记录输入路径、输出路径、样本数、dedup 开关、canary 数量、重复次数与 seed
- validation 至少验证输出行数、有效 text 样本数与 canary 插入位置有效性

### 4.4 MIA 数据划分

- 当前 MIA 默认以 `data/processed/train_no_canary.jsonl` 为 member 集
- 当前 MIA 默认以 `data/raw/repaired/nonmember.jsonl` 为 nonmember 集
- 当前实现支持从 member 与 nonmember 中按 `validation_fraction` 划出独立阈值选择集
- 低 FPR 阈值选择在验证集上完成，报告集仅用于汇报阈值应用后的 TPR/FPR

## 5. 模型与训练配置

### 5.1 基础模型

- 来源：**ModelScope**
- 仓库：`Qwen/Qwen2.5-0.5B-Instruct`
- 本地目录：`models/Qwen2.5-0.5B-Instruct`
- 下载脚本：`src/download_model.py`

### 5.2 LoRA 训练

- 训练入口：`src/train_lora.py`
- 默认训练配置：`configs/train_lora.yaml`
- 当前 baseline 标准配置：
  - `epochs=1`
  - `batch_size=2`
  - `grad_accum=8`
  - `lr=2e-4`
  - `max_len=512`
  - `lora_r=16`
  - `lora_alpha=32`
  - `lora_dropout=0.05`
  - `target_modules=q_proj,k_proj,v_proj,o_proj`

### 5.3 训练严谨性

- 本地模型目录场景强制使用离线路径加载
- 训练前校验训练文件与输出目录
- 训练期间记录 `train.log`、`train_history.json`、`metrics.json`
- 检测到 `NaN` 或 `Inf` loss 时直接终止
- 训练结束后自动执行 checkpoint 可加载性验证

## 6. Baseline 设计

当前 baseline 包括以下类型：

- **base_model**
  - 不进行 LoRA 微调，直接执行 attack、exposure、MIA
- **lora_standard**
  - 使用注入 canary 的标准训练集进行 LoRA 微调
- **lora_no_canary**
  - 使用不注入 canary 的训练集进行 LoRA 微调
- **lora_dedup**
  - 对训练文本执行精确字符串 dedup 后再注入 canary 并训练
- **decode_safe**
  - 推理阶段防护 baseline
  - 当前实现由两部分组成：
    - `repetition_penalty=1.1`
    - 对生成文本执行 `redact_pii`
  - 该 baseline 仅改变生成输出与抽取攻击评估口径，不改变模型参数

## 7. Ablation 设计

### 7.1 主消融

- `num_canaries`
- `canary_repeats`
- `epochs`
- `lr`
- `lora_r`

### 7.2 补充消融

- `max_len`
- `lora_alpha`
- `lora_dropout`
- `target_modules`
- `dedup`
- `num_samples_per_prefix`
- `temperature` / `top_p`
- `num_reference`
- `neigh_k` / `word_drop`

### 7.3 因子影响范围

- 训练级因子：
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
- 生成攻击级因子：
  - `num_samples_per_prefix`
  - `temperature`
  - `top_p`
- exposure 级因子：
  - `num_reference`
- MIA 级因子：
  - `neigh_k`
  - `word_drop`

## 8. 指标定义

### 8.1 Canary 抽取攻击

- 脚本：`src/attack_extract.py`
- 成功定义：**完整 canary 字符串命中**
- 正式指标：`canary_extraction_success_rate = hit / total`
- 正式保存：
  - `metrics.json`
  - `results.json`
  - `sample_hits.jsonl`

### 8.2 Approximate Exposure

- 脚本：`src/exposure.py`
- 当前定义：**approximate exposure based on full-canary NLL ranking**
- 公式：
  - `approx_exposure = log2(num_reference + 1) - log2(rank)`
- 参考 canary 生成分布：
  - 使用 `src/canary.py` 中的 `make_canaries(num_reference, seed + 999)` 生成
  - 参考样本与目标 canary 使用同一模板 `NAME | EMAIL | PHONE | CODE`
  - `name`、`email`、`phone`、`code` 字段均由固定随机生成规则采样
  - 当前 exposure 口径可解释为：将目标 canary 的 NLL 与同模板随机参考 canary 的 NLL 排名进行比较
- 正式保存：
  - 每个 canary 的 `target_nll`
  - `rank`
  - `approx_exposure`

### 8.3 Membership Inference Attack

- 脚本：`src/mia.py`
- 当前包含两类 baseline 攻击：
  - `loss_threshold`
  - `neighbourhood`
- AUC 与 ROC 仍在报告集中计算
- 当报告 `TPR@FPR` 的阈值选择口径时，当前优先使用独立验证集：
  - 验证集选择阈值
  - 报告集评估该阈值下的 `TPR/FPR`
- 指标计算正确性由 `tests/test_metrics.py` 覆盖：
  - ROC 端点
  - AUC
  - `TPR@FPR`
  - 低 FPR 阈值选择
  - 固定阈值下的混淆矩阵统计
- 正式保存：
  - `metrics.json`
  - `results.json`
  - `sample_scores.jsonl`

## 9. 统计汇报规则

- 关键 baseline 结果优先使用多 seed 汇总
- 汇总表统一报告：
  - 均值
  - 标准差
  - 样本量 `n`
- 单 seed 结果在文档中统一标注为 **单次运行结果**
- 失败运行、超时目录与探针目录保留，以便后续核查

## 10. 输出目录约定

- 训练：`outputs/experiments/train/<run_id>/`
- 抽取攻击：`outputs/experiments/attack_extract/<run_id>/`
- exposure：`outputs/experiments/exposure/<run_id>/`
- MIA：`outputs/experiments/mia/<run_id>/`
- 汇总图表：`artifacts/reports/final_plots/`
- 全量 registry：`configs/final_registry.yaml`

## 11. 总表、总图与结果汇总

- 汇总脚本：`src/build_final_report.py`
- 输入：`configs/final_registry.yaml`
- 输出：
  - `artifacts/reports/final_runs.csv`
  - `artifacts/reports/final_runs.md`
  - `artifacts/reports/final_summary.csv`
  - `artifacts/reports/final_summary.md`
  - `artifacts/reports/final_factor_summary.csv`
  - `artifacts/reports/final_factor_summary.md`
  - `artifacts/reports/final_plots/final_overview.png`
  - `artifacts/reports/final_plots/mia_tpr_compare.png`
  - `artifacts/reports/final_plots/ablations/*.png`

## 12. 当前边界与限制

- exposure 仍为 **approximate exposure**，当前未表述为论文级严格复现
- MIA 仍属于 baseline MIA，当前未覆盖更强白盒或更复杂黑盒攻击
- 当前多数实验在验证集选阈后的 `TPR@FPR=1e-3` 接近 `0`，低 FPR 区间的 membership 泄露信号整体偏弱
- `decode_safe` 属于推理阶段防护，当前不表述为训练期隐私防护
- 单 seed ablation 当前用于趋势审计与敏感性定位，后续仍可继续扩展多 seed
