#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/experiment.yaml}"
MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-0.5B-Instruct}"
CANARY_META="${CANARY_META:-data/processed/canaries.json}"
MEMBER_JSONL="${MEMBER_JSONL:-data/processed/train_no_canary.jsonl}"
NONMEMBER_JSONL="${NONMEMBER_JSONL:-data/raw/repaired/nonmember.jsonl}"

if ! "$PYTHON" -V >/dev/null 2>&1; then
  echo "[ERROR] 无法使用 Python 解释器: $PYTHON"
  exit 1
fi

if [[ -z "${LORA_DIR:-}" ]]; then
  if ls outputs/experiments/train/* >/dev/null 2>&1; then
    LORA_DIR="$(ls -1dt outputs/experiments/train/* | head -n1)"
  else
    echo "[ERROR] 未提供 LORA_DIR，且未找到历史训练目录"
    exit 1
  fi
fi

RUN_GROUP="${RUN_GROUP:-$(basename "$LORA_DIR")}"
ATTACK_DIR="outputs/experiments/attack_extract/${RUN_GROUP}"
EXPOSURE_DIR="outputs/experiments/exposure/${RUN_GROUP}"
MIA_DIR="outputs/experiments/mia/${RUN_GROUP}"
PLOT_DIR="outputs/experiments/plots/${RUN_GROUP}"

echo "使用 LoRA 目录: $LORA_DIR"

echo "[1/4] Attack: canary extraction"
"$PYTHON" -m src.attack_extract \
  --config "$CONFIG" \
  --model_name_or_path "$MODEL_PATH" \
  --lora_dir "$LORA_DIR" \
  --canary_meta "$CANARY_META" \
  --output_dir "$ATTACK_DIR"

echo "[2/4] Exposure (approx)"
"$PYTHON" -m src.exposure \
  --config "$CONFIG" \
  --model_name_or_path "$MODEL_PATH" \
  --lora_dir "$LORA_DIR" \
  --canary_meta "$CANARY_META" \
  --output_dir "$EXPOSURE_DIR"

echo "[3/4] MIA"
"$PYTHON" -m src.mia \
  --config "$CONFIG" \
  --model_name_or_path "$MODEL_PATH" \
  --lora_dir "$LORA_DIR" \
  --member_jsonl "$MEMBER_JSONL" \
  --nonmember_jsonl "$NONMEMBER_JSONL" \
  --output_dir "$MIA_DIR"

echo "[4/4] 可视化"
"$PYTHON" -m src.visualize_results \
  --exposure_run_dir "$EXPOSURE_DIR" \
  --mia_run_dir "$MIA_DIR" \
  --out_dir "$PLOT_DIR"

echo "[OK] 审计完成"
echo "  attack   -> $ATTACK_DIR"
echo "  exposure -> $EXPOSURE_DIR"
echo "  mia      -> $MIA_DIR"
echo "  plots    -> $PLOT_DIR"
