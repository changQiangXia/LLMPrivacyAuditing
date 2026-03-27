#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/experiment.yaml}"
MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-0.5B-Instruct}"
TRAIN_JSONL="${TRAIN_JSONL:-data/processed/train_with_canary.jsonl}"
MEMBER_JSONL="${MEMBER_JSONL:-data/processed/train_no_canary.jsonl}"
NONMEMBER_JSONL="${NONMEMBER_JSONL:-data/raw/repaired/nonmember.jsonl}"
CANARY_META="${CANARY_META:-data/processed/canaries.json}"
RUN_GROUP="${RUN_GROUP:-$(date -u +%Y%m%d_%H%M%S)}"

TRAIN_DIR="outputs/experiments/train/${RUN_GROUP}"
ATTACK_DIR="outputs/experiments/attack_extract/${RUN_GROUP}"
EXPOSURE_DIR="outputs/experiments/exposure/${RUN_GROUP}"
MIA_DIR="outputs/experiments/mia/${RUN_GROUP}"
PLOT_DIR="outputs/experiments/plots/${RUN_GROUP}"

if ! "$PYTHON" -V >/dev/null 2>&1; then
  echo "[ERROR] 无法使用 Python 解释器: $PYTHON"
  exit 1
fi

echo "========== [0/5] 环境自检 =========="
"$PYTHON" -m src.env_check

echo ""
echo "========== [1/5] LoRA 训练 =========="
"$PYTHON" -m src.train_lora \
  --config "$CONFIG" \
  --model_name_or_path "$MODEL_PATH" \
  --train_jsonl "$TRAIN_JSONL" \
  --output_dir "$TRAIN_DIR"

echo ""
echo "========== [2/5] Attack: canary extraction =========="
"$PYTHON" -m src.attack_extract \
  --config "$CONFIG" \
  --model_name_or_path "$MODEL_PATH" \
  --lora_dir "$TRAIN_DIR" \
  --canary_meta "$CANARY_META" \
  --output_dir "$ATTACK_DIR"

echo ""
echo "========== [3/5] Exposure (approx) =========="
"$PYTHON" -m src.exposure \
  --config "$CONFIG" \
  --model_name_or_path "$MODEL_PATH" \
  --lora_dir "$TRAIN_DIR" \
  --canary_meta "$CANARY_META" \
  --output_dir "$EXPOSURE_DIR"

echo ""
echo "========== [4/5] MIA =========="
"$PYTHON" -m src.mia \
  --config "$CONFIG" \
  --model_name_or_path "$MODEL_PATH" \
  --lora_dir "$TRAIN_DIR" \
  --member_jsonl "$MEMBER_JSONL" \
  --nonmember_jsonl "$NONMEMBER_JSONL" \
  --output_dir "$MIA_DIR"

echo ""
echo "========== [5/5] 可视化 =========="
"$PYTHON" -m src.visualize_results \
  --train_run_dir "$TRAIN_DIR" \
  --exposure_run_dir "$EXPOSURE_DIR" \
  --mia_run_dir "$MIA_DIR" \
  --out_dir "$PLOT_DIR"

echo ""
echo "[OK] 全流程完成"
echo "  train    -> $TRAIN_DIR"
echo "  attack   -> $ATTACK_DIR"
echo "  exposure -> $EXPOSURE_DIR"
echo "  mia      -> $MIA_DIR"
echo "  plots    -> $PLOT_DIR"
