#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-0.5B-Instruct}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_lora.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-configs/experiment.yaml}"
TRAIN_CANARY_JSONL="${TRAIN_CANARY_JSONL:-data/processed/train_with_canary.jsonl}"
TRAIN_NO_CANARY_JSONL="${TRAIN_NO_CANARY_JSONL:-data/processed/train_no_canary.jsonl}"
NONMEMBER_JSONL="${NONMEMBER_JSONL:-data/raw/repaired/nonmember.jsonl}"
CANARY_META="${CANARY_META:-data/processed/canaries.json}"

dir_non_empty() {
  local path="$1"
  [[ -d "$path" ]] && [[ -n "$(ls -A "$path" 2>/dev/null)" ]]
}

run_dir_done() {
  local stage="$1"
  local path="$2"
  case "$stage" in
    train)
      [[ -f "$path/metrics.json" ]] && [[ -f "$path/adapter_model.safetensors" ]]
      ;;
    attack|exposure|mia)
      [[ -f "$path/metrics.json" ]] && [[ -f "$path/results.json" ]]
      ;;
    *)
      return 1
      ;;
  esac
}

run_train() {
  local run_id="$1"
  local train_jsonl="$2"
  shift 2
  local train_dir="outputs/experiments/train/${run_id}"
  if run_dir_done train "$train_dir"; then
    echo "[SKIP][train] $train_dir"
    return
  fi
  echo "[RUN ][train] $train_dir"
  "$PYTHON" -m src.train_lora \
    --config "$TRAIN_CONFIG" \
    --model_name_or_path "$MODEL_PATH" \
    --train_jsonl "$train_jsonl" \
    --output_dir "$train_dir" \
    "$@"
}

run_attack() {
  local run_id="$1"
  local lora_dir="$2"
  local canary_meta="$3"
  shift 3
  local out_dir="outputs/experiments/attack_extract/${run_id}"
  if run_dir_done attack "$out_dir"; then
    echo "[SKIP][attack] $out_dir"
    return
  fi
  echo "[RUN ][attack] $out_dir"
  local cmd=(
    "$PYTHON" -m src.attack_extract
    --config "$EVAL_CONFIG"
    --model_name_or_path "$MODEL_PATH"
    --canary_meta "$canary_meta"
    --output_dir "$out_dir"
  )
  if [[ -n "$lora_dir" ]]; then
    cmd+=(--lora_dir "$lora_dir")
  fi
  cmd+=("$@")
  "${cmd[@]}"
}

run_exposure() {
  local run_id="$1"
  local lora_dir="$2"
  local canary_meta="$3"
  shift 3
  local out_dir="outputs/experiments/exposure/${run_id}"
  if run_dir_done exposure "$out_dir"; then
    echo "[SKIP][exposure] $out_dir"
    return
  fi
  echo "[RUN ][exposure] $out_dir"
  local cmd=(
    "$PYTHON" -m src.exposure
    --config "$EVAL_CONFIG"
    --model_name_or_path "$MODEL_PATH"
    --canary_meta "$canary_meta"
    --output_dir "$out_dir"
  )
  if [[ -n "$lora_dir" ]]; then
    cmd+=(--lora_dir "$lora_dir")
  fi
  cmd+=("$@")
  "${cmd[@]}"
}

run_mia() {
  local run_id="$1"
  local lora_dir="$2"
  shift 2
  local out_dir="outputs/experiments/mia/${run_id}"
  if run_dir_done mia "$out_dir"; then
    echo "[SKIP][mia] $out_dir"
    return
  fi
  echo "[RUN ][mia] $out_dir"
  local cmd=(
    "$PYTHON" -m src.mia
    --config "$EVAL_CONFIG"
    --model_name_or_path "$MODEL_PATH"
    --member_jsonl "$TRAIN_NO_CANARY_JSONL"
    --nonmember_jsonl "$NONMEMBER_JSONL"
    --output_dir "$out_dir"
  )
  if [[ -n "$lora_dir" ]]; then
    cmd+=(--lora_dir "$lora_dir")
  fi
  cmd+=("$@")
  "${cmd[@]}"
}

run_full() {
  local run_id="$1"
  local train_jsonl="$2"
  local canary_meta="$3"
  shift 3
  local train_dir="outputs/experiments/train/${run_id}"
  run_train "$run_id" "$train_jsonl" "$@"
  run_attack "${run_id}_full" "$train_dir" "$canary_meta"
  run_exposure "${run_id}_full" "$train_dir" "$canary_meta"
  run_mia "${run_id}_full_valsplit" "$train_dir"
}

prepare_canary_data() {
  local out_jsonl="$1"
  local canary_meta="$2"
  shift 2
  local stem="${out_jsonl%.jsonl}"
  local manifest="${stem}.manifest.json"
  local validation="${stem}.validation.json"
  if [[ -f "$out_jsonl" && -f "$manifest" && -f "$validation" && -f "$canary_meta" ]]; then
    echo "[SKIP][data] $out_jsonl"
    return
  fi
  echo "[RUN ][data] $out_jsonl"
  "$PYTHON" -m src.data_prep \
    --in_jsonl data/raw/repaired/train.jsonl \
    --out_jsonl "$out_jsonl" \
    --insert_canary \
    --canary_meta_out "$canary_meta" \
    "$@"
}

echo "========== completion suite begin =========="

run_attack \
  "baseline_decode_safe_seed0_full" \
  "outputs/experiments/train/baseline_standard_seed0" \
  "$CANARY_META" \
  --repetition_penalty 1.1 \
  --decode_redact

run_mia "base_model_full_valsplit" ""
run_mia "baseline_standard_seed0_full_valsplit" "outputs/experiments/train/baseline_standard_seed0"
run_mia "baseline_standard_seed1_full_valsplit" "outputs/experiments/train/baseline_standard_seed1"
run_mia "baseline_standard_seed2_full_valsplit" "outputs/experiments/train/baseline_standard_seed2"
run_mia "baseline_no_canary_seed0_full_valsplit" "outputs/experiments/train/baseline_no_canary_seed0"
run_mia "baseline_dedup_seed0_full_valsplit" "outputs/experiments/train/baseline_dedup_seed0"
run_mia "ablation_num_canaries50_seed0_full_valsplit" "outputs/experiments/train/ablation_num_canaries50_seed0"
run_mia "ablation_num_canaries100_seed0_full_valsplit" "outputs/experiments/train/ablation_num_canaries100_seed0"

run_full "baseline_no_canary_seed1" "$TRAIN_NO_CANARY_JSONL" "$CANARY_META" --seed 1
run_full "baseline_no_canary_seed2" "$TRAIN_NO_CANARY_JSONL" "$CANARY_META" --seed 2

prepare_canary_data \
  "data/processed/train_with_canary_rep1.jsonl" \
  "data/processed/canaries_rep1.json" \
  --num_canaries 200 \
  --canary_repeats 1 \
  --seed 0
prepare_canary_data \
  "data/processed/train_with_canary_rep3.jsonl" \
  "data/processed/canaries_rep3.json" \
  --num_canaries 200 \
  --canary_repeats 3 \
  --seed 0

run_full "ablation_canary_repeats1_seed0" "data/processed/train_with_canary_rep1.jsonl" "data/processed/canaries_rep1.json" --seed 0
run_full "ablation_canary_repeats3_seed0" "data/processed/train_with_canary_rep3.jsonl" "data/processed/canaries_rep3.json" --seed 0
run_full "ablation_epochs2_seed0" "$TRAIN_CANARY_JSONL" "$CANARY_META" --epochs 2 --seed 0
run_full "ablation_epochs3_seed0" "$TRAIN_CANARY_JSONL" "$CANARY_META" --epochs 3 --seed 0
run_full "ablation_lr1em4_seed0" "$TRAIN_CANARY_JSONL" "$CANARY_META" --lr 1e-4 --seed 0
run_full "ablation_lr5em4_seed0" "$TRAIN_CANARY_JSONL" "$CANARY_META" --lr 5e-4 --seed 0
run_full "ablation_maxlen256_seed0" "$TRAIN_CANARY_JSONL" "$CANARY_META" --max_len 256 --seed 0
run_full "ablation_maxlen1024_seed0" "$TRAIN_CANARY_JSONL" "$CANARY_META" --max_len 1024 --seed 0
run_full "ablation_lorar8_seed0" "$TRAIN_CANARY_JSONL" "$CANARY_META" --lora_r 8 --seed 0
run_full "ablation_lorar32_seed0" "$TRAIN_CANARY_JSONL" "$CANARY_META" --lora_r 32 --seed 0
run_full "ablation_loraalpha16_seed0" "$TRAIN_CANARY_JSONL" "$CANARY_META" --lora_alpha 16 --seed 0
run_full "ablation_loraalpha64_seed0" "$TRAIN_CANARY_JSONL" "$CANARY_META" --lora_alpha 64 --seed 0
run_full "ablation_loradrop0_seed0" "$TRAIN_CANARY_JSONL" "$CANARY_META" --lora_dropout 0.0 --seed 0
run_full "ablation_loradrop10_seed0" "$TRAIN_CANARY_JSONL" "$CANARY_META" --lora_dropout 0.1 --seed 0
run_full "ablation_targetmodules_qv_seed0" "$TRAIN_CANARY_JSONL" "$CANARY_META" --target_modules q_proj v_proj --seed 0

run_attack "ablation_numsamples20_seed0" "outputs/experiments/train/baseline_standard_seed0" "$CANARY_META" --num_samples_per_prefix 20
run_attack "ablation_numsamples100_seed0" "outputs/experiments/train/baseline_standard_seed0" "$CANARY_META" --num_samples_per_prefix 100
run_attack "ablation_sampling_t07_p09_seed0" "outputs/experiments/train/baseline_standard_seed0" "$CANARY_META" --temperature 0.7 --top_p 0.9
run_attack "ablation_sampling_t10_p095_seed0" "outputs/experiments/train/baseline_standard_seed0" "$CANARY_META" --temperature 1.0 --top_p 0.95

run_exposure "ablation_numreference500_seed0" "outputs/experiments/train/baseline_standard_seed0" "$CANARY_META" --num_reference 500
run_exposure "ablation_numreference1000_seed0" "outputs/experiments/train/baseline_standard_seed0" "$CANARY_META" --num_reference 1000

run_mia "ablation_neigh3_worddrop005_seed0" "outputs/experiments/train/baseline_standard_seed0" --neigh_k 3 --word_drop 0.05
run_mia "ablation_neigh8_worddrop020_seed0" "outputs/experiments/train/baseline_standard_seed0" --neigh_k 8 --word_drop 0.2

echo "========== completion suite done =========="
