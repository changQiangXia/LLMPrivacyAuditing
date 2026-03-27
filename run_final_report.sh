#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python}"
REGISTRY_YAML="${REGISTRY_YAML:-configs/final_registry.yaml}"
TITLE_PREFIX="${TITLE_PREFIX:-Final Audit | final_registry}"

"$PYTHON" -m src.build_final_report \
  --registry_yaml "$REGISTRY_YAML" \
  --runs_csv artifacts/reports/final_runs.csv \
  --runs_md artifacts/reports/final_runs.md \
  --summary_csv artifacts/reports/final_summary.csv \
  --summary_md artifacts/reports/final_summary.md \
  --factor_csv artifacts/reports/final_factor_summary.csv \
  --factor_md artifacts/reports/final_factor_summary.md \
  --plot_dir artifacts/reports/final_plots \
  --title_prefix "$TITLE_PREFIX"

echo "[OK] final report artifacts updated"
echo "  registry -> $REGISTRY_YAML"
echo "  summary  -> artifacts/reports/final_summary.md"
echo "  factors  -> artifacts/reports/final_factor_summary.md"
echo "  plots    -> artifacts/reports/final_plots"
