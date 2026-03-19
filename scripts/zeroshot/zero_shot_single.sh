#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ROOT_BASE:-}" ]]; then
  if [[ -d "/projects/prjs1859" ]]; then
    ROOT_BASE="/projects/prjs1859"
  elif [[ -d "/gpfs/work5/0/prjs1859" ]]; then
    ROOT_BASE="/gpfs/work5/0/prjs1859"
  else
    echo "[zero-shot] cannot resolve ROOT_BASE; set ROOT_BASE explicitly" >&2
    exit 1
  fi
fi
PROJECT_DIR="${PROJECT_DIR:-${ROOT_BASE}/Crypto_Forecast}"

CONFIG_PATH="${1:-configs/experiment.yaml}"
PROCESSED_PATH="${2:-${PROJECT_DIR}/data/processed/combined_1d_logreturn.parquet}"
HF_MODEL_ID="${3:-amazon/chronos-2}"
OUTPUT_TAG="${4:-single}"

cd "${PROJECT_DIR}"
python -m crypto_forecast.pipelines.run_infer \
  --config "${CONFIG_PATH}" \
  --processed "${PROCESSED_PATH}" \
  --model-source hf \
  --hf-model-id "${HF_MODEL_ID}" \
  --output-tag "zeroshot_${OUTPUT_TAG}"
