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
PROCESSED_DIR="${2:-${PROJECT_DIR}/data/processed}"
HF_MODEL_ID="${3:-amazon/chronos-2}"

cd "${PROJECT_DIR}"

# Batch over per-symbol processed parquet files.
# Skip combined file because it can be evaluated separately.
find "${PROCESSED_DIR}" -maxdepth 1 -type f -name "*_1d_logreturn.parquet" ! -name "combined_1d_logreturn.parquet" | sort | while read -r f; do
  base_name="$(basename "${f}" .parquet)"
  echo "[zero-shot] running ${base_name}"

  python -m crypto_forecast.pipelines.run_infer \
    --config "${CONFIG_PATH}" \
    --processed "${f}" \
    --model-source hf \
    --hf-model-id "${HF_MODEL_ID}" \
    --output-tag "zeroshot_${base_name}"
done

echo "[zero-shot] batch done"
