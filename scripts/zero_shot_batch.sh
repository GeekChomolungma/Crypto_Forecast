#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/experiment.yaml}"
PROCESSED_DIR="${2:-/gpfs/work5/0/prjs1859/Crypto_Forecast/data/processed}"
HF_MODEL_ID="${3:-amazon/chronos-2}"

cd /gpfs/work5/0/prjs1859/Crypto_Forecast

# Batch over per-symbol processed parquet files.
# Skip combined file because it can be evaluated separately.
find "${PROCESSED_DIR}" -maxdepth 1 -type f -name "*_1d_logreturn.parquet" ! -name "combined_1d_logreturn.parquet" | sort | while read -r f; do
  base_name="$(basename "${f}" .parquet)"
  echo "[zero-shot] running ${base_name}"

  python -m crypto_forecast.pipelines.run_infer \
    --config "${CONFIG_PATH}" \
    --processed "${f}" \
    --model-source hf \
    --model-ref "${HF_MODEL_ID}" \
    --output-tag "zeroshot_${base_name}"
done

echo "[zero-shot] batch done"
