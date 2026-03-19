#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/experiment.yaml}"
PROCESSED_PATH="${2:-/gpfs/work5/0/prjs1859/Crypto_Forecast/data/processed/combined_1d_logreturn.parquet}"
HF_MODEL_ID="${3:-amazon/chronos-2}"
OUTPUT_TAG="${4:-single}"

cd /gpfs/work5/0/prjs1859/Crypto_Forecast
python -m crypto_forecast.pipelines.run_infer \
  --config "${CONFIG_PATH}" \
  --processed "${PROCESSED_PATH}" \
  --model-source hf \
  --model-ref "${HF_MODEL_ID}" \
  --output-tag "zeroshot_${OUTPUT_TAG}"
