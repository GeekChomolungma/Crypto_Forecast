#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ROOT_BASE:-}" ]]; then
  if [[ -d "/projects/prjs1859" ]]; then
    ROOT_BASE="/projects/prjs1859"
  elif [[ -d "/gpfs/work5/0/prjs1859" ]]; then
    ROOT_BASE="/gpfs/work5/0/prjs1859"
  else
    echo "[infer] cannot resolve ROOT_BASE; set ROOT_BASE explicitly" >&2
    exit 1
  fi
fi
PROJECT_DIR="${PROJECT_DIR:-${ROOT_BASE}/Crypto_Forecast}"

CONFIG_PATH="${1:-configs/experiment.yaml}"
MODEL_SOURCE="${2:-local}"              # local | hf
WEIGHT_SYMBOL="${3:-BTCUSDT}"           # used for checkpoint selection when local
INFER_SYMBOL="${4:-${WEIGHT_SYMBOL}}"   # target parquet symbol
INTERVAL_OVERRIDE="${5:-}"              # optional override; defaults to data.interval in config
INIT_MODE="${6:-pretrained}"            # pretrained | random
LOSS_MODE="${7:-native}"                # native|weighted_extreme_time_decay|magnitude_weighted|directional_hybrid
CKPT_TIMESTAMP="${8:-latest}"           # latest | YYYYmmdd_HHMMSS
RUN_TAG="${9:-}"                        # optional
HF_MODEL_ID="${10:-amazon/chronos-2}"   # used when hf
OUTPUT_TAG="${11:-manual}"
LOSS_SIGNATURE="${12:-}"                # optional exact local checkpoint signature filter

if [[ "${MODEL_SOURCE}" != "local" && "${MODEL_SOURCE}" != "hf" ]]; then
  echo "[infer] invalid MODEL_SOURCE=${MODEL_SOURCE}, expected local|hf" >&2
  exit 1
fi

if [[ -z "${INFER_SYMBOL}" ]]; then
  echo "[infer] infer symbol is empty" >&2
  exit 1
fi

cd "${PROJECT_DIR}"

cmd=(python -m crypto_forecast.pipelines.run_infer
  --config "${CONFIG_PATH}"
  --model-source "${MODEL_SOURCE}"
  --infer-symbol "${INFER_SYMBOL}"
  --output-tag "${OUTPUT_TAG}"
)

if [[ -n "${INTERVAL_OVERRIDE}" ]]; then
  cmd+=(--interval "${INTERVAL_OVERRIDE}")
fi

if [[ "${MODEL_SOURCE}" == "local" ]]; then
  if [[ -z "${WEIGHT_SYMBOL}" ]]; then
    echo "[infer] weight symbol is required for local model_source" >&2
    exit 1
  fi
  cmd+=(
    --weight-symbol "${WEIGHT_SYMBOL}"
    --init-mode "${INIT_MODE}"
    --loss-mode "${LOSS_MODE}"
  )
  if [[ -n "${LOSS_SIGNATURE}" ]]; then
    cmd+=(--loss-signature "${LOSS_SIGNATURE}")
  fi

  if [[ "${CKPT_TIMESTAMP}" != "latest" ]]; then
    cmd+=(--ckpt-timestamp "${CKPT_TIMESTAMP}")
  fi
  if [[ -n "${RUN_TAG}" ]]; then
    cmd+=(--run-tag "${RUN_TAG}")
  fi
else
  cmd+=(--hf-model-id "${HF_MODEL_ID}")
fi

echo "[infer] running: ${cmd[*]}"
"${cmd[@]}"
