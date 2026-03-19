#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ROOT_BASE:-}" ]]; then
  if [[ -d "/projects/prjs1859" ]]; then
    ROOT_BASE="/projects/prjs1859"
  elif [[ -d "/gpfs/work5/0/prjs1859" ]]; then
    ROOT_BASE="/gpfs/work5/0/prjs1859"
  else
    echo "[infer-batch] cannot resolve ROOT_BASE; set ROOT_BASE explicitly" >&2
    exit 1
  fi
fi
PROJECT_DIR="${PROJECT_DIR:-${ROOT_BASE}/Crypto_Forecast}"

CONFIG_PATH="${1:-configs/experiment.yaml}"
MODEL_SOURCE="${2:-local}"                  # local | hf
WEIGHT_SYMBOLS_CSV="${3:-BTCUSDT,ETHUSDT}"  # checkpoint symbols, one-to-one with infer symbols
INFER_SYMBOLS_CSV="${4:-BTCUSDT,ETHUSDT}"   # target symbols for parquet
INIT_MODE="${5:-pretrained}"                # pretrained | random
LOSS_MODE="${6:-native}"                    # native | custom
CKPT_TIMESTAMP="${7:-latest}"               # latest | YYYYmmdd_HHMMSS
RUN_TAG="${8:-}"                            # optional
HF_MODEL_ID="${9:-amazon/chronos-2}"        # used when hf
OUTPUT_TAG_BASE="${10:-batch}"

if [[ "${MODEL_SOURCE}" != "local" && "${MODEL_SOURCE}" != "hf" ]]; then
  echo "[infer-batch] invalid MODEL_SOURCE=${MODEL_SOURCE}, expected local|hf" >&2
  exit 1
fi

cd "${PROJECT_DIR}"

IFS=',' read -r -a WEIGHT_SYMBOLS <<< "${WEIGHT_SYMBOLS_CSV}"
IFS=',' read -r -a INFER_SYMBOLS <<< "${INFER_SYMBOLS_CSV}"
if [[ ${#WEIGHT_SYMBOLS[@]} -eq 0 || ${#INFER_SYMBOLS[@]} -eq 0 ]]; then
  echo "[infer-batch] empty weight or infer symbol list" >&2
  exit 1
fi

if [[ ${#WEIGHT_SYMBOLS[@]} -ne ${#INFER_SYMBOLS[@]} ]]; then
  echo "[infer-batch] list length mismatch: weight_symbols=${#WEIGHT_SYMBOLS[@]}, infer_symbols=${#INFER_SYMBOLS[@]}" >&2
  exit 1
fi

echo "[infer-batch] model_source=${MODEL_SOURCE}"
echo "[infer-batch] weight_symbols=${WEIGHT_SYMBOLS_CSV}"
echo "[infer-batch] infer_symbols=${INFER_SYMBOLS_CSV}"

for idx in "${!INFER_SYMBOLS[@]}"; do
  weight_symbol="$(echo "${WEIGHT_SYMBOLS[$idx]}" | xargs)"
  infer_symbol="$(echo "${INFER_SYMBOLS[$idx]}" | xargs)"

  if [[ -z "${weight_symbol}" || -z "${infer_symbol}" ]]; then
    echo "[infer-batch] empty symbol pair at index=${idx}, skipping" >&2
    continue
  fi

  if [[ "${MODEL_SOURCE}" == "local" && -z "${weight_symbol}" ]]; then
    echo "[infer-batch] empty weight symbol for local mode at index=${idx}" >&2
    exit 1
  fi

  output_tag="${OUTPUT_TAG_BASE}_${weight_symbol}_to_${infer_symbol}"
  echo "[infer-batch] start pair=${weight_symbol}->${infer_symbol}, output_tag=${output_tag}"

  bash scripts/infer/infer_single.sh \
    "${CONFIG_PATH}" \
    "${MODEL_SOURCE}" \
    "${weight_symbol}" \
    "${infer_symbol}" \
    "${INIT_MODE}" \
    "${LOSS_MODE}" \
    "${CKPT_TIMESTAMP}" \
    "${RUN_TAG}" \
    "${HF_MODEL_ID}" \
    "${output_tag}"

  echo "[infer-batch] done pair=${weight_symbol}->${infer_symbol}"
done

echo "[infer-batch] all done"
