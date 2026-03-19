#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ROOT_BASE:-}" ]]; then
  if [[ -d "/projects/prjs1859" ]]; then
    ROOT_BASE="/projects/prjs1859"
  elif [[ -d "/gpfs/work5/0/prjs1859" ]]; then
    ROOT_BASE="/gpfs/work5/0/prjs1859"
  else
    echo "[finetune-batch] cannot resolve ROOT_BASE; set ROOT_BASE explicitly" >&2
    exit 1
  fi
fi
PROJECT_DIR="${PROJECT_DIR:-${ROOT_BASE}/Crypto_Forecast}"

CONFIG_PATH="${1:-configs/experiment.yaml}"
SYMBOLS_CSV="${2:-BTCUSDT,ETHUSDT,DOGEUSDT,BCHUSDT}"
INIT_MODE="${3:-pretrained}"   # pretrained | random
LOSS_MODE="${4:-native}"       # native | custom
RUN_TAG="${5:-batch}"

if [[ "${INIT_MODE}" != "pretrained" && "${INIT_MODE}" != "random" ]]; then
  echo "[finetune-batch] invalid INIT_MODE=${INIT_MODE}, expected pretrained|random" >&2
  exit 1
fi

if [[ "${LOSS_MODE}" != "native" && "${LOSS_MODE}" != "custom" ]]; then
  echo "[finetune-batch] invalid LOSS_MODE=${LOSS_MODE}, expected native|custom" >&2
  exit 1
fi

cd "${PROJECT_DIR}"

# Comma-separated symbols, e.g. "BTCUSDT,ETHUSDT,SOLUSDT"
IFS=',' read -r -a SYMBOLS <<< "${SYMBOLS_CSV}"

if [[ ${#SYMBOLS[@]} -eq 0 ]]; then
  echo "[finetune-batch] empty symbol list" >&2
  exit 1
fi

echo "[finetune-batch] symbols=${SYMBOLS_CSV}"
echo "[finetune-batch] init_mode=${INIT_MODE}, loss_mode=${LOSS_MODE}, run_tag=${RUN_TAG}"

for raw_symbol in "${SYMBOLS[@]}"; do
  symbol="$(echo "${raw_symbol}" | xargs)"
  if [[ -z "${symbol}" ]]; then
    continue
  fi

  echo "[finetune-batch] start symbol=${symbol}"
  bash scripts/finetuning/finetune_single.sh \
    "${CONFIG_PATH}" \
    "${symbol}" \
    "${INIT_MODE}" \
    "${LOSS_MODE}" \
    "${RUN_TAG}"
  echo "[finetune-batch] done symbol=${symbol}"
done

echo "[finetune-batch] all done"
