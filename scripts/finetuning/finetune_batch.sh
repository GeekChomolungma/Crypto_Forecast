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
LOSS_MODE="${4:-native}"       # native|weighted_extreme_time_decay|magnitude_weighted|directional_hybrid
RUN_TAG="${5:-batch}"
LOSS_QG="${6:-2.0}"            # loss_quantile_extreme_gamma
LOSS_QP="${7:-2.0}"            # loss_quantile_extreme_power
LOSS_TD="${8:-0.8}"            # loss_time_decay
LOSS_MA="${9:-1.0}"            # loss_magnitude_alpha
LOSS_DL="${10:-0.2}"           # loss_directional_lambda
LOSS_DT="${11:-0.1}"           # loss_directional_temperature

if [[ "${INIT_MODE}" != "pretrained" && "${INIT_MODE}" != "random" ]]; then
  echo "[finetune-batch] invalid INIT_MODE=${INIT_MODE}, expected pretrained|random" >&2
  exit 1
fi

if [[ "${LOSS_MODE}" != "native" && "${LOSS_MODE}" != "weighted_extreme_time_decay" && "${LOSS_MODE}" != "magnitude_weighted" && "${LOSS_MODE}" != "directional_hybrid" ]]; then
  echo "[finetune-batch] invalid LOSS_MODE=${LOSS_MODE}" >&2
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
echo "[finetune-batch] loss_params=qg=${LOSS_QG},qp=${LOSS_QP},td=${LOSS_TD},ma=${LOSS_MA},dl=${LOSS_DL},dt=${LOSS_DT}"

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
    "${RUN_TAG}" \
    "${LOSS_QG}" \
    "${LOSS_QP}" \
    "${LOSS_TD}" \
    "${LOSS_MA}" \
    "${LOSS_DL}" \
    "${LOSS_DT}"
  echo "[finetune-batch] done symbol=${symbol}"
done

echo "[finetune-batch] all done"
