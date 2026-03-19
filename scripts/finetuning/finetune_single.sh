#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${ROOT_BASE:-}" ]]; then
  if [[ -d "/projects/prjs1859" ]]; then
    ROOT_BASE="/projects/prjs1859"
  elif [[ -d "/gpfs/work5/0/prjs1859" ]]; then
    ROOT_BASE="/gpfs/work5/0/prjs1859"
  else
    echo "[finetune] cannot resolve ROOT_BASE; set ROOT_BASE explicitly" >&2
    exit 1
  fi
fi
PROJECT_DIR="${PROJECT_DIR:-${ROOT_BASE}/Crypto_Forecast}"

CONFIG_PATH="${1:-configs/experiment.yaml}"
SYMBOL="${2:-BTCUSDT}"
INIT_MODE="${3:-pretrained}"   # pretrained | random
LOSS_MODE="${4:-native}"       # native | custom
RUN_TAG="${5:-manual}"         # free-form label, e.g. v1 / expA

if [[ "${INIT_MODE}" != "pretrained" && "${INIT_MODE}" != "random" ]]; then
  echo "[finetune] invalid INIT_MODE=${INIT_MODE}, expected pretrained|random" >&2
  exit 1
fi

if [[ "${LOSS_MODE}" != "native" && "${LOSS_MODE}" != "custom" ]]; then
  echo "[finetune] invalid LOSS_MODE=${LOSS_MODE}, expected native|custom" >&2
  exit 1
fi

cd "${PROJECT_DIR}"

PROCESSED_PATH="${PROJECT_DIR}/data/processed/${SYMBOL}_1d_logreturn.parquet"
if [[ ! -f "${PROCESSED_PATH}" ]]; then
  echo "[finetune] processed file not found: ${PROCESSED_PATH}" >&2
  exit 1
fi

BASE_RUN_NAME="$(python - <<'PY' "${CONFIG_PATH}"
import sys, yaml
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
print(cfg['project']['run_name'])
PY
)"

TS="$(date +%Y%m%d_%H%M%S)"
SAFE_TAG="${RUN_TAG// /_}"
RUN_NAME="${BASE_RUN_NAME}__${SYMBOL}__${INIT_MODE}__${LOSS_MODE}__${SAFE_TAG}__${TS}"

echo "[finetune] symbol=${SYMBOL}"
echo "[finetune] init_mode=${INIT_MODE}, loss_mode=${LOSS_MODE}"
echo "[finetune] run_name=${RUN_NAME}"

echo "[finetune] processed=${PROCESSED_PATH}"
python -m crypto_forecast.pipelines.run_finetune \
  --config "${CONFIG_PATH}" \
  --processed "${PROCESSED_PATH}" \
  --run-name "${RUN_NAME}" \
  --init-mode "${INIT_MODE}" \
  --loss-mode "${LOSS_MODE}"
