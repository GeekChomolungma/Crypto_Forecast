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
INTERVAL_OVERRIDE="${3:-4h}"   # optional override; defaults to data.interval in config
INIT_MODE="${4:-pretrained}"   # pretrained | random
LOSS_MODE="${5:-native}"       # native|weighted_extreme_time_decay|magnitude_weighted|directional_hybrid
RUN_TAG="${6:-manual}"         # free-form label, e.g. v1 / expA
LOSS_QG="${7:-2.0}"            # loss_quantile_extreme_gamma
LOSS_QP="${8:-2.0}"            # loss_quantile_extreme_power
LOSS_TD="${9:-0.8}"            # loss_time_decay
LOSS_MA="${10:-1.0}"           # loss_magnitude_alpha
LOSS_DL="${11:-0.2}"           # loss_directional_lambda
LOSS_DT="${12:-0.1}"           # loss_directional_temperature

if [[ "${INIT_MODE}" != "pretrained" && "${INIT_MODE}" != "random" ]]; then
  echo "[finetune] invalid INIT_MODE=${INIT_MODE}, expected pretrained|random" >&2
  exit 1
fi

if [[ "${LOSS_MODE}" != "native" && "${LOSS_MODE}" != "weighted_extreme_time_decay" && "${LOSS_MODE}" != "magnitude_weighted" && "${LOSS_MODE}" != "directional_hybrid" ]]; then
  echo "[finetune] invalid LOSS_MODE=${LOSS_MODE}" >&2
  exit 1
fi

cd "${PROJECT_DIR}"

RESOLVED_INTERVAL="$(python - <<'PY' "${CONFIG_PATH}" "${INTERVAL_OVERRIDE}"
import sys, yaml

supported = {"1d", "4h", "1h", "15m"}
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
override = sys.argv[2].strip()
interval = (override or cfg["data"]["interval"]).lower()
if interval not in supported:
    raise SystemExit(f"Unsupported interval={interval!r}, expected one of {sorted(supported)}")
print(interval)
PY
)"

PROCESSED_PATH="${PROJECT_DIR}/data/processed/${SYMBOL}_${RESOLVED_INTERVAL}_logreturn.parquet"
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

SAFE_TAG="$(echo "${RUN_TAG}" | tr ' ' '_' | sed 's/[^A-Za-z0-9._-]/-/g' | sed 's/__\\+/_/g')"
if [[ -z "${SAFE_TAG}" ]]; then
  SAFE_TAG="manual"
fi
LOSS_SIGNATURE="$(python - <<'PY' "${LOSS_MODE}" "${LOSS_QG}" "${LOSS_QP}" "${LOSS_TD}" "${LOSS_MA}" "${LOSS_DL}" "${LOSS_DT}"
import sys

loss_mode = sys.argv[1]
vals = [float(x) for x in sys.argv[2:]]

loss_type = loss_mode

def fmt(v: float) -> str:
    return f"{v:.8g}".replace("-", "m").replace(".", "p")

if loss_type == "native":
    print("qg0_qp0_td0_ma0_dl0_dt0")
else:
    print("_".join([
        f"qg{fmt(vals[0])}",
        f"qp{fmt(vals[1])}",
        f"td{fmt(vals[2])}",
        f"ma{fmt(vals[3])}",
        f"dl{fmt(vals[4])}",
        f"dt{fmt(vals[5])}",
    ]))
PY
)"
RUN_NAME="${BASE_RUN_NAME}__${SYMBOL}__${RESOLVED_INTERVAL}__${INIT_MODE}__${LOSS_MODE}__lsig_${LOSS_SIGNATURE}__tag_${SAFE_TAG}"

echo "[finetune] symbol=${SYMBOL}"
echo "[finetune] interval=${RESOLVED_INTERVAL}"
echo "[finetune] init_mode=${INIT_MODE}, loss_mode=${LOSS_MODE}"
echo "[finetune] loss_signature=${LOSS_SIGNATURE}"
echo "[finetune] run_name=${RUN_NAME}"

echo "[finetune] processed=${PROCESSED_PATH}"
python -m crypto_forecast.pipelines.run_finetune \
  --config "${CONFIG_PATH}" \
  --processed "${PROCESSED_PATH}" \
  --symbol "${SYMBOL}" \
  --interval "${RESOLVED_INTERVAL}" \
  --run-name "${RUN_NAME}" \
  --init-mode "${INIT_MODE}" \
  --loss-mode "${LOSS_MODE}" \
  --loss-quantile-extreme-gamma "${LOSS_QG}" \
  --loss-quantile-extreme-power "${LOSS_QP}" \
  --loss-time-decay "${LOSS_TD}" \
  --loss-magnitude-alpha "${LOSS_MA}" \
  --loss-directional-lambda "${LOSS_DL}" \
  --loss-directional-temperature "${LOSS_DT}"
