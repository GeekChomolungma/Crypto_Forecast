#!/usr/bin/env bash
#SBATCH -p gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 48:00:00
#SBATCH -J CF_INFER_BATCH
#SBATCH -D /projects/prjs1859/Crypto_Forecast
#SBATCH -o /projects/prjs1859/users/%u/slurm/logs/%x_%j.out
#SBATCH -e /projects/prjs1859/users/%u/slurm/logs/%x_%j.err

set -euo pipefail

if [[ -z "${ROOT_BASE:-}" ]]; then
  if [[ -d "/projects/prjs1859" ]]; then
    ROOT_BASE="/projects/prjs1859"
  elif [[ -d "/gpfs/work5/0/prjs1859" ]]; then
    ROOT_BASE="/gpfs/work5/0/prjs1859"
  else
    echo "[sbatch-infer-batch] cannot resolve ROOT_BASE; set ROOT_BASE explicitly" >&2
    exit 1
  fi
fi
PROJECT_DIR="${PROJECT_DIR:-${ROOT_BASE}/Crypto_Forecast}"
LOG_DIR="${LOG_DIR:-${ROOT_BASE}/users/${USER}/slurm/logs}"

mkdir -p "${LOG_DIR}"

source "${ROOT_BASE}/load_exp_env.sh"

which python
python -c "import torch; print('torch ok', torch.__version__)"

# Positional args:
#   $1 config_path
#   $2 model_source (local|hf)
#   $3 weight_symbols_csv (e.g. BTCUSDT,LTCUSDT,XRPUSDT)
#   $4 infer_symbols_csv (e.g. BTCUSDT,LTCUSDT,XRPUSDT)
#      one-to-one mapping: weight_symbols_csv[i] -> infer_symbols_csv[i]
#   $5 interval override (optional; defaults to data.interval in config)
#   $6 init_mode (pretrained|random)
#   $7 loss_mode (native|weighted_extreme_time_decay|magnitude_weighted|directional_hybrid)
#   $8 ckpt_timestamp (latest|YYYYmmdd_HHMMSS)
#   $9 run_tag
#   $10 hf_model_id
#   $11 output_tag_base
#   $12 loss_signature (optional exact local checkpoint signature filter)
cd "${PROJECT_DIR}"

CONFIG_PATH="${1:-configs/experiment.yaml}"
MODEL_SOURCE="${2:-local}"
WEIGHT_SYMBOLS_CSV="${3:-BTCUSDT}"
INFER_SYMBOLS_CSV="${4:-BTCUSDT}"
INTERVAL_OVERRIDE="${5:-4h}"
INIT_MODE="${6:-pretrained}"
LOSS_MODE="${7:-native}"
CKPT_TIMESTAMP="${8:-latest}"
RUN_TAG="${9:-}"
HF_MODEL_ID="${10:-amazon/chronos-2}"
OUTPUT_TAG_BASE="${11:-batch}"
LOSS_SIGNATURE="${12:-qg0_qp0_td0_ma0_dl0_dt0}"

if [[ "${MODEL_SOURCE}" != "local" && "${MODEL_SOURCE}" != "hf" ]]; then
  echo "[sbatch-infer-batch] invalid MODEL_SOURCE=${MODEL_SOURCE}, expected local|hf" >&2
  exit 1
fi

IFS=',' read -r -a WEIGHT_SYMBOLS <<< "${WEIGHT_SYMBOLS_CSV}"
IFS=',' read -r -a INFER_SYMBOLS <<< "${INFER_SYMBOLS_CSV}"
if [[ ${#WEIGHT_SYMBOLS[@]} -eq 0 || ${#INFER_SYMBOLS[@]} -eq 0 ]]; then
  echo "[sbatch-infer-batch] empty weight or infer symbol list" >&2
  exit 1
fi

if [[ ${#WEIGHT_SYMBOLS[@]} -ne ${#INFER_SYMBOLS[@]} ]]; then
  echo "[sbatch-infer-batch] list length mismatch: weight_symbols=${#WEIGHT_SYMBOLS[@]}, infer_symbols=${#INFER_SYMBOLS[@]}" >&2
  exit 1
fi

echo "[sbatch-infer-batch] model_source=${MODEL_SOURCE}"
echo "[sbatch-infer-batch] weight_symbols=${WEIGHT_SYMBOLS_CSV}"
echo "[sbatch-infer-batch] infer_symbols=${INFER_SYMBOLS_CSV}"
if [[ -n "${INTERVAL_OVERRIDE}" ]]; then
  echo "[sbatch-infer-batch] interval_override=${INTERVAL_OVERRIDE}"
fi

for idx in "${!INFER_SYMBOLS[@]}"; do
  weight_symbol="$(echo "${WEIGHT_SYMBOLS[$idx]}" | xargs)"
  infer_symbol="$(echo "${INFER_SYMBOLS[$idx]}" | xargs)"

  if [[ -z "${weight_symbol}" || -z "${infer_symbol}" ]]; then
    echo "[sbatch-infer-batch] empty symbol pair at index=${idx}, skipping" >&2
    continue
  fi

  output_tag="${OUTPUT_TAG_BASE}_${weight_symbol}_to_${infer_symbol}"
  echo "[sbatch-infer-batch] start pair=${weight_symbol}->${infer_symbol}, output_tag=${output_tag}"

  bash scripts/infer/infer_single.sh \
    "${CONFIG_PATH}" \
    "${MODEL_SOURCE}" \
    "${weight_symbol}" \
    "${infer_symbol}" \
    "${INTERVAL_OVERRIDE}" \
    "${INIT_MODE}" \
    "${LOSS_MODE}" \
    "${CKPT_TIMESTAMP}" \
    "${RUN_TAG}" \
    "${HF_MODEL_ID}" \
    "${output_tag}" \
    "${LOSS_SIGNATURE}"

  echo "[sbatch-infer-batch] done pair=${weight_symbol}->${infer_symbol}"
done

echo "[sbatch-infer-batch] all done"
