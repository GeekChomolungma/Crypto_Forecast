#!/usr/bin/env bash
#SBATCH -p gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 48:00:00
#SBATCH -J CF_FT_BATCH
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
    echo "[sbatch-finetune-batch] cannot resolve ROOT_BASE; set ROOT_BASE explicitly" >&2
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
#   $2 symbols_csv (e.g. BTCUSDT,ETHUSDT)
#   $3 interval override (optional; defaults to data.interval in config)
#   $4 init_mode (pretrained|random)
#   $5 loss_mode (native|weighted_extreme_time_decay|magnitude_weighted|directional_hybrid)
#   $6 run_tag
#   $7 loss_quantile_extreme_gamma
#   $8 loss_quantile_extreme_power
#   $9 loss_time_decay
#   $10 loss_magnitude_alpha
#   $11 loss_directional_lambda
#   $12 loss_directional_temperature
cd "${PROJECT_DIR}"

CONFIG_PATH="${1:-configs/experiment.yaml}"
SYMBOLS_CSV="${2:-BTCUSDT,ETHUSDT,DOGEUSDT,BCHUSDT,BNBUSDT,LTCUSDT,XRPUSDT,ZECUSDT}"
INTERVAL_OVERRIDE="${3:-4h}"
INIT_MODE="${4:-pretrained}"
LOSS_MODE="${5:-native}"
RUN_TAG="${6:-batch}"
LOSS_QG="${7:-2.0}"
LOSS_QP="${8:-2.0}"
LOSS_TD="${9:-0.8}"
LOSS_MA="${10:-1.0}"
LOSS_DL="${11:-0.2}"
LOSS_DT="${12:-0.1}"

IFS=',' read -r -a SYMBOLS <<< "${SYMBOLS_CSV}"
if [[ ${#SYMBOLS[@]} -eq 0 ]]; then
  echo "[sbatch-finetune-batch] empty symbol list" >&2
  exit 1
fi

echo "[sbatch-finetune-batch] symbols=${SYMBOLS_CSV}"
if [[ -n "${INTERVAL_OVERRIDE}" ]]; then
  echo "[sbatch-finetune-batch] interval_override=${INTERVAL_OVERRIDE}"
fi
echo "[sbatch-finetune-batch] init_mode=${INIT_MODE}, loss_mode=${LOSS_MODE}, run_tag=${RUN_TAG}"
echo "[sbatch-finetune-batch] loss_params=qg=${LOSS_QG},qp=${LOSS_QP},td=${LOSS_TD},ma=${LOSS_MA},dl=${LOSS_DL},dt=${LOSS_DT}"

for raw_symbol in "${SYMBOLS[@]}"; do
  symbol="$(echo "${raw_symbol}" | xargs)"
  if [[ -z "${symbol}" ]]; then
    continue
  fi

  echo "[sbatch-finetune-batch] start symbol=${symbol}"
  bash scripts/finetuning/finetune_single.sh \
    "${CONFIG_PATH}" \
    "${symbol}" \
    "${INTERVAL_OVERRIDE}" \
    "${INIT_MODE}" \
    "${LOSS_MODE}" \
    "${RUN_TAG}" \
    "${LOSS_QG}" \
    "${LOSS_QP}" \
    "${LOSS_TD}" \
    "${LOSS_MA}" \
    "${LOSS_DL}" \
    "${LOSS_DT}"
  echo "[sbatch-finetune-batch] done symbol=${symbol}"
done

echo "[sbatch-finetune-batch] all done"
