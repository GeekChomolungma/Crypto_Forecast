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

# Positional args forwarded to infer_batch.sh:
#   $1 config_path
#   $2 model_source (local|hf)
#   $3 weight_symbols_csv (e.g. BTCUSDT,LTCUSDT,XRPUSDT)
#   $4 infer_symbols_csv (e.g. BTCUSDT,LTCUSDT,XRPUSDT)
#      one-to-one mapping: weight_symbols_csv[i] -> infer_symbols_csv[i]
#   $5 init_mode (pretrained|random)
#   $6 loss_mode (native|custom)
#   $7 ckpt_timestamp (latest|YYYYmmdd_HHMMSS)
#   $8 run_tag
#   $9 hf_model_id
#   $10 output_tag_base
cd "${PROJECT_DIR}"
bash scripts/infer/infer_batch.sh \
  "${1:-configs/experiment.yaml}" \
  "${2:-local}" \
  "${3:-BTCUSDT,ETHUSDT,DOGEUSDT,BCHUSDT,LTCUSDT,XRPUSDT}" \
  "${4:-BTCUSDT,ETHUSDT,DOGEUSDT,BCHUSDT,LTCUSDT,XRPUSDT}" \
  "${5:-pretrained}" \
  "${6:-native}" \
  "${7:-latest}" \
  "${8:-}" \
  "${9:-amazon/chronos-2}" \
  "${10:-batch}"
