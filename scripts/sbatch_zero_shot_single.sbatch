#!/usr/bin/env bash
#SBATCH -p gpu_h100
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -t 24:00:00
#SBATCH -J CF_ZS_SINGLE
#SBATCH -D /gpfs/work5/0/prjs1859/Crypto_Forecast
#SBATCH -o /gpfs/work5/0/prjs1859/users/%u/slurm/logs/%x_%j.out
#SBATCH -e /gpfs/work5/0/prjs1859/users/%u/slurm/logs/%x_%j.err

set -euo pipefail

ROOT_BASE="${ROOT_BASE:-/gpfs/work5/0/prjs1859}"
PROJECT_DIR="${PROJECT_DIR:-${ROOT_BASE}/Crypto_Forecast}"
LOG_DIR="${LOG_DIR:-${ROOT_BASE}/users/${USER}/slurm/logs}"

mkdir -p "${LOG_DIR}"

source "${ROOT_BASE}/load_exp_env.sh"

which python
python -c "import torch; print('torch ok', torch.__version__)"

# Positional args forwarded to zero_shot_single.sh:
#   $1 config_path
#   $2 processed_path
#   $3 hf_model_id
#   $4 output_tag
cd "${PROJECT_DIR}"
bash scripts/zero_shot_single.sh "${1:-configs/experiment.yaml}" "${2:-${PROJECT_DIR}/data/processed/combined_1d_logreturn.parquet}" "${3:-amazon/chronos-2}" "${4:-single}"
