#!/bin/bash
#SBATCH --job-name=step4-admr
#SBATCH --account=##########
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

WORK_DIR="${WORK_DIR:-/ocean/projects/cis260121p/shared}"
SCRIPT_DIR="${SCRIPT_DIR:-/ocean/projects/cis260121p/shared/py_scripts}"

python "${SCRIPT_DIR}/admr_training.py" \
    --working-dir "${WORK_DIR}" \
    --data-dir "${WORK_DIR}/pt_files/codecfake_pt" \
    --autoencoder-path "${WORK_DIR}/models/autoencoder.pt" \
    --save-path "${WORK_DIR}/models/ADMR_model.pt" \
    --csv-dir "${WORK_DIR}/csv/ADMR_split" \
    --checkpoint-dir "${WORK_DIR}/checkpoints/ADMR" \
    --epochs 25 \
    --batch-size 16
