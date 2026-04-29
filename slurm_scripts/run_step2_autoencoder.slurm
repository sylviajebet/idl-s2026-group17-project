#!/bin/bash
#SBATCH --job-name=step2-autoencoder
#SBATCH --account=##########
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

WORK_DIR="${WORK_DIR:-/ocean/projects/cis260121p/shared}"
SCRIPT_DIR="${SCRIPT_DIR:-/ocean/projects/cis260121p/shared/py_scripts}"

python "${SCRIPT_DIR}/autoencoder_training.py" \
    --data-dir "${WORK_DIR}/pt_files/codecfake_pt" \
    --save-path "${WORK_DIR}/models/autoencoder.pt" \
    --checkpoint-dir "${WORK_DIR}/checkpoints/autoencoder" \
    --epochs 50 \
    --batch-size 16
