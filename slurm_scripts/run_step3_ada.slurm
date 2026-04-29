#!/bin/bash
#SBATCH --job-name=step3-ada
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

python "${SCRIPT_DIR}/ada_training.py" \
    --working-dir "${WORK_DIR}" \
    --autoencoder-path "${WORK_DIR}/models/autoencoder.pt" \
    --save-path "${WORK_DIR}/models/ADA_model.pt" \
    --csv-dir "${WORK_DIR}/csv/ADA_split" \
    --checkpoint-dir "${WORK_DIR}/checkpoints/ADA" \
    --epochs 10 \
    --batch-size 16
