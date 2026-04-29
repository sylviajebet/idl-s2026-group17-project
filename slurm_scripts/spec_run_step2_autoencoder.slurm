#!/bin/bash
#SBATCH --job-name=spec-autoencoder
#SBATCH --account=##########
#SBATCH --output=/ocean/projects/cis260121p/shared/logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis260121p/shared/logs/%x_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

WORK_DIR="/ocean/projects/cis260121p/shared"
SCRIPT_DIR="${WORK_DIR}/py_scripts"

# mkdir -p "${WORK_DIR}/logs"
mkdir -p "${WORK_DIR}/checkpoints/spec_autoencoder"
# mkdir -p "${WORK_DIR}/models"

module load anaconda3
conda activate ${WORK_DIR}/envs/idl_deepfake_project

echo "===== SPEC AUTOENCODER TRAINING ====="

python "${SCRIPT_DIR}/spec_autoencoder_training.py" \
    --data-dir "${WORK_DIR}/codecfake_pt" \
    --save-path "${WORK_DIR}/models/spec_autoencoder.pt" \
    --checkpoint-dir "${WORK_DIR}/checkpoints/spec_autoencoder" \
    --epochs 15 \
    --batch-size 16 \
    --max-samples 20000 \
    --use-augment

echo "===== SPEC AUTOENCODER COMPLETE ====="
