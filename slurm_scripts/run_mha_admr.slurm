#!/bin/bash
#SBATCH --job-name=mha-admr-train
#SBATCH --account=##########
#SBATCH --output=/ocean/projects/cis260121p/shared/logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis260121p/shared/logs/%x_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

WORK_DIR="/ocean/projects/cis260121p/shared"
SCRIPT_DIR="${WORK_DIR}/py_scripts"

# mkdir -p "${WORK_DIR}/logs"
mkdir -p "${WORK_DIR}/checkpoints/MHA_ADMR"

module load anaconda3
conda activate ${WORK_DIR}/envs/idl_deepfake_project

echo "===== MHA-ADMR Training ====="

python "${SCRIPT_DIR}/mha_admr_training.py" \
    --working-dir "${WORK_DIR}" \
    --autoencoder-path "${WORK_DIR}/models/autoencoder.pt" \
    --save-path "${WORK_DIR}/models/MHA_ADMR_model.pt" \
    --csv-dir "${WORK_DIR}/csv/MHA_ADMR_split" \
    --checkpoint-dir "${WORK_DIR}/checkpoints/MHA_ADMR" \
    --epochs 45 \
    --lr 1e-4 \
    --batch-size 16 \
    --samples 75000 \
    --resume "/ocean/projects/cis260121p/shared/checkpoints/MHA_ADMR/checkpoint_epoch_45.pt"

echo "===== MHA-ADMR Training complete! ====="
