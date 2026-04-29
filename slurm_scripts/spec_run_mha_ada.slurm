#!/bin/bash
#SBATCH --job-name=spec-mha-ada
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
mkdir -p "${WORK_DIR}/checkpoints/spec_MHA_ADA"
# mkdir -p "${WORK_DIR}/models"

module load anaconda3
conda activate ${WORK_DIR}/envs/idl_deepfake_project

echo "===== SPEC MHA-ADA TRAINING ====="

python "${SCRIPT_DIR}/spec_mha_ada_training.py" \
    --working-dir "${WORK_DIR}" \
    --autoencoder-path "${WORK_DIR}/models/spec_autoencoder.pt" \
    --save-path "${WORK_DIR}/models/spec_MHA_ADA_model.pt" \
    --csv-dir "${WORK_DIR}/csv/MHA_ADA_split" \
    --checkpoint-dir "${WORK_DIR}/checkpoints/spec_MHA_ADA" \
    --epochs 10 \
    --lr 1e-4 \
    --batch-size 16

echo "===== SPEC MHA-ADA COMPLETE ====="
