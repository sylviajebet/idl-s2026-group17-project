#!/bin/bash
#SBATCH --job-name=mha-ada-train
#SBATCH --account=##########
#SBATCH --output=/ocean/projects/cis260121p/shared/logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis260121p/shared/logs/%x_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

WORK_DIR="/ocean/projects/cis260121p/shared"
SCRIPT_DIR="/ocean/projects/cis260121p/shared/py_scripts"

mkdir -p "${WORK_DIR}/logs"

module load anaconda3
conda activate /ocean/projects/cis260121p/shared/envs/idl_deepfake_project

echo "===== MHA-ADA Training ====="
python "${SCRIPT_DIR}/mha_ada_training.py" \
    --working-dir "${WORK_DIR}" \
    --autoencoder-path "${WORK_DIR}/models/autoencoder.pt" \
    --save-path "${WORK_DIR}/models/MHA_ADA_model.pt" \
    --csv-dir "${WORK_DIR}/csv/MHA_ADA_split" \
    --checkpoint-dir "${WORK_DIR}/checkpoints/MHA_ADA" \
    --epochs 10 \
    --lr 1e-4 \
    --batch-size 16

echo "===== MHA-ADA Training complete! ====="
