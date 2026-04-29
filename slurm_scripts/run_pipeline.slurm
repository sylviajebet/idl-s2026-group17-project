#!/bin/bash
#SBATCH --job-name=deepfake-pipeline
#SBATCH --account=##########
#SBATCH --output=/ocean/projects/cis260121p/shared/logs/%x_%j.out
#SBATCH --error=/ocean/projects/cis260121p/shared/logs/%x_%j.err
#SBATCH --partition=GPU-shared           # <-- change to your cluster's GPU partition
#SBATCH --gres=gpu:v100-32:1              # request 1 GPU
#SBATCH --cpus-per-task=5
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================
# Audio Deepfake Pipeline — Full Run
# Runs: Preprocessing → Autoencoder → ADA → ADMR
#
# Usage:
#   sbatch run_pipeline.slurm
#
# Or submit each step individually:
#   sbatch run_step1_preprocess.slurm
#   sbatch --dependency=afterok:<JOB_ID> run_step2_autoencoder.slurm
#   sbatch --dependency=afterok:<JOB_ID> run_step3_ada.slurm
#   sbatch --dependency=afterok:<JOB_ID> run_step4_admr.slurm
# ============================================================

set -euo pipefail

# ---------- EDIT THESE PATHS ----------
WORK_DIR="/ocean/projects/cis260121p/shared"          # where outputs/.pt files go
SCRIPT_DIR="/ocean/projects/cis260121p/shared/py_scripts"            # where the .py files live
CODECFAKE_DIR="/ocean/projects/cis260121p/shared/datasets/codecfake/train/extracted/train"  # raw CodecFake .wav files
ASVSPOOF_DIR="/ocean/projects/cis260121p/shared/datasets/avsspoof_2021/avsspoof_2021_extracted/ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac"  # raw ASVspoof .flac files
FOR_DIR="/ocean/projects/cis260121p/shared/datasets/for/for-original/for-original/training/fake"  # raw FakeOrReal audio files      

# --------------------------------------

mkdir -p "${WORK_DIR}/logs"

# Load your environment (uncomment/adjust as needed)
module --ignore-cache load python/3.8.6 cuda/12.4.0
# source /ocean/projects/cis260121p/shared/envs/idl_deepfake_project/bin/activate
module load anaconda3
conda activate /ocean/projects/cis260121p/shared/envs/idl_deepfake_project

pip install -r /ocean/projects/cis260121p/shared/requirements.txt

echo "===== Step 1: Preprocessing ====="
python "${SCRIPT_DIR}/preprocessing_step.py" \
    --working-dir "${WORK_DIR}" \
    --codecfake-dir "${CODECFAKE_DIR}" \
    --asvspoof-dir "${ASVSPOOF_DIR}" \
    --for-dir "${FOR_DIR}" \
    --codecfake-n 1000 \
    --asvspoof-n 30000 \
    --for-n 30000

# echo "===== Step 2: Autoencoder Training ====="
# python "${SCRIPT_DIR}/autoencoder_training.py" \
#     --data-dir "${WORK_DIR}/codecfake_pt" \
#     --save-path "${WORK_DIR}/models/autoencoder.pt" \
#     --checkpoint-dir "${WORK_DIR}/checkpoints/autoencoder" \
#     --epochs 5 \
#     --lr 1e-4 \
#     --batch-size 16 \
#     --max-samples 500

# echo "===== Step 3: ADA Training ====="
# python "${SCRIPT_DIR}/ada_training.py" \
#     --working-dir "${WORK_DIR}" \
#     --autoencoder-path "${WORK_DIR}/models/autoencoder.pt" \
#     --save-path "${WORK_DIR}/models/ADA_model.pt" \
#     --csv-dir "${WORK_DIR}/csv/ADA_split" \
#     --checkpoint-dir "${WORK_DIR}/checkpoints/ADA" \
#     --epochs 5 \
#     --lr 1e-4 \
#     --batch-size 16

# echo "===== Step 4: ADMR Training ====="
# python "${SCRIPT_DIR}/admr_training.py" \
#     --working-dir "${WORK_DIR}" \
#     --data-dir "${WORK_DIR}/codecfake_pt" \
#     --autoencoder-path "${WORK_DIR}/models/autoencoder.pt" \
#     --save-path "${WORK_DIR}/models/ADMR_model.pt" \
#     --csv-dir "${WORK_DIR}/csv/ADMR_split" \
#     --checkpoint-dir "${WORK_DIR}/checkpoints/ADMR" \
#     --epochs 5 \
#     --lr 1e-4 \
#     --batch-size 16

echo "===== Pipeline complete! ====="
