#!/bin/bash
#SBATCH --job-name=step4-admr
#SBATCH --account=cis260121p
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=63000M
#SBATCH --time=08:00:00

set -euo pipefail

WORK_DIR="${WORK_DIR:-/ocean/projects/cis260121p/shared}"
SCRIPT_DIR="${SCRIPT_DIR:-/ocean/projects/cis260121p/shared/py_scripts}"

# Load your environment (uncomment/adjust as needed)
module --ignore-cache load python/3.8.6 cuda/12.4.0
# source /ocean/projects/cis260121p/shared/envs/idl_deepfake_project/bin/activate
module load anaconda3
conda activate /ocean/projects/cis260121p/shared/envs/idl_deepfake_project

python "${SCRIPT_DIR}/admr_training.py" \
        --epochs 50 \
        --samples 100000
