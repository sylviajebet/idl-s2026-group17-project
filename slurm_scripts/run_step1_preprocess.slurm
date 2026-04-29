#!/bin/bash
#SBATCH --job-name=step1-preprocess
#SBATCH --account=##########
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:h100-80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

WORK_DIR="${WORK_DIR:-/ocean/projects/cis260121p/shared}"
SCRIPT_DIR="${SCRIPT_DIR:-/ocean/projects/cis260121p/shared/py_scripts}"
CODECFAKE_DIR="${CODECFAKE_DIR:-/ocean/projects/cis260121p/shared/datasets/codecfake/train/extracted/train}"
ASVSPOOF_DIR="${ASVSPOOF_DIR:-/ocean/projects/cis260121p/shared/datasets/avsspoof_2021/avsspoof_2021_extracted/ASVspoof2021_DF_eval_part00/ASVspoof2021_DF_eval/flac}"
FOR_DIR="${FOR_DIR:-/ocean/projects/cis260121p/shared/datasets/for/for-original/for-original/training/fake}"

mkdir -p "${WORK_DIR}/logs"

python "${SCRIPT_DIR}/preprocessing_step.py" \
    --working-dir "${WORK_DIR}" \
    --codecfake-dir "${CODECFAKE_DIR}" \
    --asvspoof-dir "${ASVSPOOF_DIR}" \
    --for-dir "${FOR_DIR}"
