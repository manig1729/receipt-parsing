#!/bin/bash
#SBATCH --job-name=donut-smoke
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=36G
#SBATCH --time=00:45:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
RUN_DIR="${PROJECT_ROOT}/outputs/smoke/${SLURM_JOB_ID}"
RUN_LOG="${PROJECT_ROOT}/logs/smoke-${SLURM_JOB_ID}.log"

mkdir -p "${PROJECT_ROOT}/logs" "${RUN_DIR}"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "Job started at $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Project root: ${PROJECT_ROOT}"

module purge || true

cd "${PROJECT_ROOT}"
source .venv/bin/activate

export HF_HOME="${PROJECT_ROOT}/.hf-cache"
export TOKENIZERS_PARALLELISM=false

nvidia-smi || true

python train_donut.py \
  --output-dir "${RUN_DIR}" \
  --num-train-epochs 1 \
  --max-steps 20 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --logging-steps 5 \
  --eval-strategy no \
  --save-strategy steps \
  --save-total-limit 1

echo "Job finished at $(date)"
