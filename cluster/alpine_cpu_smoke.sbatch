#!/bin/bash
#SBATCH --job-name=donut-cpu-smoke
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
RUN_DIR="${PROJECT_ROOT}/outputs/cpu-smoke/${SLURM_JOB_ID}"
RUN_LOG="${PROJECT_ROOT}/logs/cpu-smoke-${SLURM_JOB_ID}.log"

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

python train_donut.py \
  --output-dir "${RUN_DIR}" \
  --num-train-epochs 1 \
  --max-steps 2 \
  --per-device-train-batch-size 1 \
  --per-device-eval-batch-size 1 \
  --gradient-accumulation-steps 1 \
  --logging-steps 1 \
  --eval-strategy no \
  --save-strategy no

echo "Job finished at $(date)"
