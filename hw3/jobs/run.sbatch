#!/bin/bash
#SBATCH --job-name=hw3
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --output=jobs/logs/%x-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MUJOCO_GL=disable
export PYTHONUNBUFFERED=1
if [[ -f ../.env ]]; then
    set -a
    source ../.env
    set +a
fi
exec .venv/bin/python jobs/run_experiment.py "$1" "${SLURM_ARRAY_TASK_ID:-0}"
