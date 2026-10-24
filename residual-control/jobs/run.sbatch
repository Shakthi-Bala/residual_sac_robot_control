#!/bin/bash
#SBATCH --job-name=residual-control
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=residual-control/jobs/logs/%x-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export MUJOCO_GL=disable
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec benchmark-suite/.venv/bin/python -m "$@"
