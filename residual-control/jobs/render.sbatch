#!/bin/bash
#SBATCH --job-name=residual-gifs
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=residual-control/jobs/logs/%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
benchmark-suite/.venv/bin/python -m residual_control.render
