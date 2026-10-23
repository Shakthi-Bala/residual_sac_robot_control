# HW3: Q-Learning and Actor-Critic Algorithms

The runnable package for [the project](../README.md), based on Berkeley CS185/285 Spring 2026 HW3.

## Setup

Use Python 3.10 and install SWIG for Box2D (`sudo apt-get install swig` or `uv tool install swig`). Then:

```bash
uv sync --group report
uv run python -m unittest discover -s tests -v
```

## Training

```bash
uv run src/scripts/run_dqn.py -cfg experiments/dqn/cartpole.yaml --eval_interval 2500
uv run src/scripts/run_dqn.py -cfg experiments/dqn/lunarlander.yaml
uv run src/scripts/run_dqn.py -cfg experiments/dqn/mspacman.yaml
MUJOCO_GL=disable uv run src/scripts/run_sac.py -cfg experiments/sac/sanity_invertedpendulum.yaml
MUJOCO_GL=disable uv run src/scripts/run_sac.py -cfg experiments/sac/halfcheetah.yaml
```

Other configurations under `experiments/sac/` run automatic temperature tuning and Hopper critic comparisons. Local CSV/pickle logs and checkpoints are written to `exp/`. W&B is disabled by default; enable it with `--wandb_mode offline` or `--wandb_mode online`.

For Slurm, run from this directory:

```bash
mkdir -p jobs/logs
export HW3_GIT_COMMIT=$(git rev-parse HEAD)
sbatch --array=0-1 jobs/run.sbatch jobs/sanity.json
sbatch --array=0-4 jobs/run.sbatch jobs/cpu.json
sbatch --gres=gpu:1 --mem=32G --array=0-4 jobs/run.sbatch jobs/gpu.json
```

Adjust the resource directives for your cluster. The GPU manifest requests one GPU for each array task. Training disables rendering; use a working EGL or OSMesa backend when generating GIFs.

## Analysis and submission

After all required runs finish:

```bash
uv run --group report src/scripts/analyze_results.py
uv run --group report src/scripts/build_report.py
uv run src/scripts/package_submission.py
```

The analysis requires complete runs and rejects nonfinite metrics. It saves the assignment plots, compact curves, and run provenance under `results/`. The report builder writes `results/report.pdf`, its Markdown source, and the project README. The submission builder validates the required logs and checkpoints, ZIP integrity, root layout, and the 100 MB cap.

Submit `submit.zip` to HW3 Code and `results/report.pdf` to HW3 Report. The archive includes final checkpoints; separate best checkpoints are retained locally for visualizations. Raw run outputs and local secrets are excluded from Git.
