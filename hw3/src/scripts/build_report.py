import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, KeepTogether

from scripts.analyze_results import load_runs, plot_results, summarize
from scripts.render_gallery import gallery_markdown


def sections(summary):
    def result(key):
        run = summary[key]
        return f"best evaluation mean {run['best_eval_mean']:.1f}, final mean {run['final_eval_mean']:.1f} (episode standard deviation {run['final_eval_std']:.1f})"

    fixed = summary['sac:HalfCheetah-v4']
    auto = summary['sac_autotune:HalfCheetah-v4']
    single = summary['sac_singleq:Hopper-v4']
    clipped = summary['sac_clipq:Hopper-v4']
    sweep = [(period, summary[('dqn' if period == 1000 else f'dqn_target{period}') + ':LunarLander-v2']) for period in (250, 500, 1000, 2000)]
    best_period, _ = max(sweep, key=lambda pair: pair[1]['final_eval_mean'])
    direction = 'decreased' if auto['final_temperature'] < auto['initial_logged_temperature'] else 'increased'
    tail_min, tail_max = auto['temperature_tail_range']
    temperature_trend = 'was nearly stable within a 10% relative range' if (tail_max - tail_min) / max(tail_max, 1e-12) < .1 else 'continued to vary by more than 10% of its tail maximum'
    if single['final_q_values'] > clipped['final_q_values'] and single['final_eval_mean'] < clipped['final_eval_mean']:
        bias_observation = 'Single Q assigns larger values despite producing lower final returns. This ordering is consistent with optimistic value estimates, while clipped targets provide a more conservative estimate.'
    else:
        bias_observation = 'The observed Q-value and return ordering does not by itself establish overestimation. Clipped targets change both value estimates and the policies being trained.'
    return [
        ('Setup and evaluation', [
            'Implemented the supplied Berkeley CS185/285 Spring 2026 HW3 starter. DQN uses replay, MSE Bellman regression, epsilon-greedy exploration, gradient clipping when configured, and hard target copies. SAC uses a tanh-transformed Gaussian policy, reparameterized gradients, entropy-augmented targets, Polyak updates, and optional learned temperature and clipped double-Q.',
            'All main comparisons use seed 1 and the supplied training budgets and hyperparameters. Training, evaluation, and rendering environments receive distinct deterministic seeds. Evaluation means use 10 episodes. SAC evaluation samples the policy, matching the starter evaluation interface. Curves are raw evaluation means; the Atari training curve alone uses a trailing 20-episode mean.',
            'Best means are maxima across evaluation checkpoints, not final performance or averages across seeds. Episode standard deviations describe episodes within one evaluation, not uncertainty across training seeds. Results are one-seed assignment experiments; they cannot establish statistical significance. The earlier unseeded-environment CartPole development run is excluded from the final comparisons.',
        ], None),
        ('2.4 Basic DQN: CartPole', [
            'CartPole-v1: ' + result('dqn:CartPole-v1') + f". The highest individual evaluation episode returned {summary['dqn:CartPole-v1']['best_eval_episode']:.0f}, against the assignment target of 500.",
            'The critic predicts one value per discrete action. Its target is r + gamma * (1 - terminal) * max_a Q_target(s_next, a). Gradients apply only to the online prediction for the recorded action. Time-limit truncations retain the bootstrap term. A separate target network prevents the supervised target from changing on every optimizer step.',
        ], 'cartpole'),
        ('2.5 Double DQN: LunarLander and MsPacman', [
            'LunarLander-v2: ' + result('dqn:LunarLander-v2') + '. MsPacman: ' + result('dqn:MsPacmanNoFrameskip-v4') + '.',
            'Double DQN selects the next action with the online network and evaluates it with the target network. This reduces the coupling between selecting a noisy maximum and using that same estimate as a target.',
            'The Atari experiment uses MsPacmanNoFrameskip-v4, as specified by the supplied YAML, with four-frame stacking, 84 x 84 grayscale observations, and action repeat 4. Steps on the plot are agent environment steps after preprocessing, not individual emulator frames.',
            'Training uses the decaying epsilon-greedy policy; evaluation uses epsilon zero. Early exploratory actions often reduce training scores even when the greedy policy has found useful behavior, so the training and evaluation curves need not match. Both curves use raw game rewards: the supplied Atari wrapper records base-environment episode statistics and does not enable reward clipping. Episode sampling and the training-only rolling average also affect the visual difference.',
        ], 'lunarlander'),
        ('MsPacman: training and evaluation', [], 'mspacman'),
        ('2.6 LunarLander hyperparameter sensitivity', [
            'Swept hard target-copy periods of 250, 500, 1000 (baseline), and 2000 steps, holding the other configuration values and seed fixed. The parameter controls a useful tradeoff: frequent copies track new estimates quickly, while longer intervals give more stable but older bootstrap targets.',
            'Observed best/final evaluation means: ' + '; '.join(f"{period} steps: {run['best_eval_mean']:.1f}/{run['final_eval_mean']:.1f}" for period, run in sweep) + f'. The highest final mean in this sweep used a period of {best_period} steps.',
            'These are four settings of one parameter rather than a multi-parameter search. A single seed can produce different conclusions from a repeated-seed study, and equal seeds do not make the changing data distributions identical.',
        ], 'lunarlander_sweep'),
        ('3.2-3.4 SAC implementation and sanity checks', [
            'InvertedPendulum fixed temperature: ' + result('sac:InvertedPendulum-v4') + '. Automatic temperature: ' + result('sac_autotune:InvertedPendulum-v4') + '.',
            'The actor uses rsample() so the Q objective differentiates through sampled actions. TanhTransform includes the density correction, and Independent sums log probability across action dimensions. Entropy is estimated with negative sampled log probability. Critic weights are frozen during actor optimization while action gradients remain enabled. Target calculations run without gradients.',
            'With entropy enabled, the critic target is r + gamma * (1 - terminal) * [Q_target(s_next, a_next) - alpha * log pi(a_next | s_next)]. The actor minimizes mean(alpha * log pi(a | s) - Q(s, a)). The ensemble actor objective averages current critics; the clipped variant takes a minimum for the target backup.',
        ], 'invertedpendulum'),
        ('3.4-3.5 HalfCheetah: fixed and learned temperature', [
            'Fixed temperature: ' + result('sac:HalfCheetah-v4') + '. Automatic temperature: ' + result('sac_autotune:HalfCheetah-v4') + '.',
            f"Automatic tuning produced a {'higher' if auto['final_eval_mean'] > fixed['final_eval_mean'] else 'lower'} final evaluation mean in this run. Its final mean differs from the fixed run by {auto['final_eval_mean'] - fixed['final_eval_mean']:+.1f}; its best mean differs by {auto['best_eval_mean'] - fixed['best_eval_mean']:+.1f}. This comparison describes the observed run, not a statistically established improvement.",
            f"Temperature first fell to {auto['minimum_temperature']:.4f} near step {auto['minimum_temperature_step']}, then recovered. Overall it {direction} from the first logged value {auto['initial_logged_temperature']:.4f} to {auto['final_temperature']:.4f}. The implementation learns log(alpha), starts alpha at 0.1, and uses target entropy -6 for HalfCheetah's six actions. Its loss is -mean(log(alpha) * stop_gradient(log pi + target_entropy)).",
            f"Across the final 10% of logged updates, temperature ranged from {auto['temperature_tail_range'][0]:.4f} to {auto['temperature_tail_range'][1]:.4f}; temperature {temperature_trend} near the end.",
            f"Initial logged entropy was {auto['initial_logged_entropy']:.2f}, compared with target -6; final logged entropy was {auto['final_entropy']:.2f}. Early diffuse actions exceed the target entropy, so the dual update reduces temperature. As reward learning concentrates the policy around useful locomotion actions, entropy can fall below the constraint and temperature increases to counter further loss of exploration. This explains the observed dip and recovery; temperature responds to the entropy residual, not return directly. A fixed value of 0.1 is already tuned in the starter, so automatic tuning need not outperform it.",
            'The optional fixed-temperature sweep was not run; the required fixed-versus-automatic comparison is shown below.',
        ], 'halfcheetah_temperature'),
        ('3.6 Hopper: single Q and clipped double Q', [
            'Single Q: ' + result('sac_singleq:Hopper-v4') + '. Clipped double Q: ' + result('sac_clipq:Hopper-v4') + f". The clipped run's highest individual evaluation episode returned {clipped['best_eval_episode']:.1f}; the assignment asks for at least 1500 at least once.",
            f"{'Clipped double Q' if clipped['final_eval_mean'] > single['final_eval_mean'] else 'Single Q'} has the higher final mean in this comparison. Clipped double Q changes the final mean by {clipped['final_eval_mean'] - single['final_eval_mean']:+.1f} and the best mean by {clipped['best_eval_mean'] - single['best_eval_mean']:+.1f} relative to single Q in these runs.",
            f"Final logged mean Q-values were {single['final_q_values']:.1f} for single Q and {clipped['final_q_values']:.1f} for clipped double Q. Peak logged means were {single['peak_q_values']:.1f} and {clipped['peak_q_values']:.1f}, respectively. These measured scales should be read together with the return curves below.",
            bias_observation,
            'Taking the smaller of two target critics counters optimistic errors that can compound under bootstrapping. The lower panel plots the mean current Q-value on sampled replay batches. Comparing Q-values with learning stability helps diagnose overestimation, but these values are not a direct measurement of bias: they include entropy-related value, depend on the state distribution, and are not paired with ground-truth returns from identical state-action pairs.',
        ], 'hopper_critics'),
        ('Validation and reproducibility', [
            'Numerical unit tests check Bellman targets, terminal masking, action indexing, double-DQN selection/evaluation, detached targets, target-copy timing, actor gradient flow, critic isolation, temperature-update direction, differentiable bounded-policy entropy, and Polyak interpolation.',
            'Each selected run contains agent.pt, flags.json, log.csv, log.pkl, experiment.json, and a separate best_agent.pt for visualizations. The submitted agent.pt is the final checkpoint; plots include every logged evaluation rather than only the best checkpoint. experiment.json records configuration, arguments, source revision, device, and Torch version. Raw logs and checkpoints remain local; compact curves and summaries are versioned.',
            'Reproduction commands and resource requests are in the README and jobs/ manifests. The submission builder validates complete runs, required files, archive root layout, and the 100 MB size cap. No external tracking account is required. Discord notifications read a local ignored environment file.',
        ], None),
    ]


def write_readme(summary):
    entries = [('CartPole-v1', 'dqn'), ('LunarLander-v2', 'dqn'), ('MsPacmanNoFrameskip-v4', 'dqn'), ('HalfCheetah-v4', 'sac'), ('HalfCheetah-v4', 'sac_autotune'), ('Hopper-v4', 'sac_singleq'), ('Hopper-v4', 'sac_clipq')]
    labels = {'sac': 'SAC, fixed temperature', 'sac_autotune': 'SAC, learned temperature', 'sac_singleq': 'SAC, single Q', 'sac_clipq': 'SAC, clipped double Q'}
    table_rows = []
    for environment, name in entries:
        run = summary[name + ':' + environment]
        label = labels.get(name, 'Double DQN' if run['config'].get('use_double_q') else 'DQN')
        table_rows.append(f"| {environment.replace('NoFrameskip-v4', '')} | {label} | {run['best_eval_mean']:.1f} | {run['final_eval_mean']:.1f} |")
    table = '\n'.join(table_rows)
    text = '''# Deep RL Control Benchmarks

DQN and Soft Actor-Critic implementations evaluated on Atari, LunarLander, and MuJoCo control tasks. This project completes the Berkeley CS185/285 Spring 2026 HW3 starter and studies how target updates, entropy temperature, and critic ensembles affect learning.

## Results

All numbers below are means over 10 evaluation episodes from seed 1. Best means are selected across evaluation checkpoints; final means expose late-training regressions. These are single-seed experiments, not claims of statistical significance.

| Environment | Method | Best mean return | Final mean return |
| --- | --- | ---: | ---: |
''' + table + '''

![HalfCheetah temperature comparison](hw3/results/halfcheetah_temperature.png)

![Hopper critic comparison](hw3/results/hopper_critics.png)

The [PDF report](hw3/results/report.pdf) covers all required experiments and their limitations. [Machine-readable results](hw3/results/summary.json) and [training curves](hw3/results/curves) preserve the measurements behind the figures. Additional plots cover [CartPole](hw3/results/cartpole.png), [LunarLander](hw3/results/lunarlander.png), [the target-update sweep](hw3/results/lunarlander_sweep.png), and [MsPacman training versus evaluation](hw3/results/mspacman.png).

## Implementation

- DQN with epsilon-greedy exploration, replay, hard target updates, and double-DQN action selection.
- SAC with reparameterized tanh-Gaussian actions, entropy-augmented Bellman targets, and Polyak target updates.
- Learned temperature using a positive log parameterization and a separate optimizer.
- Single-critic and clipped double-critic backups, with numerical tests for gradient flow and update semantics.
- Reproducible Slurm CPU/GPU jobs, explicit environment seeds, local logging, source-revision metadata, and best/final checkpoints.

The [DQN agent](hw3/src/agents/dqn_agent.py) and [SAC agent](hw3/src/agents/sac_agent.py) contain the algorithm updates. Training entry points are in [scripts](hw3/src/scripts); [YAML configurations](hw3/experiments) specify the experiments. Experiments evaluate model-free agents in simulated control benchmarks.

## Run locally

Requires Python 3.10 and SWIG for Box2D. On Debian/Ubuntu, install SWIG with `sudo apt-get install swig`, or use `uv tool install swig` in a user environment.

```bash
cd hw3
uv sync --group report
uv run python -m unittest discover -s tests -v
uv run src/scripts/run_dqn.py -cfg experiments/dqn/cartpole.yaml --eval_interval 2500
MUJOCO_GL=disable uv run src/scripts/run_sac.py -cfg experiments/sac/sanity_invertedpendulum.yaml
```

Training defaults to local logging with W&B disabled. Use `--wandb_mode offline` or `--wandb_mode online` to opt in. `--seed`, `--num_threads`, and `--total_steps` are available for reproducible runs and smoke checks. MuJoCo training does not require graphics; checkpoint rendering uses a working EGL or OSMesa installation.

## Run on Slurm

From `hw3/`, with the environment already installed:

```bash
mkdir -p jobs/logs
export HW3_GIT_COMMIT=$(git rev-parse HEAD)
sbatch --array=0-1 jobs/run.sbatch jobs/sanity.json
sbatch --array=0-4 jobs/run.sbatch jobs/cpu.json
sbatch --gres=gpu:1 --mem=32G --array=0-4 jobs/run.sbatch jobs/gpu.json
```

The example script uses the cluster's `short` partition, two CPUs per task, one Torch thread, and a six-hour limit. Adjust partition, account, memory, and time for your cluster. Each GPU array task requests its own GPU. The four LunarLander settings change only the target-copy period: 250, 500, 1000, or 2000 steps.

Optional Discord notifications use `DISCORD_WEBHOOK` from a local `.env` in the repository root. Secrets, virtual environments, raw experiment outputs, checkpoints, and Slurm logs are ignored by Git.

## Rebuild the deliverables

```bash
cd hw3
MUJOCO_GL=egl uv run src/scripts/render_gallery.py
uv run --group report src/scripts/analyze_results.py
uv run --group report src/scripts/build_report.py
uv run src/scripts/package_submission.py
```

These commands require completed experiment logs in `hw3/exp/`. The report and summary record selected run directories. The archive contains `exp/` and `src/` at its root and is checked against the assignment's 100 MB limit. Upload `hw3/submit.zip` and `hw3/results/report.pdf` separately to the assignment's code and report destinations.

## Attribution and scope

Based on [Berkeley Deep RL HW3, Spring 2026](https://github.com/berkeleydeeprlcourse/homework_spring2026/tree/main/hw3). Network, distribution, wrapper, and infrastructure scaffolding originate from that starter. This repository adds the missing algorithm updates, experiment execution, tests, analysis, and deliverables. It is a learning and evaluation project; results should be repeated across seeds before drawing research or deployment conclusions.
'''
    if Path('results/media/halfcheetah.gif').exists():
        marker = '## Results'
        text = text.replace(marker, '![HalfCheetah best-checkpoint rollout](hw3/results/media/halfcheetah.gif)\n\nBest-checkpoint visualization; quantitative results below use separate evaluation episodes.\n\n' + marker, 1)
    text = text.replace('## Implementation', gallery_markdown() + '\n## Implementation', 1)
    Path('../README.md').write_text(text)


def main():
    runs = load_runs()
    plot_results(runs)
    summary = summarize(runs)
    content = sections(summary)
    markdown = ['# HW3: Q-Learning and Actor-Critic Algorithms', '']
    styles = getSampleStyleSheet()
    styles['BodyText'].fontSize = 10
    styles['BodyText'].leading = 13
    styles['Heading2'].keepWithNext = True
    story = [Paragraph('HW3: Q-Learning and Actor-Critic Algorithms', styles['Title']), Spacer(1, .2 * inch)]
    captions = {
        'cartpole': 'Evaluation mean over 10 episodes versus environment steps; basic DQN.',
        'lunarlander': 'Double-DQN evaluation mean over 10 episodes; default LunarLander configuration.',
        'mspacman': 'Raw game returns: evaluation mean over 10 episodes and trailing 20-episode training mean. Training includes epsilon-greedy exploration.',
        'lunarlander_sweep': 'Target-copy frequency was chosen because it controls the speed/stability tradeoff of bootstrap targets. Four periods are compared with all other configuration values fixed.',
        'invertedpendulum': 'Full SAC sanity-check curves with fixed and automatically tuned temperature.',
        'halfcheetah_temperature': 'Top: fixed versus automatic temperature evaluation means. Bottom: learned temperature from the automatic run.',
        'hopper_critics': 'Top: evaluation means for single and clipped double Q. Bottom: logged mean current Q-values on replay batches; these are diagnostics, not direct estimates of bias.',
    }
    for title, paragraphs, figure in content:
        markdown.extend(['## ' + title, ''])
        story.append(Paragraph(escape(title), styles['Heading2']))
        for text in paragraphs:
            markdown.extend([text, ''])
            story.extend([Paragraph(escape(text), styles['BodyText']), Spacer(1, .06 * inch)])
        if figure:
            image = Path('results') / (figure + '.png')
            markdown.extend([f'![{title}]({figure}.png)', ''])
            from PIL import Image as PILImage
            width, height = PILImage.open(image).size
            caption = captions[figure]
            markdown.extend(['*' + caption + '*', ''])
            story.append(KeepTogether([
                Image(str(image), width=6.3 * inch, height=6.3 * inch * height / width),
                Paragraph(escape(caption), styles['Italic']),
                Spacer(1, .1 * inch),
            ]))
    Path('results/report.md').write_text('\n'.join(markdown))
    def footer(canvas, doc):
        canvas.setFont('Helvetica', 8)
        canvas.drawRightString(7.5 * inch, .45 * inch, str(doc.page))
    SimpleDocTemplate('results/report.pdf', rightMargin=.75 * inch, leftMargin=.75 * inch, topMargin=.65 * inch, bottomMargin=.65 * inch).build(story, onFirstPage=footer, onLaterPages=footer)
    write_readme(summary)
    print('Wrote results/report.pdf, results/report.md, and ../README.md')


if __name__ == '__main__':
    main()
