import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml


RESULTS = Path('results')


def required_configs():
    tasks = []
    for name in ('sanity', 'cpu', 'gpu'):
        tasks.extend(json.loads(Path(f'jobs/{name}.json').read_text()))
    return {yaml.safe_load(Path(t['config']).read_text())['exp_name'] + ':' + yaml.safe_load(Path(t['config']).read_text())['env_name']: t for t in tasks}


def load_runs(require_complete=True):
    runs = {}
    for folder in sorted(Path('exp').glob('*')):
        metadata_file = folder / 'experiment.json'
        log_file = folder / 'log.csv'
        if not metadata_file.exists() or not log_file.exists():
            continue
        metadata = json.loads(metadata_file.read_text())
        config = metadata['config']
        rows = list(csv.DictReader(log_file.open()))
        evaluation = [r for r in rows if r.get('Eval_AverageReturn')]
        if not evaluation:
            continue
        total = metadata['arguments'].get('total_steps') or config['total_steps']
        complete = int(evaluation[-1]['step']) == total - 1
        if require_complete and not complete:
            continue
        for row in rows:
            for key in ('Eval_AverageReturn', 'critic_loss', 'actor_loss', 'q_values', 'temperature'):
                if row.get(key) and not np.isfinite(float(row[key])):
                    raise ValueError(f'Nonfinite {key} in {folder}')
        key = config['exp_name'] + ':' + config['env_name']
        runs[key] = {'folder': folder, 'metadata': metadata, 'rows': rows, 'evaluation': evaluation, 'complete': complete}
    missing = set(required_configs()) - set(runs)
    if require_complete and missing:
        raise RuntimeError('Missing completed runs: ' + ', '.join(sorted(missing)))
    return runs


def series(run, metric):
    rows = [r for r in run['rows'] if r.get(metric)]
    return np.array([int(r['step']) for r in rows]), np.array([float(r[metric]) for r in rows])


def line(ax, run, label, metric='Eval_AverageReturn'):
    x, y = series(run, metric)
    ax.plot(x, y, label=label, linewidth=1.6)
    ax.set_xlabel('Environment steps')
    ax.grid(alpha=.2)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(RESULTS / f'{name}.png', dpi=180)
    fig.savefig(RESULTS / f'{name}.pdf')
    plt.close(fig)


def plot_results(runs):
    RESULTS.mkdir(exist_ok=True)
    for environment, name in [('CartPole-v1', 'cartpole'), ('LunarLander-v2', 'lunarlander')]:
        key = 'dqn:' + environment
        if key in runs:
            fig, ax = plt.subplots(figsize=(8, 4))
            line(ax, runs[key], 'Evaluation mean (10 episodes)')
            ax.set(title=environment + ': DQN', ylabel='Return')
            ax.legend()
            save(fig, name)
    if 'dqn:MsPacmanNoFrameskip-v4' in runs:
        run = runs['dqn:MsPacmanNoFrameskip-v4']
        fig, ax = plt.subplots(figsize=(8, 4))
        x, y = series(run, 'Train_EpisodeReturn')
        rolling = np.array([y[max(0, i - 19):i + 1].mean() for i in range(len(y))])
        ax.plot(x, rolling, alpha=.8, label='Training return (20-episode rolling mean)')
        line(ax, run, 'Evaluation mean (10 episodes)')
        ax.set(title='MsPacman: double DQN', ylabel='Raw game return')
        ax.legend()
        save(fig, 'mspacman')
    fig, ax = plt.subplots(figsize=(8, 4))
    for period in (250, 500, 1000, 2000):
        key = ('dqn' if period == 1000 else f'dqn_target{period}') + ':LunarLander-v2'
        if key in runs:
            line(ax, runs[key], f'Target copy every {period} steps')
    if ax.lines:
        ax.set(title='LunarLander: target-network update frequency', ylabel='Evaluation return')
        ax.legend()
        save(fig, 'lunarlander_sweep')
    else:
        plt.close(fig)
    if all(k + ':HalfCheetah-v4' in runs for k in ('sac', 'sac_autotune')):
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        for key, label in [('sac', 'Fixed temperature (0.1)'), ('sac_autotune', 'Automatic temperature')]:
            line(axes[0], runs[key + ':HalfCheetah-v4'], label)
        line(axes[1], runs['sac_autotune:HalfCheetah-v4'], 'Learned temperature', metric='temperature')
        axes[0].set(title='HalfCheetah: fixed vs. automatic temperature', ylabel='Evaluation return')
        axes[1].set_ylabel('Temperature (alpha)')
        axes[0].legend(); axes[1].legend()
        save(fig, 'halfcheetah_temperature')
    if all(k + ':Hopper-v4' in runs for k in ('sac_singleq', 'sac_clipq')):
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        for key, label in [('sac_singleq', 'Single Q'), ('sac_clipq', 'Clipped double Q')]:
            run = runs[key + ':Hopper-v4']
            line(axes[0], run, label)
            line(axes[1], run, label, metric='q_values')
        axes[0].set(title='Hopper: critic comparison', ylabel='Evaluation return')
        axes[1].set_ylabel('Mean sampled Q-value')
        axes[0].legend(); axes[1].legend()
        save(fig, 'hopper_critics')
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, label in [('sac', 'Fixed temperature'), ('sac_autotune', 'Automatic temperature')]:
        key = name + ':InvertedPendulum-v4'
        if key in runs:
            line(ax, runs[key], label)
    if ax.lines:
        ax.set(title='InvertedPendulum: SAC sanity checks', ylabel='Evaluation return')
        ax.legend()
        save(fig, 'invertedpendulum')
    else:
        plt.close(fig)


def summarize(runs):
    summary = {}
    curves = RESULTS / 'curves'
    curves.mkdir(exist_ok=True)
    for key, run in runs.items():
        config = run['metadata']['config']
        evaluation = run['evaluation']
        values = [float(r['Eval_AverageReturn']) for r in evaluation]
        name = run['folder'].name
        summary[key] = {
            'directory': str(run['folder']), 'config': config,
            'seed': run['metadata']['arguments']['seed'],
            'git_commit': run['metadata']['git_commit'],
            'complete': run['complete'], 'last_step': int(evaluation[-1]['step']),
            'best_eval_mean': max(values), 'final_eval_mean': values[-1],
            'final_eval_std': float(evaluation[-1]['Eval_StdReturn']),
            'best_eval_episode': max(float(r['Eval_MaxReturn']) for r in evaluation),
        }
        q_values = [float(r['q_values']) for r in run['rows'] if r.get('q_values')]
        if q_values:
            summary[key]['final_q_values'] = q_values[-1]
            summary[key]['peak_q_values'] = max(q_values)
        temperature = [float(r['temperature']) for r in run['rows'] if r.get('temperature')]
        if temperature:
            summary[key]['initial_logged_temperature'] = temperature[0]
            summary[key]['final_temperature'] = temperature[-1]
            tail = temperature[-max(2, len(temperature) // 10):]
            summary[key]['temperature_tail_range'] = [min(tail), max(tail)]
        columns = ['step', 'Eval_AverageReturn', 'Eval_StdReturn', 'Train_EpisodeReturn', 'q_values', 'temperature', 'entropy']
        with (curves / f'{name}.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(run['rows'])
    (RESULTS / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: {'best': v['best_eval_mean'], 'final': v['final_eval_mean'], 'complete': v['complete']} for k, v in summary.items()}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    runs = load_runs(require_complete=not args.partial)
    plot_results(runs)
    summarize(runs)


if __name__ == '__main__':
    main()
