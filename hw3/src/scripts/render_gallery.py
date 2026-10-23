import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


DEMOS = [
    ('dqn:CartPole-v1', 'cartpole', 'DQN · CartPole', 500),
    ('dqn:LunarLander-v2', 'lunarlander', 'Double DQN · LunarLander', 1000),
    ('dqn:MsPacmanNoFrameskip-v4', 'mspacman', 'Double DQN · MsPacman', 600),
    ('sac:HalfCheetah-v4', 'halfcheetah', 'SAC · Fixed temperature', 300),
    ('sac_autotune:HalfCheetah-v4', 'halfcheetah_autotune', 'SAC · Learned temperature', 300),
    ('sac_singleq:Hopper-v4', 'hopper_singleq', 'Hopper · Single Q', 600),
    ('sac_clipq:Hopper-v4', 'hopper', 'Hopper · Clipped double Q', 600),
    ('sac:InvertedPendulum-v4', 'invertedpendulum_fixed', 'Pendulum · Fixed temperature', 600),
    ('sac_autotune:InvertedPendulum-v4', 'invertedpendulum', 'Pendulum · Learned temperature', 600),
    ('dqn_target250:LunarLander-v2', 'lunarlander_target250', 'LunarLander · Target period 250', 1000),
    ('dqn_target500:LunarLander-v2', 'lunarlander_target500', 'LunarLander · Target period 500', 1000),
    ('dqn_target2000:LunarLander-v2', 'lunarlander_target2000', 'LunarLander · Target period 2000', 1000),
]


def gallery_markdown(prefix='hw3/results/media/'):
    lines = [
        '## Policy demonstrations', '',
        'Every trained configuration is shown below using its best checkpoint and visualization seed 10001. These individual rollouts illustrate behavior; evaluation averages are reported separately. Each GIF has a JSON file recording its checkpoint, seed, length, and return.', '',
    ]
    def cards(demos, columns):
        lines.append('<table>')
        for i in range(0, len(demos), columns):
            lines.append('<tr>')
            for _, name, label, _ in demos[i:i + columns]:
                width = 250 if columns == 3 else 320
                lines.append(f'<td width="{100 // columns}%"><strong>{label}</strong><br><img src="{prefix}{name}.gif" width="{width}" alt="{label} policy rollout"></td>')
            lines.append('</tr>')
        lines.extend(['</table>', ''])

    for title, demos, columns in [
        ('Discrete action control', DEMOS[:3], 3),
        ('HalfCheetah: fixed versus learned temperature', DEMOS[3:5], 2),
        ('Hopper: single versus clipped double Q', DEMOS[5:7], 2),
    ]:
        lines.extend(['### ' + title, ''])
        cards(demos, columns)
    lines.extend(['### SAC sanity checks', ''])
    cards(DEMOS[7:9], 2)
    lines.extend(['### LunarLander target-update sweep', ''])
    cards(DEMOS[9:], 3)
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    summary = json.loads(Path('results/summary.json').read_text())
    environment = {**os.environ, 'MUJOCO_GL': os.environ.get('MUJOCO_GL', 'egl')}
    for key, name, _, steps in DEMOS:
        output = Path(f'results/media/{name}.gif')
        if not args.overwrite and output.exists() and output.with_suffix('.json').exists():
            cached = json.loads(output.with_suffix('.json').read_text())
            checkpoint = Path(summary[key]['directory']) / 'best_agent.pt'
            if (cached['run'] == summary[key]['directory'] and cached.get('seed') == 10001
                    and cached.get('checkpoint') == 'best_agent.pt'
                    and output.stat().st_mtime >= checkpoint.stat().st_mtime):
                continue
        subprocess.run([
            sys.executable, 'src/scripts/render_policy.py', summary[key]['directory'],
            '--output', str(output), '--max_steps', str(steps), '--seed', '10001',
        ], env=environment, check=True)
    Path('results/media/README.md').write_text(gallery_markdown(prefix=''))
    print(f'Gallery ready: {len(DEMOS)} configurations')


if __name__ == '__main__':
    main()
