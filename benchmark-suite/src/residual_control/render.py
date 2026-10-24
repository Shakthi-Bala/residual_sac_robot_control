import argparse
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from infrastructure import pytorch_util as ptu
from residual_control.artifacts import save_json
from residual_control.env import ROOT, Scenario, UR16eTrackingEnv
from residual_control.evaluate import load_policy
from residual_control.protocol import CONDITIONS


DEMOS = [('nominal_reach', 'nominal', 'reach', 'best'),
         ('combined_reach', 'combined', 'reach', 'best'),
         ('combined_tracking', 'combined', 'tracking', 'best'),
         ('ood_payload_tracking', 'ood_payload', 'tracking', 'best'),
         ('ood_delay_tracking', 'ood_delay', 'tracking', 'best'),
         ('final_tracking', 'combined', 'tracking', 'final')]


def font(size):
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    ptu.init_gpu(use_gpu=False)
    methods = ['controller', 'standalone', 'residual']
    labels = ['Cartesian PI', 'Standalone SAC', 'Residual SAC']
    controller = json.loads((ROOT / 'results/controller_tuning.json').read_text())['selected']
    loaded = {checkpoint: [load_policy(method, args.seed, checkpoint) for method in methods[1:]]
              for checkpoint in ['best', 'final']}
    policies = {checkpoint: [None] + [item[0] for item in entries] for checkpoint, entries in loaded.items()}
    checkpoint_steps = {checkpoint: {method: item[1]['best_validation_step' if checkpoint == 'best' else 'training_interactions']
                                    for method, item in zip(methods[1:], entries)}
                        for checkpoint, entries in loaded.items()}
    environments = [UR16eTrackingEnv(method, **controller) for method in methods]
    folder = ROOT / 'results/media'
    folder.mkdir(parents=True, exist_ok=True)
    for name, condition, task, checkpoint in DEMOS:
        agents = policies[checkpoint]
        payload, delay = CONDITIONS[condition]
        case = Scenario(20000, task, payload, delay)
        observations = [env.reset(scenario=case) for env in environments]
        frames, finished = [], [False] * 3
        for step in range(environments[0].horizon):
            for i, (env, agent) in enumerate(zip(environments, agents)):
                if not finished[i]:
                    action = agent.deterministic_action(observations[i]) if agent else np.zeros(6)
                    observations[i], _, finished[i], _ = env.step(action)
            if step % 5:
                continue
            canvas = Image.new('RGB', (1440, 448), '#101b2d')
            draw = ImageDraw.Draw(canvas)
            for i, (env, label) in enumerate(zip(environments, labels)):
                canvas.paste(Image.fromarray(env.render()), (480 * i, 60))
                draw.text((480 * i + 16, 32), label, font=font(20), fill='white')
                error = env.trace[-1]['error'] * 1000
                state = ' | terminated' if env.failures else ''
                draw.text((480 * i + 16, 426), f'error {error:.1f} mm{state}', font=font(14), fill='white')
            draw.text((16, 6), f'{checkpoint} checkpoint | {task} | +{payload:g} kg | {delay * 20} ms | {(step + 1) * .02:.2f} s',
                      font=font(16), fill='#a9d7ff')
            frames.append(canvas.resize((1080, 336)))
        path = folder / f'{name}.gif'
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=False)
        save_json(path.with_suffix('.json'), {'target_seed': 20000, 'training_seed': args.seed,
                  'selection': 'fixed seed 1 and target 20000; best validation or final checkpoint as labeled; not selected for test performance',
                  'checkpoint': checkpoint, 'checkpoint_steps': checkpoint_steps[checkpoint], 'task': task, 'condition': condition, 'payload_kg': payload, 'delay_ms': delay * 20,
                  'playback': '10 fps, real time', 'metrics': {m: e.metrics() for m, e in zip(methods, environments)}})
        print(path, flush=True)
    for env in environments:
        env.close()


if __name__ == '__main__':
    main()
