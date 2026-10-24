import argparse
import json
from pathlib import Path

import numpy as np
import torch

from agents.dqn_agent import DQNAgent
from agents.sac_agent import SoftActorCritic
from infrastructure import pytorch_util as ptu
from scripts.run_dqn import make_config as make_dqn_config
from scripts.run_sac import make_config as make_sac_config
from scripts.gif_style import policy_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max_steps', type=int, default=600)
    parser.add_argument('--seed', type=int, default=10001)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ptu.init_gpu(use_gpu=False)
    metadata = json.loads((args.run / 'experiment.json').read_text())
    config_file = metadata['arguments']['config_file']
    dqn = 'dqn' in config_file
    config = (make_dqn_config if dqn else make_sac_config)(config_file)
    env = config['make_env'](eval=True, render=True)
    if dqn:
        agent = DQNAgent(env.observation_space.shape, env.action_space.n, **config['agent_kwargs'])
    else:
        agent = SoftActorCritic(env.observation_space.shape, env.action_space.shape[0], **config['agent_kwargs'])
    agent.load_state_dict(torch.load(args.run / 'best_agent.pt', map_location='cpu', weights_only=True))
    agent.eval()
    observation = np.asarray(env.reset(seed=args.seed))
    frames = []
    total_reward = 0.
    for step in range(args.max_steps):
        frame = env.render(mode="rgb_array") if hasattr(env.unwrapped, "ale") else env.render()
        if isinstance(frame, list):
            frame = frame[-1]
        frames.append(frame)
        observation, reward, done, _ = env.step(agent.get_action(np.asarray(observation)))
        total_reward += reward
        if done:
            break
    env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fps = env.metadata.get('render_fps') or (15 if hasattr(env.unwrapped, 'ale') else 30)
    details = {
        'run': str(args.run), 'checkpoint': 'best_agent.pt',
        'seed': args.seed, 'steps': step + 1, 'return': float(total_reward),
        'note': 'One best-checkpoint visualization rollout; not an evaluation average.',
    }
    details['presentation'] = 'light comparison cards, navy header, full rollout return'
    from scripts.render_gallery import DEMOS
    label = next((row[2] for row in DEMOS if row[1] == args.output.stem), args.output.stem)
    from PIL import Image
    styled = [policy_frame(Image.fromarray(frame), label, details) for frame in frames[::2]]
    styled[0].save(args.output, save_all=True, append_images=styled[1:], duration=round(2000 / fps / 10) * 10, loop=0)
    args.output.with_suffix('.json').write_text(json.dumps(details, indent=2) + '\n')
    print(f'{args.output}: {step + 1} steps, return {total_reward:.1f}; best checkpoint, visualization seed {args.seed}')


if __name__ == '__main__':
    main()
