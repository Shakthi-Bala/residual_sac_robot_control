import argparse
import csv
import json
from dataclasses import asdict
import time

import numpy as np
import torch
import yaml

from infrastructure import pytorch_util as ptu
from infrastructure.replay_buffer import ReplayBuffer
from residual_control.agent import make_agent
from residual_control.artifacts import digest, notify, revision, save_json, save_weights
from residual_control.env import ROOT, Scenario, UR16eTrackingEnv
from residual_control.protocol import rollout, validation_cases


def validation(env, agent):
    metrics = [rollout(env, case, agent.deterministic_action) for case in validation_cases()]
    return {k: float(np.mean([row[k] for row in metrics])) for k in metrics[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['standalone', 'residual'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / 'configs/experiment.yaml').read_text())
    if args.steps:
        config['training_steps'] = args.steps
    if args.smoke:
        config.update(training_steps=400, random_steps=50, learning_starts=100, validation_interval=200)
    controller = json.loads((ROOT / 'results/controller_tuning.json').read_text())['selected']
    folder = ROOT / 'runs' / ('smoke' if args.smoke else 'main') / f'{args.method}_seed{args.seed}'
    if (folder / 'complete.json').exists():
        raise RuntimeError(f'Completed run already exists: {folder}')
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    exploration_rng = np.random.default_rng(args.seed + 1000)
    ptu.init_gpu(use_gpu=False)
    env = UR16eTrackingEnv(args.method, **controller)
    evaluation_env = UR16eTrackingEnv(args.method, **controller)
    agent = make_agent(env.observation_space.shape, config)
    replay = ReplayBuffer(capacity=config['training_steps'])
    metadata = {'method': args.method, 'seed': args.seed, 'config': config, 'controller': controller,
                'source_revision': revision(), 'model_sha256': digest(ROOT / 'models/ur16e.xml'),
                'validation_cases': [asdict(c) for c in validation_cases()], 'device': 'cpu',
                'evaluation_policy': 'tanh of Gaussian mean', 'torch_version': torch.__version__,
                'training_interactions': 0, 'validation_interactions': 0, 'complete': False}
    save_json(folder / 'metadata.json', metadata)
    val_file = (folder / 'validation.csv').open('w')
    val_writer = None
    train_file = (folder / 'training.csv').open('w')
    train_writer = None
    best_score = float('inf')
    best_step = 0
    threshold_step = None
    streak = 0
    began = time.monotonic()

    def evaluate(step):
        nonlocal val_writer, best_score, best_step, threshold_step, streak
        metrics = validation(evaluation_env, agent)
        metadata['validation_interactions'] += int(round(metrics['episode_steps'] * len(validation_cases())))
        row = {'step': step, **metrics}
        if val_writer is None:
            val_writer = csv.DictWriter(val_file, fieldnames=list(row), lineterminator='\n')
            val_writer.writeheader()
        val_writer.writerow(row)
        val_file.flush()
        score = metrics['tracking_rmse_m'] + .05 * (1. - metrics['success'])
        if score < best_score:
            best_score, best_step = score, step
            save_weights(folder / 'best.pt', agent.state_dict())
        passed = metrics['tracking_rmse_m'] <= config['tracking_rmse_threshold_m'] and metrics['success'] >= config['success_threshold']
        streak = streak + 1 if passed else 0
        if streak >= config['threshold_consecutive_evaluations'] and threshold_step is None:
            threshold_step = step
        print(json.dumps({'validation': row, 'best_step': best_step}), flush=True)

    def reset(step):
        task = 'reach' if step < config['reach_only_steps'] else str(rng.choice(['reach', 'tracking']))
        return env.reset(scenario=Scenario(int(rng.integers(1000000, 2 ** 31)), task,
            float(rng.uniform(*config['training_payload_kg'])),
            int(rng.integers(config['training_delay_steps'][0], config['training_delay_steps'][1] + 1))))

    evaluate(0)
    observation = reset(0)
    for step in range(1, config['training_steps'] + 1):
        action = exploration_rng.uniform(-1., 1., 6).astype(np.float32) if step <= config['random_steps'] else agent.get_action(observation)
        next_observation, reward, done, info = env.step(action)
        replay.insert(observation=observation, action=action, reward=reward,
                      next_observation=next_observation, done=done and not info['TimeLimit.truncated'])
        observation = reset(step) if done else next_observation
        if step >= config['learning_starts']:
            batch = ptu.from_numpy(replay.sample(config['batch_size']))
            stats = agent.update(batch['observations'], batch['actions'], batch['rewards'], batch['next_observations'], batch['dones'], step)
            if not all(np.isfinite(v) for v in stats.values()):
                raise FloatingPointError(f'Nonfinite learner metric at step {step}')
            if step % 1000 == 0 or step == config['training_steps']:
                row = {'step': step, **stats}
                if train_writer is None:
                    train_writer = csv.DictWriter(train_file, fieldnames=list(row), lineterminator='\n')
                    train_writer.writeheader()
                train_writer.writerow(row)
                train_file.flush()
        if step % config['validation_interval'] == 0 or step == config['training_steps']:
            evaluate(step)
            save_weights(folder / 'final.pt', agent.state_dict())
    metadata.update(training_interactions=config['training_steps'], complete=True,
                    wall_seconds=time.monotonic() - began, best_validation_step=best_step,
                    threshold_interactions=threshold_step)
    save_json(folder / 'metadata.json', metadata)
    save_json(folder / 'complete.json', {'complete': True, 'training_steps': config['training_steps']})
    train_file.close(); val_file.close()
    env.close(); evaluation_env.close()
    if not args.smoke:
        notify(f"Residual-control training complete: {args.method}, seed {args.seed}, {config['training_steps']} interactions. Best validation checkpoint: {best_step}. Threshold reached: {threshold_step}.")


if __name__ == '__main__':
    main()
