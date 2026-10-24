import argparse
import csv
import json

import numpy as np
import torch

from infrastructure import pytorch_util as ptu
from residual_control.agent import make_agent
from residual_control.artifacts import digest, save_json
from residual_control.env import ROOT, UR16eTrackingEnv
from residual_control.protocol import rollout, test_cases


def load_policy(method, seed, checkpoint='best'):
    folder = ROOT / 'runs/main' / f'{method}_seed{seed}'
    metadata = json.loads((folder / 'metadata.json').read_text())
    if not metadata['complete']:
        raise RuntimeError(f'Incomplete training: {folder}')
    agent = make_agent((87,), metadata['config'])
    agent.load_state_dict(torch.load(folder / f'{checkpoint}.pt', map_location='cpu', weights_only=True))
    agent.eval()
    return agent, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['controller', 'standalone', 'residual'], required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    ptu.init_gpu(use_gpu=False)
    controller = json.loads((ROOT / 'results/controller_tuning.json').read_text())['selected']
    env = UR16eTrackingEnv(args.method, **controller)
    agent, metadata = load_policy(args.method, args.seed) if args.method != 'controller' else (None, {})
    rows = []
    for condition, scenario in test_cases():
        row = {'method': args.method, 'training_seed': args.seed, 'condition': condition,
               'task': scenario.task, 'target_seed': scenario.seed, 'payload_kg': scenario.payload,
               'delay_ms': scenario.delay * 20,
               **rollout(env, scenario, agent.deterministic_action if agent else None)}
        rows.append(row)
    folder = ROOT / 'results/evaluation'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{args.method}_seed{args.seed}.csv'
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    save_json(path.with_suffix('.json'), {'method': args.method, 'training_seed': args.seed,
              'checkpoint_step': metadata.get('best_validation_step'), 'training_revision': metadata.get('source_revision'),
              'model_sha256': digest(ROOT / 'models/ur16e.xml'), 'test_interactions': sum(r['episode_steps'] for r in rows),
              'episodes': len(rows), 'selection': 'best validation checkpoint; test targets never select weights',
              'mean_metrics': {k: float(np.mean([r[k] for r in rows])) for k in
                               ('success', 'tracking_rmse_m', 'final_error_m', 'failure')}})
    env.close()
    print(path, flush=True)


if __name__ == '__main__':
    main()
