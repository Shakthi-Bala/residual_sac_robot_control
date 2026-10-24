import itertools
import json
from dataclasses import asdict

import numpy as np

from residual_control.env import ROOT, UR16eTrackingEnv
from residual_control.protocol import calibration_cases, rollout


def main():
    cases = calibration_cases()
    rows = []
    for kp, ki, assumed_payload in itertools.product((2., 4., 6., 8.), (0., .5, 1.), (0., 1.5)):
        env = UR16eTrackingEnv('controller', kp=kp, ki=ki, assumed_payload=assumed_payload)
        metrics = [rollout(env, case) for case in cases]
        env.close()
        mean = {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}
        score = mean['tracking_rmse_m'] + .03 * mean['failure'] + .01 * (1 - mean['success']) + .0005 * mean['command_acceleration_rms']
        row = {'kp': kp, 'ki': ki, 'assumed_payload': assumed_payload, 'score': score, **mean}
        rows.append(row)
        print(json.dumps(row), flush=True)
    best = min(rows, key=lambda row: row['score'])
    output = {
        'selected': {k: best[k] for k in ('kp', 'ki', 'assumed_payload')},
        'objective': 'mean RMSE + 0.03 failure + 0.01 (1-success) + 0.0005 command acceleration RMS',
        'calibration_cases': [asdict(c) for c in cases], 'grid': rows,
        'calibration_interactions': sum(int(round(r['episode_steps'] * len(cases))) for r in rows),
        'selection_uses_test_set': False,
    }
    (ROOT / 'results/controller_tuning.json').write_text(json.dumps(output, indent=2) + '\n')
    print('Selected:', output['selected'], flush=True)


if __name__ == '__main__':
    main()
