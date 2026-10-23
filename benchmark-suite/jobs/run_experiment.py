import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import requests
import yaml


def notify(message):
    webhook = os.environ.get('DISCORD_WEBHOOK')
    if webhook:
        try:
            response = requests.post(webhook, json={'content': message}, timeout=20)
            response.raise_for_status()
        except requests.RequestException as error:
            print(f'Discord notification failed: {type(error).__name__}', flush=True)


def main():
    task = json.loads(Path(sys.argv[1]).read_text())[int(sys.argv[2])]
    config = yaml.safe_load(Path(task['config']).read_text())
    before = set(Path('exp').glob('*')) if Path('exp').exists() else set()
    command = [sys.executable, f"src/scripts/run_{task['algorithm']}.py", '-cfg', task['config'], '--seed', str(task.get('seed', 1))]
    command.extend(task.get('args', []))
    print('Command:', ' '.join(command), flush=True)
    result = subprocess.run(command)
    if result.returncode:
        notify(f"HW3 run failed: {task['config']} (Slurm {os.environ.get('SLURM_JOB_ID', 'local')}). Check the job log.")
        raise SystemExit(result.returncode)
    prefix = config['env_name'].replace('NoFrameskip-v4', '').replace('NoFrameskip-v0', '') + '_' + config['exp_name'] + '_sd' + str(task.get('seed', 1)) + '_'
    runs = [p for p in set(Path('exp').glob(prefix + '*')) - before if (p / 'log.csv').exists()]
    if len(runs) != 1:
        raise RuntimeError(f'Expected one output directory for {prefix}, found {runs}')
    run = runs[0]
    rows = [r for r in csv.DictReader((run / 'log.csv').open()) if r.get('Eval_AverageReturn')]
    best = max(float(r['Eval_AverageReturn']) for r in rows)
    final = float(rows[-1]['Eval_AverageReturn'])
    notify(f"HW3 finished: {config['env_name']} / {config['exp_name']}, seed {task.get('seed', 1)}. Best evaluation mean: {best:.1f}; final: {final:.1f}. Logs: {run.name}.")


if __name__ == '__main__':
    main()
