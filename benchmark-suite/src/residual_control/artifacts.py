import hashlib
import json
import os
from pathlib import Path
import subprocess

import requests
import torch

from residual_control.env import ROOT


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def save_weights(path, state):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def revision():
    if os.environ.get('PROJECT_REVISION'):
        return os.environ['PROJECT_REVISION']
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 'unavailable'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def notify(message, attachment=None):
    webhook = os.environ.get('DISCORD_WEBHOOK')
    local = ROOT.parent / '.env'
    if not webhook and local.exists():
        for line in local.read_text().splitlines():
            if line.startswith('DISCORD_WEBHOOK='):
                webhook = line.split('=', 1)[1].strip()
    if not webhook:
        return
    try:
        if attachment:
            with open(attachment, 'rb') as stream:
                response = requests.post(webhook, data={'content': message}, files={'file': (Path(attachment).name, stream)}, timeout=25)
        else:
            response = requests.post(webhook, json={'content': message}, timeout=25)
        print(f'Discord HTTP {response.status_code}', flush=True)
    except requests.RequestException as error:
        print('Discord unavailable:', type(error).__name__, flush=True)
