import argparse
import os
from pathlib import Path

import requests


def post_update(message, attachment=None):
    webhook = os.environ.get('DISCORD_WEBHOOK')
    if not webhook:
        environment_file = Path('../.env')
        if environment_file.exists():
            for line in environment_file.read_text().splitlines():
                if line.startswith('DISCORD_WEBHOOK='):
                    webhook = line.split('=', 1)[1].strip()
    if not webhook:
        raise RuntimeError('Set DISCORD_WEBHOOK or add it to the ignored root .env')
    if attachment:
        with open(attachment, 'rb') as stream:
            response = requests.post(webhook, data={'content': message}, files={'file': (Path(attachment).name, stream)}, timeout=30)
    else:
        response = requests.post(webhook, json={'content': message}, timeout=30)
    if not response.ok:
        raise RuntimeError(f'Discord returned HTTP {response.status_code}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('message')
    parser.add_argument('--attachment', type=Path)
    args = parser.parse_args()
    post_update(args.message, args.attachment)
    print('Discord update sent')


if __name__ == '__main__':
    main()
