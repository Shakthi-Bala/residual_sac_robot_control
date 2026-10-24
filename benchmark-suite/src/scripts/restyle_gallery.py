import json
from pathlib import Path

from PIL import Image, ImageSequence

from scripts.gif_style import policy_frame
from scripts.render_gallery import DEMOS


STYLE = 'light comparison cards, navy header, full rollout return'


def main():
    folder = Path(__file__).resolve().parents[2] / 'results/media'
    for _, name, label, _ in DEMOS:
        path = folder / f'{name}.gif'
        metadata = path.with_suffix('.json')
        details = json.loads(metadata.read_text())
        if details.get('presentation') == STYLE:
            continue
        with Image.open(path) as source:
            durations, frames = [], []
            for frame in ImageSequence.Iterator(source):
                durations.append(frame.info.get('duration', 100))
                frames.append(policy_frame(frame.convert('RGB'), label, details))
        temporary = path.with_suffix('.tmp.gif')
        frames[0].save(temporary, save_all=True, append_images=frames[1:], duration=durations,
                       loop=0, optimize=False)
        temporary.replace(path)
        details['presentation'] = STYLE
        metadata.write_text(json.dumps(details, indent=2) + '\n')
        print(path, flush=True)


if __name__ == '__main__':
    main()
