from pathlib import Path
import zipfile

from scripts.analyze_results import load_runs


def main():
    runs = load_runs()
    output = Path('submit.zip')
    required = ('agent.pt', 'flags.json', 'log.csv', 'log.pkl', 'experiment.json')
    files = [Path('README.md'), Path('pyproject.toml'), Path('uv.lock')]
    for directory in ('src', 'experiments', 'jobs', 'tests'):
        files.extend(p for p in Path(directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'logs' not in p.parts)
    files.extend([Path('results/report.pdf'), Path('results/report.md'), Path('results/summary.json')])
    files.extend(Path('results').glob('*.png'))
    files.extend(Path('results/curves').glob('*.csv'))
    for run in runs.values():
        for name in required:
            path = run['folder'] / name
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append(path)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.as_posix())
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Archive integrity check failed')
        roots = {name.split('/')[0] for name in archive.namelist()}
        if not {'exp', 'src', 'README.md', 'pyproject.toml', 'uv.lock'} <= roots:
            raise RuntimeError(f'Invalid archive layout: {roots}')
    size = output.stat().st_size
    if size >= 100_000_000:
        raise RuntimeError(f'Archive exceeds 100 MB: {size} bytes')
    print(f'{output}: {size / 1_000_000:.2f} MB; {len(runs)} completed runs; integrity and root layout passed')


if __name__ == '__main__':
    main()
