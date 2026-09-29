"""Assemble a Windows delivery from the single authoritative source tree.

Run on the Windows build machine with --provision before giving the bundle to
operators. Do not include a developer venv, node_modules or local .env files.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def assemble(root: Path, out: Path) -> None:
    root, out = root.resolve(), out.resolve()
    if out == root / 'backend':
        raise ValueError('delivery directory cannot be the backend subdirectory')
    backend = out / 'backend'
    backend.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root / 'wineid', backend / 'wineid', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(root / 'artifacts', backend / 'artifacts', dirs_exist_ok=True)
    for name in ('strapi_output0709.csv', 'producer_aliases.txt', 'constraints.txt',
                 'requirements-api.txt', 'requirements-clip.txt', 'requirements-vision.txt'):
        shutil.copy2(root / name, backend / name)
    if out != root:  # In-place development bundle: web/local/guide already present.
        shutil.copytree(root / 'web', out / 'web', dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('node_modules', 'dist', '.env', '.env.*'))
        shutil.copytree(root / 'local', out / 'local', dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('logs', 'run'))
        shutil.copy2(root / 'RUN_GUIDE.md', out / 'RUN_GUIDE.md')


def provision(out: Path) -> None:
    if sys.platform != 'win32':
        raise RuntimeError('provision on Windows: virtual environments are not portable')
    backend = out / 'backend'
    subprocess.run([sys.executable, '-m', 'venv', str(backend / '.venv')], check=True)
    python = backend / '.venv' / 'Scripts' / 'python.exe'
    subprocess.run([str(python), '-m', 'pip', 'install', '-c', 'constraints.txt',
                    '-r', 'requirements-api.txt', '-r', 'requirements-clip.txt'], cwd=backend, check=True)
    subprocess.run(['npm.cmd', 'ci'], cwd=out / 'web', check=True)
    # The CLIP worker is offline at runtime: preload pinned remote code/weights
    # now and fail this build rather than delivering an unbootable bundle.
    cache = backend / 'artifacts' / 'hf-cache'
    cache.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'HF_HOME': str(cache), 'HF_HUB_OFFLINE': '0', 'TRANSFORMERS_OFFLINE': '0'}
    subprocess.run([str(python), '-c',
                    'from wineid.jina_clip import JinaCLIPEmbedder; '
                    'JinaCLIPEmbedder(device="cpu", allow_remote_code=True)'],
                   cwd=backend, env=env, check=True)
    # Ensure it can reload without network using the same offline subprocess
    # environment as the API worker (but do not start the API/keyed MinerU).
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    subprocess.run([str(python), '-c',
                    'from wineid.jina_clip import JinaCLIPEmbedder; '
                    'JinaCLIPEmbedder(device="cpu", allow_remote_code=True)'],
                   cwd=backend, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, nargs='?', default=Path(__file__).resolve().parent / 'delivery')
    parser.add_argument('--provision', action='store_true', help='Windows venv, npm ci, pinned HF cache')
    args = parser.parse_args()
    assemble(Path(__file__).resolve().parent, args.destination)
    if args.provision:
        provision(args.destination.resolve())
    print(f'Bundle: {args.destination.resolve()}')


if __name__ == '__main__':
    main()
