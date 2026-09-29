"""Offline release helper: inventory local Paddle model files and runtime versions.

Run with the *OCR venv* Python before release; verify the output against trusted
weight sources and license records, then copy it read-only to the API host.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from .warm_ocr import verify_manifest


def build(det, rec):
    try:
        paddle = 'paddlepaddle'
        paddle_version = importlib.metadata.version(paddle)
    except importlib.metadata.PackageNotFoundError:
        paddle = 'paddlepaddle-gpu'
        paddle_version = importlib.metadata.version(paddle)
    result = {'format': 'wineid-paddle-v1', 'ocr_version': 'PP-OCRv5', 'lang': 'ru',
              'packages': {'paddleocr': importlib.metadata.version('paddleocr'),
                           paddle: paddle_version}}
    for name, root in [('det', det), ('rec', rec)]:
        directory = Path(root)
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError('missing model directory')
        files = {}
        for path in sorted(directory.rglob('*')):
            if path.is_symlink():
                raise ValueError('model symlink not allowed')
            if path.is_file():
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        digest.update(block)
                files[path.relative_to(directory).as_posix()] = digest.hexdigest()
        if not files:
            raise ValueError('empty model directory')
        result[name] = files
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--det', required=True)
    parser.add_argument('--rec', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    path = Path(args.output)
    with path.open('x', encoding='utf-8') as out:
        json.dump(build(args.det, args.rec), out, ensure_ascii=False, indent=2)
        out.write('\n')
    verify_manifest(path, args.det, args.rec)
    print('Created OCR manifest (review provenance and licenses before release):', path)


if __name__ == '__main__':
    main()
