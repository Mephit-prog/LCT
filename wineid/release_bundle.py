"""Offline release-file inventory, not a signature or a deployment certification.

Usage: python -m wineid.release_bundle build ROOT manifest.json --revision GIT_SHA
         --file csv=strapi_output0709.csv --file aliases=producer_aliases.txt ...
       python -m wineid.release_bundle verify ROOT manifest.json

A bundle must contain local copies of all listed files. No network, models or
secrets are accessed. Verification rejects symlinks and path escapes. The file
inventory does not replace environment locks, license review, policy validation,
real smoke/load tests, or a signed distribution channel.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

FORMAT = 'wineid-release-files-v1'
ROLES = frozenset({'csv', 'aliases', 'policy', 'classes', 'gallery', 'adapter',
                   'ocr_manifest', 'tesseract', 'rus-traineddata', 'eng-traineddata'})
SHA = re.compile(r'[0-9a-f]{64}\Z')
REVISION = re.compile(r'(?:[0-9a-f]{40}|[0-9a-f]{64})\Z')


def _location(root, name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('invalid bundle path')
    relative = Path(name)
    if (relative.is_absolute() or relative.as_posix() != name
            or any(part in ('.', '..') for part in relative.parts)):
        raise ValueError('unsafe bundle path')
    path = root / relative
    if not path.resolve().is_relative_to(root) or any(part.is_symlink() for part in
            (path, *path.parents) if part != root and part.is_relative_to(root)):
        raise ValueError('symlink or escaped bundle path')
    if not path.is_file():
        raise ValueError('missing bundle file')
    return path


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _roles(files):
    if (not isinstance(files, dict) or not {'csv', 'aliases'} <= set(files)
            or any(not isinstance(role, str) or not re.fullmatch(r'[a-z][a-z0-9-]*', role)
                   or (role not in ROLES and not role.startswith('lock-'))
                   for role in files)):
        raise ValueError('bundle needs CSV/aliases and known roles')
    if any(not isinstance(name, str) for name in files.values()):
        raise ValueError('invalid bundle path')
    if len(set(files.values())) != len(files):
        raise ValueError('duplicate bundle paths')


def build(root, files, revision):
    root = Path(root).resolve(strict=True)
    if not root.is_dir() or not isinstance(revision, str) or not REVISION.fullmatch(revision):
        raise ValueError('directory and full git revision required')
    _roles(files)
    return {'format': FORMAT, 'revision': revision,
            'files': {role: {'path': name, 'sha256': _hash(_location(root, name))}
                      for role, name in sorted(files.items())}}


def verify(root, manifest):
    root = Path(root).resolve(strict=True)
    if (not root.is_dir() or not isinstance(manifest, dict)
            or set(manifest) != {'format', 'revision', 'files'}
            or manifest['format'] != FORMAT or not isinstance(manifest['revision'], str)
            or not REVISION.fullmatch(manifest['revision'])):
        raise ValueError('invalid release file inventory')
    entries = manifest['files']
    if not isinstance(entries, dict):
        raise ValueError('invalid release files')
    _roles({role: item.get('path') if isinstance(item, dict) else None
            for role, item in entries.items()})
    for entry in entries.values():
        if (not isinstance(entry, dict) or set(entry) != {'path', 'sha256'}
                or not isinstance(entry['sha256'], str)
                or not SHA.fullmatch(entry['sha256'])):
            raise ValueError('invalid release file entry')
        if _hash(_location(root, entry['path'])) != entry['sha256']:
            raise ValueError('release file checksum mismatch')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    build_cmd = sub.add_parser('build')
    build_cmd.add_argument('root', type=Path)
    build_cmd.add_argument('manifest', type=Path)
    build_cmd.add_argument('--revision', required=True)
    build_cmd.add_argument('--file', action='append', required=True, metavar='ROLE=PATH')
    verify_cmd = sub.add_parser('verify')
    verify_cmd.add_argument('root', type=Path)
    verify_cmd.add_argument('manifest', type=Path)
    args = parser.parse_args()
    if args.action == 'build':
        files = {}
        for item in args.file:
            role, sep, path = item.partition('=')
            if not sep or role in files:
                parser.error('use distinct --file ROLE=PATH entries')
            files[role] = path
        manifest = build(args.root, files, args.revision)
        with args.manifest.open('x', encoding='utf-8') as out:
            json.dump(manifest, out, ensure_ascii=False, indent=2, sort_keys=True)
            out.write('\n')
    else:
        verify(args.root, json.loads(args.manifest.read_text(encoding='utf-8')))
    print(json.dumps({'status': 'ok', 'manifest': str(args.manifest)}))


if __name__ == '__main__':
    main()
