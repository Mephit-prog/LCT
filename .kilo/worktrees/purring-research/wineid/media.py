"""Inventory uploads and derivative suffix groups; these are NOT wine mappings."""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, UnidentifiedImageError

PREFIXES = ('thumbnail_', 'small_', 'medium_', 'large_')
# Strapi names convention: slug_HASH.ext; group identity is a hypothesis only.
SUFFIX = re.compile(r'^(.*)_([a-fA-F0-9]{10})$')


def inventory(root):
    root = Path(root)
    if not root.is_dir():
        raise ValueError('uploads directory not found')
    formats, mismatched, derivative_groups = Counter(), [], defaultdict(list)
    ext_to_mime = {'.jpg':'image/jpeg', '.jpeg':'image/jpeg', '.png':'image/png',
                   '.webp':'image/webp', '.gif':'image/gif'}
    count = 0
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.is_symlink():
            continue
        count += 1
        try:
            with Image.open(path) as image:
                fmt = image.format
                mime = Image.MIME.get(fmt, 'unknown')
                size = [image.width, image.height]
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            with path.open('rb') as source:
                signature = source.read(12)
            mime = ('image/jpeg' if signature.startswith(b'\xff\xd8\xff') else
                    'image/png' if signature.startswith(b'\x89PNG\r\n\x1a\n') else
                    'image/webp' if signature.startswith(b'RIFF') and signature[8:12] == b'WEBP' else
                    'application/pdf' if signature.startswith(b'%PDF-') else
                    'video/mp4' if signature[4:8] == b'ftyp' else 'unknown')
            size = None
        formats[mime] += 1
        if path.suffix.lower() in ext_to_mime and ext_to_mime[path.suffix.lower()] != mime:
            mismatched.append(path.name)
        stem = path.stem
        variant = 'original'
        for prefix in PREFIXES:
            if stem.startswith(prefix):
                stem, variant = stem[len(prefix):], prefix[:-1]
                break
        match = SUFFIX.fullmatch(stem)
        if match:
            # Include stem before suffix to avoid merging different media with
            # coincidentally identical ten-character hashes.
            # Casefolded group key: deliberate (P2-17) — suffix-stem identity is a
            # hypothesis only; case-only differences ("Bottle" vs "bottle") are
            # not treated as separate media families.
            derivative_groups[(match.group(1).casefold(), match.group(2).lower())].append(
                {'path':path.name, 'variant':variant, 'mime':mime, 'dimensions':size})
    return {'file_count':count, 'mime_counts':dict(formats),
            'extension_mismatches':mismatched,
            'suffix_groups':len(derivative_groups),
            'groups_with_multiple_files':sum(len(g)>1 for g in derivative_groups.values()),
            'groups':{'|'.join(key):value for key, value in derivative_groups.items()},
            'note':'suffix groups and filenames are unverified candidate media, not slug relations'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('uploads_dir')
    parser.add_argument('output', help='new JSON file outside eval/')
    args = parser.parse_args()
    result = inventory(args.uploads_dir)
    with open(args.output, 'x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k != 'groups'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
