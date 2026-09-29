"""Link catalog-asserted photo names to local Strapi *candidate* asset families.

The CSV is authoritative for Slug and its `Название фото` field. Upload names
are not CMS media relations; no inferred link is labelled verified, and no
validation/test split or independent recognition accuracy is created here.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from .catalog import load_catalog, normalize
from .media import PREFIXES, SUFFIX
from .roi import MAX_BYTES

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def families_by_name(root):
    """Collect potential original/derivative families from a flat uploads folder."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('uploads directory does not exist')
    families = defaultdict(list)
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.is_symlink():
            continue
        stem, role = path.stem, 'original'
        for prefix in PREFIXES:
            if stem.startswith(prefix):
                stem, role = stem[len(prefix):], prefix[:-1]
                break
        match = SUFFIX.fullmatch(stem)
        if match:
            # Keep the original stem as well as the 10-char suffix: normalization
            # can merge different assets which must never become one GT object.
            key = (match.group(1).casefold(), match.group(2).lower())
            families[key].append((path, role))
    by_name = defaultdict(set)
    for key in families:
        by_name[normalize(key[0])].add(key)
    return root, families, by_name


def _digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def catalog_media_report(csv_path, uploads):
    wines, stats = load_catalog(csv_path)
    if any('Название фото' not in wine.raw for wine in wines):
        raise ValueError('CSV missing Название фото')
    root, families, by_name = families_by_name(uploads)
    shared_names = defaultdict(set)
    for wine in wines:
        name = wine.raw['Название фото'].strip()
        if name:
            shared_names[name.casefold()].add(wine.slug)
    staged = []
    for wine in wines:
        photo_name = wine.raw['Название фото'].strip()
        photo = by_name.get(normalize(Path(photo_name).stem), set()) if photo_name else set()
        slug = by_name.get(normalize(wine.slug), set())
        if photo and slug and photo != slug:
            options = photo | slug
            issue, method = 'photo_slug_conflict', None
        else:
            options = photo or slug
            issue = 'missing_csv_photo_name' if not photo_name else (
                'shared_csv_photo_name' if len(shared_names[photo_name.casefold()]) > 1 else (
                    'unmatched' if not options else 'multiple_asset_families' if len(options) > 1 else None))
            method = ('photo_name_and_slug' if photo and slug else
                      'csv_photo_name' if photo else 'slug_filename_only' if slug else None)
        staged.append((wine, photo_name, options, issue, method))
    # If the same candidate media is reachable from multiple catalog rows,
    # neither row may silently claim it as a unique exact SKU reference.
    consumers = defaultdict(set)
    for wine, _, options, _, _ in staged:
        for family in options:
            consumers[family].add(wine.slug)
    rows = []
    for wine, photo_name, options, issue, method in staged:
        candidates = []
        for family in sorted(options):
            files = sorted(families[family], key=lambda pair: pair[0].name)
            candidates.append({'family_stem': family[0], 'strapi_suffix': family[1],
                               'originals': [path.name for path, role in files if role == 'original'],
                               'derivatives': [path.name for path, role in files if role != 'original']})
        if issue is None and any(len(consumers[family]) > 1 for family in options):
            issue = 'asset_candidate_shared_between_slugs'
        original = None
        if issue is None:
            originals = candidates[0]['originals']
            if len(originals) != 1:
                issue = 'missing_or_multiple_originals'
            else:
                original = root / originals[0]
                if original.suffix.lower() not in IMAGE_EXTENSIONS:
                    issue = 'unsupported_original'
                elif original.stat().st_size > MAX_BYTES:
                    issue = 'oversized_original'
        rows.append({'slug': wine.slug, 'csv_photo_name': photo_name,
                     'csv_sha256': stats['csv_sha256'], 'csv_row_ids': list(wine.source_row_ids),
                     'mapping_status': 'candidate' if issue is None else 'unresolved',
                     'link_method': method, 'issue': issue, 'candidate_families': candidates,
                     'image_path': original.name if issue is None else None,
                     'image_sha256': _digest(original) if issue is None else None})
    counts = Counter(row['issue'] or row['link_method'] for row in rows)
    return rows, {'csv_sha256': stats['csv_sha256'], 'catalog_slugs': len(wines),
                  'upload_root': str(root), 'upload_families': len(families),
                  'candidate_links': sum(row['mapping_status'] == 'candidate' for row in rows),
                  'unresolved': sum(row['mapping_status'] == 'unresolved' for row in rows),
                  'breakdown': dict(sorted(counts.items())),
                  'note': 'CSV supplies catalog SKU/photo-name assertions. Local filename/suffix matches '
                          'are candidates only, NOT verified media relations or independent query GT.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', help='catalog CSV with Slug and Название фото')
    parser.add_argument('uploads', help='flat Strapi uploads directory')
    parser.add_argument('output', help='new JSONL file; not independent evaluation GT')
    args = parser.parse_args()
    out = Path(args.output).resolve()
    root = Path(args.uploads).resolve()
    if out.is_relative_to(root) or out.exists() or not out.parent.is_dir():
        parser.error('output must be new, outside uploads and inside an existing directory')
    rows, summary = catalog_media_report(args.csv, root)
    with out.open('x', encoding='utf-8') as dest:
        for row in rows:
            dest.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
