"""Build wines.csv / images.csv per tz-wine-label-retrieval.md (ТЗ §2, §4).

- wines.csv: one row per SKU (wine_id=slug, name_ru, producer, color, vintage,
  ean, producer_split). Split is by producer (80/10/10, deterministic hash).
- images.csv: one row per usable local file with sha256, view, source, ocr_hit,
  usable_train; only catalog_media ``candidate`` originals that pass the
  implementable filters (size >=15 KB, short side >=256 px, no duplicate
  sha256 across wine_ids). pHash near-duplicate pairs across different wine_ids
  are dropped too (ТЗ §4.1); two-bottle and watermark checks are NOT implemented
  here and must be verified separately.
- ``path`` in images.csv is a bare filename relative to the uploads root passed
  as ``--uploads``. Training resolves it with ``--images-root <that root>``
  (see wineid.train_clip); never assume the images.csv directory itself.

Candidate links are NOT verified media relations (catalog_media contract).
"""
import argparse
import csv
import hashlib
import time
from collections import defaultdict
from pathlib import Path

from .catalog import load_catalog, structured_wine
from .catalog_media import catalog_media_report
from .phash import near_duplicate_indices, phash
from .roi import decode, MAX_BYTES

MIN_FILE_BYTES = 15 * 1024
MIN_SHORT_SIDE = 256

SPLIT_BUCKETS = ('train', 'train', 'train', 'train',
                 'train', 'train', 'train', 'train',
                 'validation', 'test')


def producer_split(producer: str) -> str:
    """Deterministic producer-level split bucket (ТЗ §4.3: 80/10/10)."""
    digest = hashlib.sha256(producer.encode('utf-8')).digest()
    index = int.from_bytes(digest[:2], 'big') % 100
    bucket = min(index // 10, 9)
    return SPLIT_BUCKETS[bucket]


def image_filters(path: Path, digest: str, *, with_phash: bool = False):
    """Implementable subset of ТЗ §4.1 filters; empty reason when usable.

    Returns ``(ok, reason)`` or, with ``with_phash=True``, ``(ok, reason, phash)``
    so the expensive decode is shared with the near-duplicate pass.
    """
    size = path.stat().st_size
    if size < MIN_FILE_BYTES:
        return (False, 'too_small', None) if with_phash else (False, 'too_small')
    if size > MAX_BYTES:
        # Check the byte limit before reading the file into memory (P2-15).
        return (False, 'oversized', None) if with_phash else (False, 'oversized')
    try:
        image, _ = decode(path.read_bytes(), time.monotonic() + 30)
    except Exception as exc:
        reason = f'undecodable:{exc}'
        return (False, reason, None) if with_phash else (False, reason)
    try:
        if min(image.size) < MIN_SHORT_SIDE:
            return (False, 'short_side_too_small', None) if with_phash else (False, 'short_side_too_small')
        value = phash(image) if with_phash else None
    finally:
        image.close()
    return (True, '', value) if with_phash else (True, '')


def build_tz_datasets(csv_path, uploads):
    """Return (wines_rows, images_rows, summary) without writing files."""
    wines, stats = load_catalog(csv_path)
    media_rows, media_summary = catalog_media_report(csv_path, uploads)
    by_slug = {row['slug']: row for row in media_rows}

    # sha256 across wine_ids: remove both links (ТЗ §4.1).
    digest_owners: dict[str, list[str]] = defaultdict(list)
    for slug, row in by_slug.items():
        if row.get('image_sha256'):
            digest_owners[row['image_sha256']].append(slug)
    duplicated = {d for d, owners in digest_owners.items() if len(owners) > 1}

    wines_rows = []
    for w in wines:
        fields = structured_wine(w)
        wines_rows.append({
            'wine_id': w.slug,
            'name_ru': fields['name_ru'],
            'producer': fields['producer'],
            'color': fields['color'],
            'vintage': fields['vintage'],
            'ean': '',
            'producer_split': producer_split(fields['producer'] or w.producer),
        })

    images_rows = []
    phash_values = []
    phash_owners = []
    rejected = defaultdict(int)
    for slug, row in by_slug.items():
        if row['mapping_status'] != 'candidate' or not row.get('image_path'):
            rejected['unresolved_link'] += 1
            continue
        digest = row['image_sha256']
        if digest in duplicated:
            rejected['sha256_shared_between_slugs'] += 1
            continue
        root = Path(uploads).resolve()
        path = root / row['image_path']
        if not path.is_file():
            rejected['missing_file'] += 1
            continue
        ok, reason, value = image_filters(path, digest, with_phash=True)
        if not ok:
            rejected[reason] += 1
            continue
        # One studio original per wine: usable_train requires >=2 different
        # usable frames incl. a non-studio one (ТЗ §4.2) => always 0 here.
        images_rows.append({
            'image_id': f'{slug}-0001',
            'wine_id': slug,
            'path': row['image_path'],
            'view': 'studio_front',
            'sha256': digest,
            'source': 'strapi',
            'source_url': '',
            'ocr_hit': 'none',
            'usable_train': 0,
        })
        phash_values.append(value)
        phash_owners.append(slug)

    # pHash near-duplicates across different wine_ids: drop both links (ТЗ §4.1).
    dropped = near_duplicate_indices(phash_values, phash_owners)
    if dropped:
        rejected['phash_near_duplicate'] += len(dropped)
        images_rows = [row for i, row in enumerate(images_rows) if i not in dropped]

    summary = {
        'csv_sha256': stats['csv_sha256'],
        'catalog_slugs': len(wines),
        'wines_rows': len(wines_rows),
        'images_rows': len(images_rows),
        'candidate_links': media_summary['candidate_links'],
        'unresolved': media_summary['unresolved'],
        'rejected': dict(sorted(rejected.items())),
        'note': 'images.csv rows are candidate studio links; pHash near-duplicate '
                'pairs across wine_ids are dropped (ТЗ §4.1). multi-bottle and '
                'watermark filters are not implemented here.',
    }
    return wines_rows, images_rows, summary


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError('no rows to write')
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default='strapi_output0709.csv')
    parser.add_argument('--uploads', required=True,
                        help='flat Strapi uploads directory')
    parser.add_argument('--out', required=True,
                        help='output directory for wines.csv and images.csv')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wines_path, images_path = out / 'wines.csv', out / 'images.csv'
    if wines_path.exists() or images_path.exists():
        parser.error('output files must not exist')
    wines_rows, images_rows, summary = build_tz_datasets(args.csv, args.uploads)
    write_csv(wines_path, wines_rows)
    write_csv(images_path, images_rows)
    import json
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()