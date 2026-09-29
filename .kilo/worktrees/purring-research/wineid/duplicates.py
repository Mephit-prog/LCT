"""Near-duplicate review report for query/gallery manifests.

Implements the clip_review_improvement.md recommendation: pHash near-duplicates
are surfaced as a **manual-review list** (and the ТЗ §8 Hamming 8-16 band is
listed separately); the tool never auto-confirms independence, never deletes or
merges rows and never promotes a candidate media link to verified.

Manifest rows use ``image_path`` relative to the manifest directory and may carry
``query_id``/``asset_id``, ``source_group`` and ``split``.
"""
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from .phash import NEAR_DUPLICATE_DISTANCE, distance_band_pairs, hamming, phash
from .roi import MAX_BYTES, decode


@dataclass(frozen=True)
class Entry:
    row_id: str
    phash: int | None
    source_group: str | None
    split: str | None
    reason: str | None = None


def _hash_row(row, root: Path) -> Entry:
    row_id = row.get('query_id') or row.get('asset_id') or '?'
    path = (root / str(row.get('image_path', ''))).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return Entry(row_id, None, row.get('source_group'), row.get('split'), 'missing_or_unsafe')
    if path.stat().st_size > MAX_BYTES:
        return Entry(row_id, None, row.get('source_group'), row.get('split'), 'oversized')
    try:
        image, _ = decode(path.read_bytes(), time.monotonic() + 30)
    except Exception as exc:  # noqa: BLE001 - report, never crash the audit
        return Entry(row_id, None, row.get('source_group'), row.get('split'), f'undecodable:{exc}')
    try:
        value = phash(image)
    finally:
        image.close()
    return Entry(row_id, value, row.get('source_group'), row.get('split'))


def near_duplicate_report(rows, root, *, max_distance: int = NEAR_DUPLICATE_DISTANCE,
                          band=(8, 16)) -> dict:
    """Return a manual-review report; it is not an independence decision."""
    root = Path(root).resolve()
    entries = [_hash_row(row, root) for row in rows]
    hashes = [e.phash for e in entries]
    owners = [e.row_id for e in entries]

    near = []
    for i in range(len(hashes)):
        if hashes[i] is None:
            continue
        for j in range(i + 1, len(hashes)):
            if hashes[j] is None or owners[i] == owners[j]:
                continue
            distance = hamming(hashes[i], hashes[j])
            if distance < max_distance:
                near.append(_pair(entries[i], entries[j], distance))
    band_pairs = [
        _pair(entries[i], entries[j], distance)
        for i, j, distance in distance_band_pairs(hashes, owners, low=band[0], high=band[1])
    ]
    return {
        'rows': len(rows),
        'hashed': sum(e.phash is not None for e in entries),
        'unreadable': [{'id': e.row_id, 'reason': e.reason} for e in entries if e.phash is None],
        'near_duplicates': near,
        'band_pairs': band_pairs,
        'note': 'Manual review only: a pair is not proof of derivative/shoot leakage, '
                'and rows are never removed automatically.',
    }


def _pair(left: Entry, right: Entry, distance: int) -> dict:
    return {'left': left.row_id, 'right': right.row_id, 'distance': distance,
            'left_split': left.split, 'right_split': right.split,
            'left_group': left.source_group, 'right_group': right.source_group,
            'same_group': left.source_group == right.source_group,
            'cross_split': left.split != right.split}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', help='JSONL with image_path (relative to the manifest)')
    parser.add_argument('--max-distance', type=int, default=NEAR_DUPLICATE_DISTANCE)
    parser.add_argument('--band-low', type=int, default=8)
    parser.add_argument('--band-high', type=int, default=16)
    args = parser.parse_args()
    path = Path(args.manifest).resolve()
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    report = near_duplicate_report(rows, path.parent, max_distance=args.max_distance,
                                   band=(args.band_low, args.band_high))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
