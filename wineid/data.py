"""Verified gallery and independent-query manifest audits; never infer truth from names."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from .roi import decode, InvalidImage, MAX_BYTES
import time


def jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def query_manifest_errors(rows, *, gallery_hashes=(), gallery_groups=(), catalog_ids=None):
    """Shared whole-manifest identity, GT and split checks (before split selection).

    Callers must verify that declared image hashes match the bytes on disk.
    """
    errors, ids, hashes, groups = [], set(), {}, {}
    gallery_hashes, gallery_groups = set(gallery_hashes), set(gallery_groups)
    for row in rows:
        qid, group, split = row.get('query_id'), row.get('source_group'), row.get('split')
        digest = row.get('image_sha256')
        if not isinstance(qid, str) or not qid or qid in ids:
            errors.append(f'query: missing/duplicate id {qid}')
        else:
            ids.add(qid)
        if not isinstance(group, str) or not group:
            errors.append(f'query: missing source_group {qid}')
        else:
            if group in groups and groups[group] != split:
                errors.append(f'query: source_group across splits {group}')
            groups[group] = split
        if split not in ('train', 'validation', 'test'):
            errors.append(f'query: missing split {qid}')
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            errors.append(f'query: invalid image hash {qid}')
        else:
            if digest in hashes and hashes[digest] != split:
                errors.append(f'query: image hash across splits {qid}')
            hashes[digest] = split
            if digest in gallery_hashes:
                errors.append(f'query/gallery leakage {qid}')
        if isinstance(group, str) and group in gallery_groups:
            errors.append(f'query/gallery leakage {qid}')
        status, slug = row.get('gt_status'), row.get('gt_slug')
        if status not in ('known', 'unknown', 'ambiguous'):
            errors.append(f'query: invalid gt_status {qid}')
        elif status == 'known' and (not isinstance(slug, str) or not slug or
                                    not row.get('gt_source') or
                                    (catalog_ids is not None and slug not in catalog_ids)):
            errors.append(f'query: independent GT missing {qid}')
        elif status != 'known' and slug is not None:
            errors.append(f'query: GT must be null {qid}')
    return errors, groups


def calibration_test_leakage_errors(calibration_rows, test_rows, *, calibration_path=None,
                                   test_path=None):
    """Shared calibration/test isolation check, reusable outside the visual evaluator.

    Returns human-readable errors for a shared source_group, image hash, query_id
    or the same manifest file. Callers decide whether to raise or report.
    """
    errors = []
    if {r.get('source_group') for r in calibration_rows} & {r.get('source_group') for r in test_rows}:
        errors.append('calibration/test shared source_group')
    if {r.get('image_sha256') for r in calibration_rows} & {r.get('image_sha256') for r in test_rows}:
        errors.append('calibration/test shared image hash')
    if {r.get('query_id') for r in calibration_rows} & {r.get('query_id') for r in test_rows}:
        errors.append('calibration/test shared query_id')
    if calibration_path is not None and test_path is not None and \
            Path(calibration_path).resolve() == Path(test_path).resolve():
        errors.append('calibration and test are the same manifest')
    return errors


def retrieval_manifest_errors(rows, available_ids):
    """Shared text-to-image retrieval manifest audit (JINA_CLIP.md §3).

    The documented schema is query_id/text/relevant_image_ids/hard_negative; it
    carries no split/source_group, so independence stays the author's duty. When
    optional ``source_group``/``image_sha256`` fields are present they are checked
    for cross-split leakage too.
    """
    errors, ids, groups, hashes = [], set(), {}, {}
    available = set(available_ids)
    for row in rows:
        qid, relevant = row.get('query_id'), row.get('relevant_image_ids')
        if not isinstance(qid, str) or not qid or qid in ids:
            errors.append(f'retrieval: missing/duplicate query_id {qid}')
        else:
            ids.add(qid)
        if not isinstance(row.get('text'), str) or not row['text'].strip():
            errors.append(f'retrieval: empty text {qid}')
        if (not isinstance(relevant, list) or not relevant
                or any(not isinstance(i, str) or not i for i in relevant)
                or len(set(relevant)) != len(relevant)
                or not set(relevant) <= available):
            errors.append(f'retrieval: invalid relevant_image_ids {qid}')
        if type(row.get('hard_negative', False)) is not bool:
            errors.append(f'retrieval: invalid hard_negative {qid}')
        split = row.get('split')
        if split is not None:
            if split not in ('train', 'validation', 'test'):
                errors.append(f'retrieval: invalid split {qid}')
            group = row.get('source_group')
            if not isinstance(group, str) or not group:
                errors.append(f'retrieval: missing source_group {qid}')
            elif group in groups and groups[group] != split:
                errors.append(f'retrieval: source_group across splits {group}')
            else:
                groups[group] = split
            digest = row.get('image_sha256')
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in '0123456789abcdef' for c in digest)):
                errors.append(f'retrieval: invalid image hash {qid}')
            elif digest in hashes and hashes[digest] != split:
                errors.append(f'retrieval: image hash across splits {qid}')
            else:
                hashes[digest] = split
    return errors


def audit(catalog, gallery_path, queries_path):
    """Validate provenance and split leakage; returns counts and errors, not quality claims.

    Gallery rows require manual evidence, asset_id, wine_id, role, bbox, SHA256,
    session/source_group; queries require independent GT and source_group.
    Hash checks prevent accidental derivative/image substitution.
    """
    gallery = jsonl(gallery_path)
    queries = jsonl(queries_path)
    ids = {w.wine_id for w in catalog}
    errors, seen = [], set()
    asset_groups = defaultdict(set)
    base = Path(gallery_path).resolve().parent
    qbase = Path(queries_path).resolve().parent
    def inspect(row, origin, root):
        key = (origin, row.get('asset_id', row.get('query_id')))
        if not key[1] or key in seen:
            errors.append(f'{origin}: missing/duplicate id {key[1]}')
        seen.add(key)
        path = (root / str(row.get('image_path', ''))).resolve()
        if not path.is_relative_to(root):
            errors.append(f'{origin}: unsafe path {key[1]}')
            return
        if not path.is_file():
            errors.append(f'{origin}: missing image {key[1]}')
            return
        if path.stat().st_size > MAX_BYTES:
            # Byte limit before reading the file into memory (P2-15).
            errors.append(f'{origin}: oversized image {key[1]}')
            return
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if row.get('image_sha256') != digest:
            errors.append(f'{origin}: image hash mismatch {key[1]}')
        asset_groups[digest].add((origin, row.get('source_group')))
        try:
            image, mime = decode(data, time.monotonic()+8)
            bbox = row.get('bbox')
            if not (isinstance(bbox, list) and len(bbox) == 4 and
                    all(type(x) is int for x in bbox) and
                    0 <= bbox[0] < bbox[2] <= image.width and
                    0 <= bbox[1] < bbox[3] <= image.height):
                errors.append(f'{origin}: invalid oriented bbox {key[1]}')
        except (InvalidImage, TimeoutError) as exc:
            errors.append(f'{origin}: invalid image {key[1]}: {exc}')
        if not row.get('source_group'):
            errors.append(f'{origin}: missing source_group {key[1]}')
    for row in gallery:
        inspect(row, 'gallery', base)
        if row.get('wine_id') not in ids:
            errors.append(f'gallery: unknown wine_id {row.get("wine_id")}')
        if row.get('mapping_status') != 'verified' or not row.get('verification_method') or not row.get('verified_by'):
            errors.append(f'gallery: manual verification required {row.get("asset_id")}')
        if row.get('role') not in ('front', 'back', 'neck'):
            errors.append(f'gallery: missing role {row.get("asset_id")}')
    for row in queries:
        inspect(row, 'query', qbase)
    query_errors, groups = query_manifest_errors(
        queries, gallery_hashes={r.get('image_sha256') for r in gallery},
        gallery_groups={r.get('source_group') for r in gallery}, catalog_ids=ids)
    errors.extend(query_errors)
    for digest, origins in asset_groups.items():
        if len({o for o, _ in origins}) > 1:
            errors.append(f'query/gallery identical image {digest}')
    gallery_groups = {r.get('source_group') for r in gallery}
    for r in queries:
        if r.get('source_group') in gallery_groups:
            errors.append(f'query/gallery shared source_group {r.get("source_group")}')
    return {'gallery_assets': len(gallery), 'gallery_wines': len({r.get('wine_id') for r in gallery}),
            'catalog_wines': len(ids), 'gallery_coverage': len({r.get('wine_id') for r in gallery})/len(ids) if ids else None,
            'queries': len(queries), 'query_source_groups': len(groups),
            'errors': sorted(set(errors))}
