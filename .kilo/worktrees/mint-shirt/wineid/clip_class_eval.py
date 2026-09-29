"""Audited offline photo-to-SKU ranking for Jina class indexes (not an accept policy).

No model is loaded by this module. Full-manifest checks precede selecting a split;
unknown and ambiguous photos are never treated as known SKUs.
"""
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

from .data import jsonl, query_manifest_errors
from .phash import distance_band_pairs, phash
from .roi import MAX_BYTES
from .clip_zero_shot import prompt_collisions, read_image

SCHEMA = 'class-ranking-eval-v1'
IMAGE_MODE = 'exif-rgb-full-frame-v1'


def fraction(numerator, denominator):
    return {'numerator': numerator, 'denominator': denominator,
            'value': numerator / denominator if denominator else None}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit_class_queries(index, manifest):
    """Check every row and on-disk hash before a split is selected or weights loaded."""
    if index.kind != 'classes':
        raise ValueError('photo-to-SKU evaluation requires a class index')
    path = Path(manifest).resolve()
    rows = jsonl(path)
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError('empty or invalid class query manifest')
    errors, _ = query_manifest_errors(rows, catalog_ids={item['id'] for item in index.items})
    if errors:
        raise ValueError(errors[0])
    root = path.parent
    seen_hashes = set()
    for row in rows:
        qid = row['query_id']
        if not isinstance(row.get('gt_source'), str) or not row['gt_source'].strip():
            raise ValueError(f'independent GT source required (including unknown/ambiguous): {qid}')
        name = row.get('image_path')
        if (not isinstance(name, str) or not name.strip() or Path(name).is_absolute()
                or '..' in Path(name).parts or '\\' in name):
            raise ValueError(f'unsafe/missing image path: {qid}')
        photo = (root / name).resolve()
        if not photo.is_relative_to(root) or not photo.is_file():
            raise ValueError(f'missing/unsafe image: {qid}')
        with photo.open('rb') as source:
            data = source.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES or hashlib.sha256(data).hexdigest() != row['image_sha256']:
            raise ValueError(f'image oversized or hash mismatch: {qid}')
        if row['image_sha256'] in seen_hashes:
            raise ValueError(f'duplicate image in query manifest: {qid}')
        seen_hashes.add(row['image_sha256'])
    return rows, sha256_file(path)


def _prompt_groups(index, normalized=False):
    groups = {}
    for item in index.items:
        prompts = item['prompts']
        if normalized:
            prompts = [' '.join(unicodedata.normalize('NFKC', text).casefold().split())
                       for text in prompts]
        groups.setdefault(tuple(prompts), []).append(item['id'])
    return {cid: group for group in groups.values() if len(group) > 1 for cid in group}


def _collision_stats(collisions, count):
    groups = {tuple(group) for group in collisions.values()}
    return {'groups': len(groups), 'classes': len(collisions),
            'class_fraction': fraction(len(collisions), count),
            'largest_group': max(map(len, groups), default=0)}


def _group_interval(known, key):
    """Descriptive percentile cluster bootstrap, resampling shoots, not individual frames."""
    groups = {}
    for row in known:
        groups.setdefault(row['source_group'], []).append(row)
    if len(groups) < 2:
        return None
    sizes = np.array([len(group) for group in groups.values()])
    successes = np.array([sum(row[key] for row in group) for group in groups.values()])
    rng = np.random.default_rng(0)
    # Bound peak memory even if the pilot has many independent shoots.
    values = []
    for start in range(0, 2000, 128):
        draws = rng.integers(len(sizes), size=(min(128, 2000 - start), len(sizes)))
        values.extend(successes[draws].sum(axis=1) / sizes[draws].sum(axis=1))
    low, high = np.quantile(values, [0.025, 0.975])
    return {'low': float(low), 'high': float(high), 'source_groups': len(groups),
            'method': 'percentile bootstrap by source_group (2000 draws, seed 0)'}


def class_report(index, manifest, encoder, *, split='validation', archive_sha256=None):
    """Compute raw Top-1/5 ranking on an independently annotated manifest.

    Validation/test is selected only after whole-manifest audit. No thresholds,
    unknown rejection, or exact-SKU acceptance is claimed by these metrics.
    """
    rows, manifest_sha256 = audit_class_queries(index, manifest)
    if split not in ('validation', 'test'):
        raise ValueError('invalid evaluation split')
    selected = [row for row in rows if row['split'] == split]
    if not selected:
        raise ValueError(f'no {split} queries')
    index.check_encoder(encoder)
    collisions = prompt_collisions(index.items)
    normalized = _prompt_groups(index, normalized=True)
    batch_size = getattr(encoder, 'batch_size', 8)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('invalid encoder batch size')
    output = []
    root = Path(manifest).resolve().parent
    for start in range(0, len(selected), batch_size):
        batch = selected[start:start + batch_size]
        images = []
        hashes = []
        try:
            for row in batch:
                image, digest = read_image(root / row['image_path'])
                if digest != row['image_sha256']:
                    image.close()
                    raise ValueError(f'image hash changed after audit: {row["query_id"]}')
                images.append(image)
                # pHash of the decoded frame supports the ТЗ §8 hard-pair slice.
                hashes.append(phash(image))
            ranking_batch = index.classify(images, encoder, top_k=5)
            for row, ranking, phash_value in zip(batch, ranking_batch, hashes):
                gt = row['gt_slug'] if row['gt_status'] == 'known' else None
                top_id = ranking[0]['id']
                tied = collisions.get(top_id, [])
                gt_group = collisions.get(gt, []) if gt is not None else []
                normalized_gt_group = normalized.get(gt, []) if gt is not None else []
                output.append({'query_id': row['query_id'], 'image_sha256': row['image_sha256'],
                               'source_group': row['source_group'], 'split': split,
                               'gt_status': row['gt_status'], 'gt_slug': gt,
                               'phash': phash_value,
                               'ranking': ranking, 'raw_top1': top_id,
                               'top1_correct': gt is not None and top_id == gt,
                               'top5_correct': gt is not None and gt in [c['id'] for c in ranking],
                               'status': 'ambiguous_prompt' if tied else 'ranked_only',
                               'predicted_class_id': None if tied else top_id,
                               'indistinguishable_class_ids': tied,
                               'gt_prompt_collision': gt_group,
                               'gt_normalized_prompt_collision': normalized_gt_group,
                               'wrong_top1_outside_gt_prompt_group': bool(
                                   gt_group and top_id not in gt_group)})
        finally:
            for image in images:
                image.close()
    known = [row for row in output if row['gt_status'] == 'known']
    unique = [row for row in known if not row['gt_prompt_collision']]
    gt_colliding = [row for row in known if row['gt_prompt_collision']]
    # ТЗ §8: Recall@1 restricted to queries in a pHash 8-16 hard pair (visually
    # similar but distinct labels). Diagnostic slice, not a separate decision.
    known_ids = [row['query_id'] for row in known]
    known_hashes = [row['phash'] for row in known]
    band_pairs = distance_band_pairs(known_hashes, known_ids, low=8, high=16)
    hard_ids = {known_ids[i] for i, _, _ in band_pairs} | {known_ids[j] for _, j, _ in band_pairs}
    hard_rows = [row for row in known if row['query_id'] in hard_ids]
    config = {'encoder': index.meta['encoder'], 'prompt_variant': index.meta['prompt_variant'],
              'ensemble': index.meta['ensemble'], 'source_sha256': index.meta.get('source_sha256'),
              'class_ids_sha256': hashlib.sha256(json.dumps(
                  sorted(item['id'] for item in index.items), ensure_ascii=False).encode()).hexdigest(),
              'class_count': len(index.items), 'image_mode': IMAGE_MODE,
              'query_manifest_sha256': manifest_sha256, 'index_archive_sha256': archive_sha256,
              'split': split}
    return {'schema': SCHEMA, 'config': config,
            'prompt_collisions': {'exact': _collision_stats(collisions, len(index.items)),
                                  'normalized': _collision_stats(normalized, len(index.items))},
            'metrics': {
                'total': len(output), 'known': len(known),
                'unknown': sum(r['gt_status'] == 'unknown' for r in output),
                'ambiguous_gt': sum(r['gt_status'] == 'ambiguous' for r in output),
                'known_source_groups': len({r['source_group'] for r in known}),
                'raw_top1_known': fraction(sum(r['top1_correct'] for r in known), len(known)),
                'raw_top5_known': fraction(sum(r['top5_correct'] for r in known), len(known)),
                'raw_top1_known_group_interval': _group_interval(known, 'top1_correct'),
                'raw_top5_known_group_interval': _group_interval(known, 'top5_correct'),
                'raw_top1_known_phash_band_8_16': fraction(
                    sum(r['top1_correct'] for r in hard_rows), len(hard_rows)),
                'phash_band_8_16_pairs': len(band_pairs),
                'unique_prompt_top1': fraction(sum(r['top1_correct'] for r in unique), len(unique)),
                'known_gt_in_prompt_collision': fraction(len(gt_colliding), len(known)),
                'known_gt_in_normalized_prompt_collision': fraction(
                    sum(bool(r['gt_normalized_prompt_collision']) for r in known), len(known)),
                'wrong_top1_outside_gt_prompt_group': fraction(
                    sum(r['wrong_top1_outside_gt_prompt_group'] for r in gt_colliding), len(gt_colliding)),
                'top1_ambiguous_prompt': fraction(sum(r['status'] == 'ambiguous_prompt' for r in output),
                                                  len(output)),
                'statuses': dict(Counter(r['status'] for r in output))},
            'queries': output,
            'note': 'Raw ranks (including arbitrary exact-prompt tie order), not SKU acceptance; '
                    'no unknown false-accept metric or calibrated confidence. '
                    'Intervals describe source-group resampling, not deployment guarantees.'}


def compare_class_reports(left, right):
    """Paired ranking comparison; enforce same images, GT, encoder and class set."""
    if left.get('schema') != SCHEMA or right.get('schema') != SCHEMA:
        raise ValueError('invalid class report schema')
    a, b = left['config'], right['config']
    source = a.get('source_sha256')
    if not isinstance(source, str) or len(source) != 64 or any(c not in '0123456789abcdef' for c in source):
        raise ValueError('paired comparison requires a SHA256 catalog/class source snapshot')
    for key in ('encoder', 'image_mode', 'query_manifest_sha256', 'source_sha256',
                'class_ids_sha256', 'class_count', 'split'):
        if a.get(key) != b.get(key):
            raise ValueError(f'paired config mismatch: {key}')
    def keyed(report):
        result = {}
        for row in report['queries']:
            qid = row['query_id']
            if qid in result:
                raise ValueError(f'duplicate query_id: {qid}')
            result[qid] = row
        return result
    old, new = keyed(left), keyed(right)
    if not old or old.keys() != new.keys():
        raise ValueError('different or empty paired query sets')
    for qid in old:
        if any(old[qid].get(key) != new[qid].get(key) for key in (
                'image_sha256', 'source_group', 'split', 'gt_status', 'gt_slug')):
            raise ValueError(f'paired image/GT mismatch: {qid}')
    known = [qid for qid, row in old.items() if row['gt_status'] == 'known']
    helped = [qid for qid in known if not old[qid]['top1_correct'] and new[qid]['top1_correct']]
    harmed = [qid for qid in known if old[qid]['top1_correct'] and not new[qid]['top1_correct']]
    return {'known': len(known), 'left': left['metrics'], 'right': right['metrics'],
            'helped_query_ids': helped, 'harmed_query_ids': harmed,
            'helped_among_known': fraction(len(helped), len(known)),
            'harmed_among_known': fraction(len(harmed), len(known)),
            'note': 'Paired raw ranking only; not an acceptance-policy comparison.'}
