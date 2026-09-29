"""Paired offline CLIP ablation. No visual accept without a validation-derived policy.

Only independently annotated manual query ROI and audited reference gallery are
eligible. The three unlabelled eval/ photos are not validation data.
"""
import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

from .data import jsonl, query_manifest_errors, calibration_test_leakage_errors
from .report import report
from .roi import manual_roi
from .visual import (CLIPEmbedder, Gallery, IncompatibleGallery, MODEL, PRETRAINED,
                     PREPROCESS, CROP_VERSION, build_gallery, _unit_vectors)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class VisualPolicy:
    version: str
    min_text_score: float
    min_cosine: float
    min_margin: float
    config: dict
    calibration_manifest_sha256: str
    calibration_source_groups: list
    artifact_sha256: str

    def __post_init__(self):
        if (self.version != 'visual-policy-v1' or not all(math.isfinite(x) for x in
                (self.min_text_score, self.min_cosine, self.min_margin))
                or not 0 <= self.min_text_score <= 100
                or not -1 <= self.min_cosine <= 1 or not 0 <= self.min_margin <= 2
                or not isinstance(self.config, dict)
                or not isinstance(self.calibration_manifest_sha256, str)
                or len(self.calibration_manifest_sha256) != 64
                or not isinstance(self.calibration_source_groups, list)
                or not self.calibration_source_groups
                or any(not isinstance(g, str) or not g for g in self.calibration_source_groups)
                or len(set(self.calibration_source_groups)) != len(self.calibration_source_groups)
                or self.artifact_sha256 != fingerprint(self.payload())):
            raise ValueError('invalid visual policy artifact or checksum')

    def payload(self):
        return {k: getattr(self, k) for k in ('version', 'min_text_score', 'min_cosine',
                'min_margin', 'config', 'calibration_manifest_sha256',
                'calibration_source_groups')}


def audit_queries(rows, gallery, root):
    """Audit the entire manifest before choosing a split (including actual image bytes)."""
    for row in rows:
        qid = row.get('query_id')
        photo = (root / str(row.get('image_path', ''))).resolve()
        if not photo.is_relative_to(root):
            raise ValueError(f'unsafe image path: {qid}')
        if hashlib.sha256(photo.read_bytes()).hexdigest() != row.get('image_sha256'):
            raise ValueError(f'image hash mismatch: {qid}')
    errors, groups = query_manifest_errors(
        rows, gallery_hashes={s['image_sha256'] for s in gallery.sources},
        gallery_groups={s['source_group'] for s in gallery.sources})
    if errors:
        raise ValueError(errors[0])
    return groups


def rank(candidates, observations, weight=0.5):
    """Return CLIP and linear-fusion rankings, or None when Top-K has missing refs.

    Full coverage is required to avoid rewarding missing data as zero similarity.
    The fusion score is not a calibrated probability.
    """
    if not isinstance(weight, (float, int)) or not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('weight must be in [0,1]')
    if not candidates:
        return None, None
    if any(c['wine_id'] not in observations or observations[c['wine_id']].get('cosine') is None
           for c in candidates):
        return None, None
    if any(not isinstance(c['score'], (int, float)) or not math.isfinite(c['score'])
           or not 0 <= c['score'] <= 100 or
           not isinstance(observations[c['wine_id']]['cosine'], (int, float)) or
           not math.isfinite(observations[c['wine_id']]['cosine']) or
           not -1.001 <= observations[c['wine_id']]['cosine'] <= 1.001 for c in candidates):
        raise ValueError('invalid candidate score or cosine')
    by_clip = sorted(candidates, key=lambda c: (-observations[c['wine_id']]['cosine'], c['rank']))
    by_fusion = sorted(candidates, key=lambda c: (
        -((1-weight)*c['score']/100 + weight*(observations[c['wine_id']]['cosine']+1)/2), c['rank']))
    return [c['slug'] for c in by_clip], [c['slug'] for c in by_fusion]


def visual_decision(candidates, observations, ranking, policy):
    if not candidates:
        return 'unknown', None, ['no_text_candidates']
    if ranking is None:
        return 'ambiguous', None, ['missing_reference']
    if policy is None:
        return 'ambiguous', None, ['visual_policy_not_calibrated']
    by_slug = {c['slug']: c for c in candidates}
    top = by_slug[ranking[0]]
    cosine = observations[top['wine_id']]['cosine']
    if top['score'] < policy.min_text_score or cosine < policy.min_cosine:
        return 'unknown', None, ['insufficient_text_or_visual_match']
    # A singleton has no visual competitor and cannot establish distinctiveness.
    if len(ranking) == 1:
        return 'ambiguous', None, ['no_visual_competitor']
    competitor = max(observations[by_slug[slug]['wine_id']]['cosine']
                     for slug in ranking if slug != top['slug'])
    if cosine - competitor < policy.min_margin:
        return 'ambiguous', None, ['close_visual_candidates']
    return 'accepted', top['slug'], ['experimental_visual_policy']


def evaluate(queries, predictions, gallery, embed, *, split='validation', role='front',
             mode='fusion', weight=0.5, policy=None, top_k=20, calibration_manifest=None):
    """Pair baseline JSONL with annotated query ROI; reject leakage and version drift."""
    if mode not in ('fusion', 'clip') or not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('invalid mode or weight')
    if role not in ('front', 'back', 'neck') or split not in ('validation', 'test'):
        raise ValueError('invalid role or split')
    queries_path = Path(queries).resolve()
    if type(top_k) is not int or top_k < 1:
        raise ValueError('invalid top_k')
    all_queries = jsonl(queries_path)
    audit_queries(all_queries, gallery, queries_path.parent)
    query_rows = [r for r in all_queries if r['split'] == split]
    if policy is not None:
        if calibration_manifest is None:
            raise ValueError('calibration manifest required for policy')
        calibration_path = Path(calibration_manifest).resolve()
        if hashlib.sha256(calibration_path.read_bytes()).hexdigest() != policy.calibration_manifest_sha256:
            raise ValueError('calibration manifest hash mismatch')
        calibration_rows = jsonl(calibration_path)
        calibration_groups = audit_queries(calibration_rows, gallery, calibration_path.parent)
        if (set(calibration_groups) != set(policy.calibration_source_groups)
                or any(s != 'validation' for s in calibration_groups.values())):
            raise ValueError('invalid calibration source groups/split')
        config = {'mode': mode, 'weight': weight, 'role': role, 'top_k': top_k,
                  'model': gallery.meta['model'], 'pretrained': gallery.meta['pretrained'],
                  'preprocess': gallery.meta['preprocess'],
                  'crop_version': gallery.meta['crop_version'],
                  'gallery_manifest_sha256': gallery.meta['manifest_sha256'],
                  'gallery_archive_sha256': gallery.archive_sha256,
                  'csv_sha256': gallery.meta['csv_sha256']}
        if policy.config != config:
            raise IncompatibleGallery('visual policy configuration mismatch')
        if split == 'test':
            leakage = calibration_test_leakage_errors(
                calibration_rows, query_rows, calibration_path=calibration_path,
                test_path=queries_path)
            if leakage:
                raise ValueError(leakage[0])
    query_manifest_sha256 = hashlib.sha256(queries_path.read_bytes()).hexdigest()
    baseline_sha256 = hashlib.sha256(Path(predictions).read_bytes()).hexdigest()
    prediction_rows = jsonl(predictions)
    baseline = {}
    for row in prediction_rows:
        if row.get('split') != split:
            continue
        qid = row['query_id']
        if qid in baseline:
            raise ValueError(f'duplicate prediction: {qid}')
        baseline[qid] = row
    if not query_rows or set(baseline) != {r['query_id'] for r in query_rows} or len(query_rows) != len(baseline):
        raise ValueError('empty, duplicate or unmatched query set')
    asset_hashes = {s['image_sha256'] for s in gallery.sources}
    asset_groups = {s['source_group'] for s in gallery.sources}
    output = []
    for row in query_rows:
        start = time.monotonic()
        qid = row['query_id']
        original = baseline[qid]
        if row.get('gt_status') not in ('known', 'unknown', 'ambiguous') or (
                row.get('gt_status') == 'known' and (not row.get('gt_slug') or not row.get('gt_source'))) or (
                row.get('gt_status') != 'known' and row.get('gt_slug') is not None):
            raise ValueError(f'invalid GT: {qid}')
        if (original.get('gt_slug') != row.get('gt_slug') or
                original.get('gt_status') != row.get('gt_status') or
                original.get('source_group') != row.get('source_group')):
            raise ValueError(f'GT/source mismatch: {qid}')
        if not row.get('source_group') or row['source_group'] in asset_groups:
            raise ValueError(f'gallery/query source-group leakage: {qid}')
        versions = original.get('versions', {})
        if versions.get('csv_sha256') != gallery.meta['csv_sha256'] or versions.get('crop') != CROP_VERSION:
            raise IncompatibleGallery(f'catalog or query crop mismatch: {qid}')
        photo = (queries_path.parent / row['image_path']).resolve()
        if not photo.is_relative_to(queries_path.parent):
            raise ValueError(f'unsafe image path: {qid}')
        data = photo.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != row.get('image_sha256') or digest != original.get('image_sha256') or digest in asset_hashes:
            raise ValueError(f'image hash mismatch or gallery/query leakage: {qid}')
        roi = manual_roi(data, tuple(row['bbox']), time.monotonic()+8)
        if (original.get('roi') or {}).get('bbox') != list(roi.bbox) or (
                original.get('roi') or {}).get('roi_id') != roi.roi_id:
            raise ValueError(f'baseline ROI mismatch: {qid}')
        candidates = original.get('candidates', [])
        if not isinstance(candidates, list) or len(candidates) > top_k or len({c['wine_id'] for c in candidates}) != len(candidates) or any(
                c['wine_id'] != c['slug'] or c['rank'] != i or
                not isinstance(c['score'], (int, float)) or not math.isfinite(c['score']) or
                not 0 <= c['score'] <= 100 for i, c in enumerate(candidates, 1)):
            raise ValueError(f'invalid text Top-K: {qid}')
        if candidates:
            vec = _unit_vectors(embed([roi.crop], model_name=gallery.meta['model'],
                                       pretrained=gallery.meta['pretrained']), 1,
                                gallery.vectors.shape[1])[0]
            observations = gallery.scores(vec, [c['wine_id'] for c in candidates], role=role)
        else:
            observations = {}
        clip, fusion = rank(candidates, observations, weight)
        ranking = clip if mode == 'clip' else fusion
        status, slug, reasons = visual_decision(candidates, observations, ranking, policy)
        output.append({'query_id': qid, 'image_sha256': digest, 'split': split,
                       'gt_slug': row.get('gt_slug'), 'gt_status': row.get('gt_status'),
                       'source_group': row['source_group'], 'status': status, 'slug': slug,
                       'run_config': {'mode': mode, 'weight': weight, 'role': role, 'top_k': top_k,
                                      'gallery_archive_sha256': gallery.archive_sha256,
                                      'gallery_manifest_sha256': gallery.meta['manifest_sha256'],
                                      'queries_manifest_sha256': query_manifest_sha256,
                                      'baseline_sha256': baseline_sha256,
                                      'policy': {**policy.payload(), 'artifact_sha256': policy.artifact_sha256}
                                      if policy else None},
                       'reason_codes': reasons, 'candidates': candidates,
                       'clip_scores': observations, 'clip_reranked': clip, 'fusion_reranked': fusion,
                       'confidence_calibrated': None,
                       'versions': {**versions, 'clip_model': gallery.meta['model'],
                                    'clip_pretrained': gallery.meta['pretrained'],
                                    'clip_preprocess': PREPROCESS, 'gallery_manifest_sha256': gallery.meta['manifest_sha256'],
                                    'visual_policy': policy.version if policy else None},
                       'timings_ms': {'clip_stage': round((time.monotonic()-start)*1000),
                                      'total': original.get('timings_ms', {}).get('total', 0) +
                                               round((time.monotonic()-start)*1000)}})
    return output


def comparison(baseline, visual, mode='fusion'):
    """Paired retrieval/decision report; oracle/mock rows never prove OCR accuracy."""
    old = {r['query_id']: r for r in baseline}
    new = {r['query_id']: r for r in visual}
    if (not old or old.keys() != new.keys() or len(old) != len(baseline)
            or len(new) != len(visual)):
        raise ValueError('duplicate or unmatched paired results')
    if mode not in ('clip', 'fusion'):
        raise ValueError('invalid mode')
    for qid in old:
        if any(old[qid].get(key) != new[qid].get(key)
               for key in ('gt_slug', 'gt_status', 'source_group', 'image_sha256', 'split')):
            raise ValueError(f'paired GT/source mismatch: {qid}')
    name = 'clip_reranked' if mode == 'clip' else 'fusion_reranked'
    eligible = [qid for qid, r in new.items() if r[name] is not None and r.get('gt_slug')]
    gt_in_k = [qid for qid in eligible if new[qid]['gt_slug'] in [c['slug'] for c in old[qid]['candidates']]]
    def top1_old(qid):
        return old[qid]['candidates'][0]['slug'] if old[qid]['candidates'] else None
    def fraction(n, d):
        return {'numerator': n, 'denominator': d, 'value': n/d if d else None}
    all_known = [qid for qid, r in new.items() if r.get('gt_slug')]
    known_in_k = [qid for qid in all_known if new[qid]['gt_slug'] in
                  [c['slug'] for c in old[qid]['candidates']]]
    missing = {qid: [wine for wine, score in (r.get('clip_scores') or {}).items()
                     if score.get('cosine') is None] for qid, r in new.items() if r[name] is None}
    harmed = [qid for qid in gt_in_k if top1_old(qid) == new[qid]['gt_slug']
              and new[qid][name][0] != new[qid]['gt_slug']]
    helped = [qid for qid in gt_in_k if top1_old(qid) != new[qid]['gt_slug']
              and new[qid][name][0] == new[qid]['gt_slug']]
    correct_text = [qid for qid in all_known if top1_old(qid) == new[qid]['gt_slug']]
    return {'baseline': report(list(old.values())), 'visual': report(list(new.values())),
            'text_recall_at_k': fraction(len(known_in_k), len(all_known)),
            'eligible_text_recall_at_k': fraction(len(gt_in_k), len(eligible)),
            'missing_reference_by_query': missing,
            'harmed_among_correct_text_top1': fraction(len(harmed), len(correct_text)),
            'helped_among_eligible': fraction(len(helped), len(gt_in_k)),
            'visual_coverage': fraction(sum(r[name] is not None for r in new.values()), len(new)),
            'gt_in_text_top_k': len(gt_in_k),
            'conditional_top1_at_gt_in_k': {
                'text': fraction(sum(top1_old(qid) == new[qid]['gt_slug'] for qid in gt_in_k), len(gt_in_k)),
                'visual': fraction(sum(new[qid][name][0] == new[qid]['gt_slug'] for qid in gt_in_k), len(gt_in_k))},
            'unconditional_retrieval_top1': {
                'text': fraction(sum(top1_old(qid) == new[qid]['gt_slug'] for qid in all_known), len(all_known)),
                'visual': fraction(sum(new[qid][name] is not None and new[qid][name][0] == new[qid]['gt_slug']
                                       for qid in all_known), len(all_known))},
            'harmed_query_ids': harmed,
            'helped_query_ids': helped,
            'oracle_text_included': any((r.get('ocr') or {}).get('model') == 'mock' for r in old.values()),
            'note': 'Cosine/fusion scores are not probabilities; mock text is not OCR evidence.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    b.add_argument('gallery_manifest')
    b.add_argument('output_npz')
    b.add_argument('--csv', default='strapi_output0709.csv')
    b.add_argument('--model', default=MODEL)
    b.add_argument('--revision', '--pretrained', dest='pretrained', default=PRETRAINED)
    e = sub.add_parser('evaluate')
    e.add_argument('gallery_npz')
    e.add_argument('queries_manifest')
    e.add_argument('baseline_jsonl')
    e.add_argument('output_jsonl')
    e.add_argument('--csv', default='strapi_output0709.csv')
    e.add_argument('--model', default=MODEL)
    e.add_argument('--revision', '--pretrained', dest='pretrained', default=PRETRAINED)
    e.add_argument('--split', choices=['validation', 'test'], default='validation')
    e.add_argument('--role', choices=['front', 'back', 'neck'], default='front')
    e.add_argument('--mode', choices=['clip', 'fusion'], default='fusion')
    e.add_argument('--weight', type=float, default=0.5)
    e.add_argument('--top-k', type=int, default=20)
    e.add_argument('--policy', help='validation-derived VisualPolicy JSON (no defaults)')
    e.add_argument('--calibration-manifest', help='validation manifest used for policy')
    for command in (b, e):
        command.add_argument('--allow-remote-code', action='store_true',
                             help='approve HF code/downloads after CC BY-NC 4.0 license review')
        command.add_argument('--device', default=None)
        command.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    if not args.allow_remote_code:
        parser.error('review Jina license and HF code first; pass --allow-remote-code to run')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    from .catalog import load_catalog
    wines, stats = load_catalog(args.csv)
    if args.command == 'build':
        meta = build_gallery(args.gallery_manifest, wines, args.output_npz,
                             model_name=args.model, pretrained=args.pretrained,
                             batch_size=args.batch_size, device=args.device, allow_remote_code=True)
        print(json.dumps({k: v for k, v in meta.items() if k != 'sources'}, ensure_ascii=False))
        return
    gallery = Gallery(args.gallery_npz, model=args.model, pretrained=args.pretrained,
                      csv_sha256=stats['csv_sha256'])
    policy = VisualPolicy(**json.loads(Path(args.policy).read_text())) if args.policy else None
    rows = evaluate(args.queries_manifest, args.baseline_jsonl, gallery,
                    CLIPEmbedder(args.model, args.pretrained, device=args.device,
                                 batch_size=args.batch_size, allow_remote_code=True), split=args.split,
                    role=args.role, mode=args.mode, weight=args.weight, policy=policy,
                    top_k=args.top_k, calibration_manifest=args.calibration_manifest)
    out = Path(args.output_jsonl).resolve()
    if out.exists() or out in (Path(args.queries_manifest).resolve(), Path(args.baseline_jsonl).resolve()):
        parser.error('output must be new and distinct from inputs')
    with out.open('x', encoding='utf-8') as dest:
        for row in rows:
            dest.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    baseline = [r for r in jsonl(args.baseline_jsonl) if r.get('split') == args.split]
    print(json.dumps(comparison(baseline, rows, args.mode), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
