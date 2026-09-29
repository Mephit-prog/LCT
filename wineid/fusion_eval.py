"""Offline paired fusion evaluator over *independently annotated* query manifests.

Evidence JSONL: query_id, evidence:[CandidateEvidence dicts], ocr_text,
latency_ms (optional). Predictions must be recorded without GT lookup. `tune`
only works on validation and never labels synthetic OCR as a real-world metric.
"""
import argparse
import hashlib
import json
import math
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

from .blend import Blender, BlendConfig, CandidateEvidence
from .blend_decision import rank_and_decide
from .canonical import ProducerAliases
from .fusion_policy import FusionPolicy
from .search import BLEND_VERSION
from .text_policy import fingerprint, load_policy
from .catalog import load_catalog
from .data import jsonl, query_manifest_errors, calibration_test_leakage_errors
from .roi import MAX_BYTES, decode


def checked_queries(path, wines, gallery_manifest=None):
    if gallery_manifest:
        from .data import audit
        checked = audit(wines, gallery_manifest, path)
        if checked['errors']:
            raise ValueError(checked['errors'][0])
    rows = jsonl(path)
    gallery = jsonl(gallery_manifest) if gallery_manifest else []
    errors, _ = query_manifest_errors(rows, catalog_ids={w.slug for w in wines},
        gallery_hashes={r.get('image_sha256') for r in gallery},
        gallery_groups={r.get('source_group') for r in gallery})
    if errors:
        raise ValueError(errors[0])
    root = Path(path).resolve().parent
    for row in rows:
        location = (root / str(row.get('image_path', ''))).resolve()
        if not location.is_relative_to(root) or not location.is_file() or location.stat().st_size > MAX_BYTES:
            raise ValueError(f"invalid query image: {row['query_id']}")
        image_bytes = location.read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != row['image_sha256']:
            raise ValueError(f"image hash mismatch: {row['query_id']}")
        decoded, _ = decode(image_bytes, time.monotonic() + 8)
        decoded.close()
    return rows


def _ratio(a, b):
    return {'numerator': a, 'denominator': b, 'value': a / b if b else None}


def verify_runtime_evidence(queries, saved_rows, pipeline, *, root, seconds):
    """Re-infer from checked manifest bytes; refuse forged/self-hashed JSONL.

    Only return the fresh observations to the evaluator. Hashes inside saved
    evidence are not signatures. Never use manual bbox or oracle transcripts for
    the calibrated HTTP acceptance profile.
    """
    if not isinstance(pipeline.policy, FusionPolicy) or pipeline.blender is None:
        raise ValueError('fusion policy runtime required')
    if any(q.get('bbox') is not None or q.get('manual_text') is not None for q in queries):
        raise ValueError('policy evaluation forbids manual bbox/oracle text')
    from .fusion_run import observe_queries
    mode = pipeline.roi_mode if pipeline.roi_mode != 'refuse' else 'manual'
    if pipeline.ocr is not None and mode == 'manual':
        raise ValueError('policy evaluation needs a calibrated automatic OCR ROI')
    verified = observe_queries(queries, root=root, index=pipeline.index,
        blender=pipeline.blender, vision=pipeline.vision, ocr=pipeline.ocr,
        ocr_mode='live' if pipeline.ocr is not None else 'none', roi_mode=mode,
        seconds=seconds, runtime_profile=pipeline.fusion_config, pipeline=pipeline)
    saved = {r.get('query_id'): r for r in saved_rows}
    if len(saved) != len(saved_rows) or set(saved) != {r['query_id'] for r in verified}:
        raise ValueError('query/evidence id mismatch')
    for actual in verified:
        original = saved[actual['query_id']]
        keys = ('image_sha256', 'catalog_sha256', 'fusion_config_sha256',
                'evidence', 'ocr_text', 'ocr_status', 'reason_codes', 'roi',
                'online_status', 'online_slug')
        if any(json.dumps(original.get(key), ensure_ascii=False, sort_keys=True) !=
               json.dumps(actual[key], ensure_ascii=False, sort_keys=True)
               for key in keys):
            raise ValueError('stored evidence differs from runtime re-inference')
    return verified


def evaluate(wines, queries, evidence_rows, *, config=None, split='validation',
             policy=None, pipeline=None, aliases=None,
             calibration_manifest=None, test_manifest=None):
    """Evaluate one split. Acceptance requires the identical runtime profile.

    Diagnostic ranking without a policy still accepts old evidence formats. A
    policy caller must use trusted observations: hashes inside a JSONL file are
    NOT authentication. The CLI re-infers from the manifest before evaluation.
    """
    if split not in ('validation', 'test'):
        raise ValueError('split must be validation or test')
    if policy is not None and split == 'test':
        if not calibration_manifest or not test_manifest:
            raise ValueError('test acceptance requires calibration/test manifest leakage audit')
        cal_path = Path(calibration_manifest)
        if hashlib.sha256(cal_path.read_bytes()).hexdigest() != policy.calibration_manifest_sha256:
            raise ValueError('calibration manifest hash mismatch')
        errors = calibration_test_leakage_errors(jsonl(cal_path), jsonl(test_manifest),
            calibration_path=cal_path, test_path=test_manifest)
        if errors:
            raise ValueError(errors[0])
    if policy is not None:
        if not isinstance(policy, FusionPolicy) or pipeline is None or pipeline.policy != policy:
            raise ValueError('acceptance requires a pipeline with the same fusion policy')
        if {w.slug for w in pipeline.index.wines} != {w.slug for w in wines}:
            raise ValueError('catalog mismatch')
        policy.check_config(pipeline.fusion_config)
        blender = pipeline.blender
        if blender is None or (config is not None and config != blender.config):
            raise ValueError('blend configuration mismatch')
        classes = getattr(pipeline.vision, 'classes', None)
    else:
        blender = Blender(wines, aliases=aliases, config=config)
        classes = None
    by_id = {}
    for row in evidence_rows:
        qid = row.get('query_id')
        if not qid or qid in by_id:
            raise ValueError('missing/duplicate evidence query_id')
        by_id[qid] = row
    ids = {q['query_id'] for q in queries}
    if ids != set(by_id):
        raise ValueError('query/evidence id mismatch')
    if policy is not None:
        expected = {'class': getattr(pipeline.vision, 'class_fingerprint', None),
                    'photo': (pipeline.vision.gallery.archive_sha256 if pipeline.vision and
                              pipeline.vision.gallery else None),
                    'text': BLEND_VERSION}
        for query in queries:
            data = by_id[query['query_id']]
            if (data.get('image_sha256') != query['image_sha256']
                    or data.get('catalog_sha256') != pipeline.csv_sha256
                    or data.get('fusion_config_sha256') != fingerprint(pipeline.fusion_config)
                    or data.get('ocr_status') == 'manual_oracle'):
                raise ValueError('evidence image/catalog/config mismatch or oracle text')
            for observation in data['evidence']:
                if observation.get('provenance') != expected.get(observation.get('source')):
                    raise ValueError('evidence source provenance mismatch')
    selected = [q for q in queries if q['split'] == split]
    if not selected:
        raise ValueError('empty split')
    known = correct1 = correct5 = family = member = member_n = union = 0
    raw_pool = budget_pool = 0
    unknown = false_accept = accepted = accepted_known = accepted_correct = ambiguous = errors_count = 0
    margins_right, margins_wrong, latencies, predictions = [], [], [], []
    missing = {'text': 0, 'photo': 0, 'class': 0}
    for query in selected:
        data = by_id[query['query_id']]
        observations = [CandidateEvidence(**e) for e in data['evidence']]
        present = {o.source for o in observations}
        for source in missing:
            missing[source] += source not in present
        ranked, status, slug, _, _ = rank_and_decide(
            blender, observations, data.get('ocr_text'), policy=policy, classes=classes,
            config=pipeline.fusion_config if policy else None)
        if policy and data.get('online_status') is not None:
            # Re-inferred observations also carry the online refusal/timeout /
            # capacity decision. Ranking alone must not override that decision.
            status, slug = data['online_status'], data.get('online_slug')
        elif policy and 'deadline' in data.get('failures', []):
            status, slug = 'error', None
        top = ranked[0]['slug'] if ranked else None
        slugs = [r['slug'] for r in ranked]
        margin = (ranked[0]['blend_score'] - ranked[1]['blend_score'] if len(ranked) > 1 else None)
        if status == 'accepted':
            accepted += 1
        if query['gt_status'] == 'ambiguous':
            ambiguous += 1
        if data.get('failures'):
            errors_count += 1
        gt = query.get('gt_slug')
        if query['gt_status'] == 'known':
            known += 1
            correct1 += gt == top
            correct5 += gt in slugs[:5]
            observed_slugs = {o.slug for o in observations if o.rank <= blender.config.top_k}
            union += gt in observed_slugs
            raw_pool += gt in {o.slug for o in observations}
            budget_pool += gt in slugs
            same_family = top is not None and blender.key[top] == blender.key[gt]
            family += same_family
            if same_family and len(blender.families[blender.key[gt]]) > 1:
                member_n += 1
                member += top == gt
            if status == 'accepted':
                accepted_known += 1
                accepted_correct += gt == slug
            if margin is not None:
                (margins_right if gt == top else margins_wrong).append(margin)
        elif query['gt_status'] == 'unknown':
            unknown += 1
            false_accept += status == 'accepted'
        if isinstance(data.get('latency_ms'), (float, int)):
            latencies.append(data['latency_ms'])
        predictions.append({'query_id': query['query_id'], 'gt_status': query['gt_status'],
                            'gt_slug': gt, 'predicted_slug': top, 'status': status,
                            'accepted_slug': slug, 'margin': margin, 'top5': slugs[:5]})
    latencies.sort()
    metrics = {'split': split, 'queries': len(selected), 'known': known,
               'ambiguous': ambiguous, 'queries_with_source_failures': errors_count,
               'accuracy_at_1': _ratio(correct1, known), 'accuracy_at_5': _ratio(correct5, known),
               'union_recall': _ratio(union, known),
               'raw_observation_recall': _ratio(raw_pool, known),
               'after_family_and_budget_recall': _ratio(budget_pool, known),
               'family_accuracy_at_1': _ratio(family, known),
               'member_accuracy_given_correct_multi_sku_family': _ratio(member, member_n),
               'accepted_precision': _ratio(accepted_correct, accepted),
               'accepted_coverage_known': _ratio(accepted_known, known),
               'unknown_false_accept': _ratio(false_accept, unknown),
               'missing_sources': missing,
               'mean_margin_correct': statistics.mean(margins_right) if margins_right else None,
               'mean_margin_wrong': statistics.mean(margins_wrong) if margins_wrong else None,
               'latency_p95_ms': latencies[max(0, int(.95 * len(latencies) + .9999) - 1)] if latencies else None,
               'predictions': predictions}
    return metrics


def calibrate_temperature(wines, queries, evidence_rows, *, config=None, aliases=None):
    """Fit conditional-on-pool log-loss on validation known wines with GT in pool.

    A separate unknown-detection calibration is still needed before deployment.
    """
    if any(q['split'] != 'validation' for q in queries):
        raise ValueError('temperature calibration requires validation only')
    by_id = {r['query_id']: r for r in evidence_rows}
    if len(by_id) != len(evidence_rows) or set(by_id) != {q['query_id'] for q in queries}:
        raise ValueError('query/evidence mismatch')
    blender = Blender(wines, aliases=aliases, config=config)
    entries = []
    known = 0
    for row in queries:
        if row['gt_status'] != 'known':
            continue
        known += 1
        data = by_id[row['query_id']]
        ranked = blender.rank([CandidateEvidence(**e) for e in data['evidence']],
                              data.get('ocr_text'))
        slugs = [r['slug'] for r in ranked]
        if row['gt_slug'] in slugs:
            entries.append(([r['blend_score'] for r in ranked], slugs.index(row['gt_slug'])))
    if not entries:
        raise ValueError('no ground truth in candidate pool')
    grid = (.1, .2, .5, 1., 2., 5., 10.)
    def loss(t):
        return sum(math.log(sum(math.exp((x - max(scores)) / t) for x in scores)) -
                   (scores[target] - max(scores)) / t for scores, target in entries) / len(entries)
    best = min(grid, key=lambda t: (loss(t), t))
    return {'temperature': best, 'conditional_log_loss': loss(best),
            'known_in_pool': len(entries), 'known_total': known,
            'note': 'Not calibrated for catalog-unknown queries or missed pool GT.'}


def tune_validation(wines, queries, evidence_rows, *, config=None, aliases=None):
    """Small deterministic ablation grid; do NOT use test/eval for fitting."""
    if any(q['split'] != 'validation' for q in queries):
        raise ValueError('tuning requires a validation-only manifest')
    base = config or BlendConfig()
    results = []
    for img in (.5, 1., 2.):
        for txt in (.3, .7, 1.):
            for mode in ('gap', 'rrf'):
                weights = {**base.weights, 'photo': img, 'text': txt}
                trial = replace(base, weights=weights, mode=mode)
                result = evaluate(wines, queries, evidence_rows, config=trial, aliases=aliases)
                results.append((trial, result))
    results.sort(key=lambda pair: (-(pair[1]['accuracy_at_1']['value'] or 0),
                                    -(pair[1]['family_accuracy_at_1']['value'] or 0),
                                    pair[0].mode, pair[0].weights['photo'], pair[0].weights['text']))
    return [{'config': asdict(cfg), 'accuracy_at_1': stats['accuracy_at_1'],
             'family_accuracy_at_1': stats['family_accuracy_at_1']}
            for cfg, stats in results]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('queries', help='audited manifest with independent GT')
    parser.add_argument('evidence', help='raw observations, one row per query_id')
    parser.add_argument('--csv', default='strapi_output0709.csv')
    parser.add_argument('--gallery-manifest', help='optional verified gallery provenance')
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--config', help='frozen JSON BlendConfig for test')
    parser.add_argument('--aliases', default='producer_aliases.txt', help='runtime producer aliases')
    parser.add_argument('--policy', help='v4 fusion acceptance policy; re-infers every query from bytes with WINE_* runtime')
    parser.add_argument('--tune', action='store_true', help='validation only')
    parser.add_argument('--calibrate-temperature', action='store_true', help='validation only')
    parser.add_argument('--calibration-manifest', help='reject calibration/test leakage')
    args = parser.parse_args()
    wines, _ = load_catalog(args.csv)
    queries = checked_queries(args.queries, wines, args.gallery_manifest)
    if args.calibration_manifest:
        calibration = checked_queries(args.calibration_manifest, wines, args.gallery_manifest)
        errors = calibration_test_leakage_errors(calibration, queries,
            calibration_path=args.calibration_manifest, test_path=args.queries)
        if errors:
            parser.error(errors[0])
    evidence_rows = jsonl(args.evidence)
    config = BlendConfig(**json.loads(Path(args.config).read_text())) if args.config else BlendConfig()
    if args.policy and (args.tune or args.calibrate_temperature):
        parser.error('policy is for frozen evaluation only')
    if args.policy and args.split == 'test' and not args.calibration_manifest:
        parser.error('test acceptance requires --calibration-manifest')
    if args.tune and args.calibrate_temperature:
        parser.error('choose one operation')
    if (args.tune or args.calibrate_temperature) and args.split != 'validation':
        parser.error('tuning and calibration require validation')
    aliases = ProducerAliases.load(args.aliases) if Path(args.aliases).is_file() else None
    if args.policy:
        from .runtime import open_runtime
        import os
        policy = load_policy(args.policy, allow_legacy=False)
        if args.calibration_manifest and (hashlib.sha256(Path(args.calibration_manifest).read_bytes()).hexdigest()
                                          != policy.calibration_manifest_sha256):
            parser.error('calibration manifest hash mismatch')
        if not isinstance(policy, FusionPolicy):
            parser.error('requires a fusion policy')
        with open_runtime({**os.environ, 'WINE_CSV': args.csv,
                           'WINE_POLICY': args.policy,
                           **({'WINE_BLEND_CONFIG': args.config} if args.config else {}),
                           'WINE_ALIASES': args.aliases}) as runtime:
            # Stored SHA fields alone cannot authenticate evidence. Re-infer all
            # audited query bytes with the running (local) release profile.
            verified = verify_runtime_evidence(queries, evidence_rows, runtime.pipeline,
                root=Path(args.queries).resolve().parent, seconds=runtime.settings.timeout)
            report = evaluate(wines, queries, verified, split=args.split,
                              policy=policy, pipeline=runtime.pipeline,
                              calibration_manifest=args.calibration_manifest,
                              test_manifest=args.queries if args.split == 'test' else None)
    else:
        report = (tune_validation(wines, queries, evidence_rows, config=config,
                                  aliases=aliases) if args.tune
                  else calibrate_temperature(wines, queries, evidence_rows, config=config,
                                              aliases=aliases)
                       if args.calibrate_temperature else
                       evaluate(wines, queries, evidence_rows, config=config, split=args.split,
                                aliases=aliases))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
