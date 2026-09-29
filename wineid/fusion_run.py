"""Run the *actual* pinned CLIP + optional OCR fusion on local photos.

Unlabelled images produce rankings, never accuracy. A labelled manifest is audited
before loading weights; its GT is only read by the evaluator, not by inference.
No candidate gallery is constructed from Strapi filename matches.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

from .blend import Blender, BlendConfig, CandidateEvidence, VERSION, clean_ocr
from .canonical import ProducerAliases
from .catalog import load_catalog
from .clip_zero_shot import read_image
from .fusion_eval import checked_queries, evaluate
from .data import jsonl
from .fusion_vision import load_vision
from .roi import MAX_BYTES, center_crop_80_roi, manual_roi, decode
from .pipeline import Pipeline
from .text_policy import fingerprint
from .search import BLEND_VERSION, Index


def unlabelled_queries(paths):
    """Hash and decode before loading the model; never infer a slug from a filename."""
    rows = []
    for i, path in enumerate(paths, 1):
        path = Path(path).resolve()
        image, digest = read_image(path)
        image.close()
        rows.append({'query_id': f'image-{i}', 'image_path': str(path),
                     'image_sha256': digest})
    if len({row['image_sha256'] for row in rows}) != len(rows):
        raise ValueError('duplicate image bytes; provide each query once')
    return rows


def candidate_queries(images_csv, uploads, wines, *, limit=40, seed='0'):
    """Deterministic real packshot diagnostic, NOT independent query labels.

    Manifest links CSV wines to renamed uploads by heuristic matching. Compare to
    those links only as candidate agreement, never as accuracy/validation GT.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError('positive sample limit required')
    uploads = Path(uploads).resolve()
    if not uploads.is_dir():
        raise ValueError('uploads root is missing')
    by_id = {w.wine_id: w for w in wines}
    with Path(images_csv).open(encoding='utf-8', newline='') as file:
        entries = list(csv.DictReader(file))
    if not entries or len({e['image_id'] for e in entries}) != len(entries):
        raise ValueError('missing or duplicate image IDs')
    if any(e['wine_id'] not in by_id for e in entries):
        raise ValueError('candidate image has unknown wine_id')
    entries.sort(key=lambda e: (hashlib.sha256((seed + ':' + e['image_id']).encode()).hexdigest(),
                                e['image_id']))
    chosen, seen_producers = [], set()
    for row in entries:
        producer = by_id[row['wine_id']].producer
        if producer not in seen_producers:
            chosen.append(row)
            seen_producers.add(producer)
            if len(chosen) == limit:
                break
    if len(chosen) < limit:
        chosen_ids = {e['image_id'] for e in chosen}
        chosen.extend(e for e in entries if e['image_id'] not in chosen_ids)
        chosen = chosen[:limit]
    rows = []
    for item in chosen:
        path = (uploads / item['path']).resolve()
        if not path.is_relative_to(uploads) or not path.is_file() or path.stat().st_size > MAX_BYTES:
            raise ValueError('unsafe/missing candidate upload')
        image, digest = read_image(path)
        image.close()
        if digest != item['sha256']:
            raise ValueError('candidate upload SHA mismatch')
        rows.append({'query_id': item['image_id'], 'image_path': item['path'],
                     'image_sha256': digest, '_candidate_slug': by_id[item['wine_id']].slug})
    return rows, uploads


def observe_queries(rows, *, root, index, blender, vision, ocr=None, ocr_mode='none',
                    roi_mode='manual', seconds=30., runtime_profile=None, pipeline=None):
    """Serialize all raw per-source observations, including candidates outside Top-K.

    `manual` is an oracle transcription from the manifest (not OCR accuracy).
    A provider request requires an explicit ROI mode and caller's consent.
    """
    if ocr_mode not in ('none', 'manual', 'live'):
        raise ValueError('invalid OCR mode')
    if (roi_mode not in ('manual', 'center_80_crop', 'full_frame_experimental',
                         'auto_bbox_experimental') or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or seconds <= 0):
        raise ValueError('invalid ROI mode or deadline')
    if ocr_mode == 'live' and ocr is None:
        raise ValueError('live OCR not configured')
    if vision is None and ocr_mode == 'none':
        raise ValueError('no candidate source')
    root = Path(root).resolve() if root is not None else None
    # Non-oracle evidence uses the *same* ROI/OCR/vision/search path as HTTP.
    # Manual transcripts remain a separately marked, uncalibrated diagnostic.
    if ocr_mode != 'manual':
        online = pipeline or Pipeline(index, ocr if ocr_mode == 'live' else None,
            csv_sha256=index.wines[0].csv_sha256, vision=vision, blender=blender,
            roi_mode='refuse' if roi_mode == 'manual' else roi_mode,
            request_timeout=seconds)
        if (online.index is not index or online.blender is not blender or online.vision is not vision
                or online.ocr is not (ocr if ocr_mode == 'live' else None)):
            raise ValueError('offline/online source profile mismatch')
        if online.policy is not None and seconds != online.request_timeout:
            raise ValueError('offline/online deadline profile mismatch')
    output = []
    for row in rows:
        start = time.monotonic()
        deadline = start + seconds
        path = Path(row['image_path'])
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        if root is not None and not path.is_relative_to(root):
            raise ValueError('query path outside manifest directory')
        if not path.is_file() or path.stat().st_size > MAX_BYTES:
            raise ValueError('missing or oversized query image')
        with path.open('rb') as source:
            data = source.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError('oversized query image')
        digest = hashlib.sha256(data).hexdigest()
        if row['image_sha256'] != digest:
            raise ValueError('query image hash mismatch')
        image, _ = decode(data, deadline)
        evidence = []
        failures = []
        ocr_status = 'not_run'
        text = None
        try:
            if ocr_mode != 'manual':
                bbox = row.get('bbox')
                if ocr_mode == 'live' and roi_mode == 'manual' and bbox is None:
                    raise ValueError('manual bbox required for live OCR (or opt in to center_80_crop)')
                result = online.predict(data, bbox=bbox,
                    seconds=max(0, deadline - time.monotonic()), _include_evidence=True)
                evidence = result.get('_evidence', [])
                ranked = result['candidates']
                observed = result['ocr']
                failures = [r for r in result['reason_codes'] if r.startswith(('ocr_', 'vision_'))
                            and r != 'ocr_not_configured']
                if 'deadline' in result['reason_codes']:
                    failures.append('deadline')
                output.append({'query_id': row['query_id'], 'image_sha256': digest,
                    'catalog_sha256': index.wines[0].csv_sha256,
                    'fusion_config_sha256': (fingerprint(online.fusion_config)
                        if runtime_profile is not None and bbox is None
                        and seconds <= online.request_timeout
                        and runtime_profile == online.fusion_config else None),
                    'evidence': evidence, 'ocr_text': (observed or {}).get('cleaned_text'),
                    'ocr_status': observed['status'] if observed else 'not_run',
                    'failures': failures, 'latency_ms': round((time.monotonic()-start)*1000),
                    'top_candidates': ranked, 'predicted_slug': ranked[0]['slug'] if ranked else None,
                    'status': result['status'] if online.policy else
                              'ranked_only' if ranked else 'no_evidence',
                    'online_status': result['status'], 'online_slug': result['slug'],
                    'reason_codes': result['reason_codes'], 'roi': result['roi'],
                    'accepted_slug': result['slug'] if online.policy else None})
                continue
            if vision is not None:
                visual_image = image
                if row.get('bbox') is not None:
                    roi = manual_roi(data, row['bbox'], deadline)
                    from PIL import Image
                    visual_image = Image.open(io.BytesIO(roi.crop)).convert('RGB')
                try:
                    evidence.extend(vision.observe(visual_image, deadline))
                except (RuntimeError, TimeoutError) as exc:
                    failures.append('vision_' + str(exc))
                finally:
                    if visual_image is not image:
                        visual_image.close()
            if ocr_mode == 'manual':
                raw = row.get('manual_text')
                if not isinstance(raw, str):
                    raise ValueError('manual_text required for every query in oracle mode')
                ocr_status = 'manual_oracle'
                text = clean_ocr(raw)
            elif ocr_mode == 'live':
                # Explicit crop or manual bbox; never send a full scene as a found label.
                if roi_mode == 'manual' and row.get('bbox') is None:
                    raise ValueError('manual bbox required for live OCR (or opt in to center_80_crop)')
                try:
                    roi = (manual_roi(data, row['bbox'], deadline) if row.get('bbox') is not None
                           else center_crop_80_roi(data, deadline))
                    observed = ocr.recognize(roi, deadline)
                    ocr_status = observed.status
                    if observed.status == 'ok' and observed.raw_text:
                        text = clean_ocr(observed.raw_text)
                    elif observed.error_code:
                        failures.append('ocr_' + observed.error_code)
                except (TimeoutError, ValueError) as exc:
                    failures.append('ocr_' + str(exc))
                    ocr_status = 'error'
            if text and time.monotonic() < deadline:
                evidence.extend(CandidateEvidence(c.slug, 'text', c.score, c.rank,
                                'text_score_0_100', BLEND_VERSION)
                                for c in index.search(text, k=blender.config.top_k))
            elif text:
                failures.append('deadline_before_text_search')
            ranked = blender.rank(evidence, text)
            output.append({'query_id': row['query_id'], 'image_sha256': digest,
                           'catalog_sha256': index.wines[0].csv_sha256,
                           # Oracle/manual bbox observations are diagnostics, not a
                           # calibrated online profile; leave the digest absent.
                           'fusion_config_sha256': (fingerprint(runtime_profile)
                               if runtime_profile and ocr_mode == 'none' and row.get('bbox') is None
                               else None),
                           'evidence': [asdict(e) for e in evidence],
                           'ocr_text': text, 'ocr_status': ocr_status,
                           'failures': failures, 'latency_ms': round((time.monotonic()-start)*1000),
                           'top_candidates': ranked[:20],
                           'predicted_slug': ranked[0]['slug'] if ranked else None,
                           'status': 'ranked_only' if ranked else 'no_evidence'})
        finally:
            image.close()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--images', nargs='+', help='real photos without labels: rankings only')
    source.add_argument('--manifest', help='independent labelled query JSONL (audited before weights)')
    source.add_argument('--candidate-images', help='images.csv; unverified uploads diagnostic ONLY')
    parser.add_argument('--uploads', help='root of renamed Strapi uploads, for --candidate-images')
    parser.add_argument('--sample', type=int, default=40, help='max diagnostic images; default 40')
    parser.add_argument('--seed', default='0', help='fixed deterministic diagnostic sample')
    parser.add_argument('--csv', default='strapi_output0709.csv')
    parser.add_argument('--classes', help='version-matched class NPZ')
    parser.add_argument('--gallery', help='only a manually verified gallery NPZ')
    parser.add_argument('--gallery-manifest', help='verified manifest for leakage audit')
    parser.add_argument('--adapter', help='LoRA checkpoint used to build the index')
    parser.add_argument('--config', help='frozen BlendConfig JSON')
    parser.add_argument('--aliases', default='producer_aliases.txt')
    parser.add_argument('--ocr', choices=('none', 'manual', 'live'), default='none')
    parser.add_argument('--roi-mode', choices=('manual', 'center_80_crop',
                        'full_frame_experimental', 'auto_bbox_experimental'), default='manual')
    parser.add_argument('--allow-live-ocr', action='store_true', help='consent for external OCR ROI transmission')
    parser.add_argument('--allow-remote-code', action='store_true', help='approve local execution of pinned HF code')
    parser.add_argument('--device', default=None)
    parser.add_argument('--seconds', type=float, default=30.)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--output', required=True, help='new output prefix: .evidence.jsonl and .report.json')
    args = parser.parse_args()
    if args.gallery and not args.gallery_manifest:
        parser.error('--gallery requires --gallery-manifest for leakage audit')
    if args.candidate_images and not args.uploads:
        parser.error('--candidate-images requires --uploads')
    if args.ocr == 'manual' and not args.manifest:
        parser.error('manual transcription requires a manifest')
    if args.ocr == 'live':
        provider = os.environ.get('WINE_OCR_PROVIDER', 'none')
        if provider in ('none', 'mock'):
            parser.error('live OCR requires an explicit non-mock WINE_OCR_PROVIDER')
        if provider == 'mistral' and (not args.allow_live_ocr or not os.environ.get('MISTRAL_API_KEY')):
            parser.error('external OCR requires --allow-live-ocr and MISTRAL_API_KEY')
    if (args.classes or args.gallery) and not args.allow_remote_code:
        parser.error('approve license and remote code with --allow-remote-code')
    if not (args.classes or args.gallery or args.ocr != 'none'):
        parser.error('configure at least one candidate source')
    prefix = Path(args.output)
    evidence_path = Path(str(prefix) + '.evidence.jsonl')
    report_path = Path(str(prefix) + '.report.json')
    if any(p.exists() for p in (evidence_path, report_path)) or not prefix.parent.is_dir():
        parser.error('output must be new and its parent directory must exist')

    wines, catalog = load_catalog(args.csv)
    root = Path(args.manifest).resolve().parent if args.manifest else None
    if args.manifest:
        queries = checked_queries(args.manifest, wines, args.gallery_manifest)
        if args.ocr == 'manual' and any(not isinstance(r.get('manual_text'), str) for r in queries):
            parser.error('manual_text required for all manifest rows')
    elif args.candidate_images:
        queries, root = candidate_queries(args.candidate_images, args.uploads, wines,
                                          limit=args.sample, seed=args.seed)
    else:
        queries = unlabelled_queries(args.images)
    if args.gallery_manifest and not args.manifest:
        gallery_hashes = {r.get('image_sha256') for r in jsonl(args.gallery_manifest)}
        if any(r['image_sha256'] in gallery_hashes for r in queries):
            raise ValueError('query/gallery identical image; remove leakage before inference')
    aliases = ProducerAliases.load(args.aliases) if Path(args.aliases).is_file() else None
    index = Index(wines, aliases=aliases)
    config = BlendConfig(**json.loads(Path(args.config).read_text(encoding='utf-8'))) if args.config else BlendConfig()
    blender = Blender(wines, aliases=aliases, config=config)
    vision = load_vision(wines, classes_path=args.classes, gallery_path=args.gallery,
                         adapter_path=args.adapter, allow_remote_code=args.allow_remote_code,
                         device=args.device, isolated=True)
    from .ocr import configured_ocr
    ocr = None
    try:
        ocr = configured_ocr(os.environ) if args.ocr == 'live' else None
        if hasattr(ocr, 'start'):
            ocr.start()
        # Only no-OCR/no-bbox observations match the online full-image path.
        runtime_profile = (Pipeline(index, csv_sha256=catalog['csv_sha256'],
                                    vision=vision, blender=blender,
                                    roi_mode='refuse' if args.roi_mode == 'manual' else args.roi_mode,
                                    request_timeout=args.seconds).fusion_config
                           if args.ocr == 'none' else None)
        evidence = observe_queries(queries, root=root, index=index, blender=blender, vision=vision,
                                   ocr=ocr, ocr_mode=args.ocr, roi_mode=args.roi_mode,
                                   seconds=args.seconds, runtime_profile=runtime_profile)
        encoder_config = vision.encoder.config if vision else None
    finally:
        if hasattr(ocr, 'close'):
            ocr.close()
        if vision:
            vision.close()
    report = {'catalog_sha256': catalog['csv_sha256'], 'catalog_wines': len(wines),
              'class_index_sha256': getattr(vision.classes, 'archive_sha256', None) if vision else None,
              'gallery_index_sha256': vision.gallery.archive_sha256 if vision and vision.gallery else None,
              'encoder': encoder_config,
              'class_prompt': ({k: vision.classes.meta.get(k) for k in
                                ('prompt_variant', 'ensemble', 'reduction', 'source_sha256')}
                               if vision and vision.classes else None),
              'config': asdict(config), 'fusion_version': VERSION,
              'search_version': BLEND_VERSION, 'ocr_mode': args.ocr,
              'data_status': ('independent_gt_requires_human_provenance_review' if args.manifest
                              else 'candidate_upload_links_not_ground_truth' if args.candidate_images
                              else 'unlabelled_real_images_no_accuracy'),
              'results': ([{'query_id': r['query_id'], 'image_sha256': r['image_sha256'],
                           'status': r['status'], 'predicted_slug': r['predicted_slug'],
                           'top5': [c['slug'] for c in r['top_candidates'][:5]],
                           'sources': sorted({e['source'] for e in r['evidence']}),
                           'ocr_status': r['ocr_status'], 'failures': r['failures'],
                           'latency_ms': r['latency_ms']} for r in evidence])}
    if args.manifest:
        report['evaluation'] = evaluate(wines, queries, evidence, config=config, split=args.split,
                                        aliases=aliases)
    elif args.candidate_images:
        pairs = list(zip(queries, evidence))
        report['candidate_link_diagnostic'] = {
            'images_csv_sha256': hashlib.sha256(Path(args.candidate_images).read_bytes()).hexdigest(),
            'seed': args.seed, 'sample_requested': args.sample, 'sampled': len(pairs),
            'agreement_at_1': {'numerator': sum(q['_candidate_slug'] == r['predicted_slug'] for q, r in pairs),
                               'denominator': len(pairs)},
            'agreement_at_5': {'numerator': sum(q['_candidate_slug'] in
                                  [c['slug'] for c in r['top_candidates'][:5]] for q, r in pairs),
                               'denominator': len(pairs)},
            'note': 'Heuristic CSV/upload links and studio images; NOT independent accuracy.',
            'pairs': [{'query_id': q['query_id'], 'candidate_slug': q['_candidate_slug'],
                       'predicted_slug': r['predicted_slug']} for q, r in pairs]}
    # Refuse overwrite; publish only after all inference and evaluation succeed.
    with evidence_path.open('x', encoding='utf-8') as file:
        for row in evidence:
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    with report_path.open('x', encoding='utf-8') as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write('\n')
    print(json.dumps({'evidence': str(evidence_path), 'report': str(report_path),
                      'queries': len(queries), 'data_status': report['data_status']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
