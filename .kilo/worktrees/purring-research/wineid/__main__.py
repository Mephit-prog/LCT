"""Catalog and offline manual-ROI tools. Never read images from eval/ for tuning."""
import argparse
import hashlib
import json
from pathlib import Path
from .catalog import export_catalog, load_catalog
from .ocr import MistralOCR
from .pipeline import Pipeline
from .search import Index
from .text_policy import load_policy
from .canonical import ProducerAliases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='strapi_output0709.csv')
    default_aliases = 'producer_aliases.txt' if Path('producer_aliases.txt').is_file() else None
    parser.add_argument('--aliases', default=default_aliases,
                        help='producer alias file: "Canonical, variant1, variant2"')
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('catalog')
    c.add_argument('output')
    a = sub.add_parser('audit')
    a.add_argument('gallery', help='manually verified gallery JSONL')
    a.add_argument('queries', help='independently annotated query JSONL')
    s = sub.add_parser('search')
    s.add_argument('text')
    r = sub.add_parser('offline')
    r.add_argument('manifest', help='JSONL: query_id, image_path, bbox, split, gt_slug or null, manual_text optional')
    r.add_argument('output')
    r.add_argument('--policy', help='validation-derived policy JSON')
    r.add_argument('--roi-mode', default='manual',
                   choices=['manual', 'auto_bbox_experimental', 'full_frame_experimental'],
                   help='explicit ablation; defaults to annotated manual bbox')
    r.add_argument('--allow-live-ocr', action='store_true', help='requires MISTRAL_API_KEY and data permission')
    args = parser.parse_args()
    if args.command == 'catalog':
        print(json.dumps(export_catalog(args.csv, args.output), ensure_ascii=False))
        return
    wines, report = load_catalog(args.csv)
    aliases = ProducerAliases.load(args.aliases) if args.aliases else None
    index = Index(wines, aliases=aliases)
    if args.command == 'search':
        print(json.dumps({'backend': index.backend, 'candidates': [c.__dict__ for c in index.search(args.text)]}, ensure_ascii=False))
        return
    if args.command == 'audit':
        from .data import audit
        print(json.dumps(audit(wines, args.gallery, args.queries), ensure_ascii=False, indent=2))
        return
    import os
    policy = load_policy(args.policy) if args.policy else None
    key = os.environ.get('MISTRAL_API_KEY')
    if args.allow_live_ocr and not key:
        parser.error('MISTRAL_API_KEY required')
    pipeline = Pipeline(index, MistralOCR(key) if args.allow_live_ocr else None, policy,
                        report['csv_sha256'], 'refuse' if args.roi_mode == 'manual' else args.roi_mode)
    manifest = Path(args.manifest).resolve()
    out = Path(args.output).resolve()
    if manifest == out or out.exists():
        parser.error('output must be new and different from manifest')
    seen = set()
    with manifest.open(encoding='utf-8') as source, out.open('x', encoding='utf-8') as dest:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row['query_id']
            if qid in seen:
                raise ValueError('duplicate query_id')
            seen.add(qid)
            photo = (manifest.parent / row['image_path']).resolve()
            if not photo.is_relative_to(manifest.parent):
                raise ValueError('unsafe image path')
            if args.roi_mode == 'manual' and (not isinstance(row.get('bbox'), list) or len(row['bbox']) != 4):
                raise ValueError('manual bbox required; no silent automatic ROI')
            if 'manual_text' not in row and not args.allow_live_ocr:
                raise ValueError('manual_text or --allow-live-ocr required')
            data = photo.read_bytes()
            bbox = tuple(row['bbox']) if args.roi_mode == 'manual' else None
            result = pipeline.predict(data, bbox=bbox, manual_text=row.get('manual_text'))
            dest.write(json.dumps({'query_id': qid, 'image_sha256': hashlib.sha256(data).hexdigest(),
                'split': row.get('split'), 'gt_slug': row.get('gt_slug'),
                'gt_status': row.get('gt_status'), 'source_group': row.get('source_group'), **result},
                ensure_ascii=False, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
