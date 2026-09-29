"""Paired manual/automatic ROI ablation on a fixed independent validation split.

Do not use eval/ queries or mock transcriptions as evidence of OCR improvements.
"""
import argparse
import json
from .data import jsonl
from .report import report


def compare(manual, experimental):
    def keyed(rows):
        mapping = {}
        for row in rows:
            qid = row['query_id']
            if qid in mapping:
                raise ValueError(f'duplicate query_id: {qid}')
            mapping[qid] = row
        return mapping
    a, b = keyed(manual), keyed(experimental)
    if not a or a.keys() != b.keys():
        raise ValueError('different or empty query sets')
    for qid in a:
        left, right = a[qid], b[qid]
        if left.get('image_sha256') != right.get('image_sha256'):
            raise ValueError(f'different image for {qid}')
        if any(left.get(field) != right.get(field)
               for field in ('split', 'gt_slug', 'gt_status', 'source_group')):
            raise ValueError(f'different GT/split/source group for {qid}')
        for field in ('csv_sha256', 'search_backend', 'policy'):
            if left.get('versions', {}).get(field) != right.get('versions', {}).get(field):
                raise ValueError(f'different {field} for {qid}')
    pairs = [(a[key], b[key]) for key in a]
    def stats(rows):
        return {'roi_found': sum(r.get('roi') is not None for r in rows),
                'roi_total': len(rows), 'metrics': report(rows)}
    mocked = any(r.get('ocr') and r['ocr'].get('model') == 'mock'
                 for left, right in pairs for r in (left, right))
    return {'manual': stats(manual), 'experimental': stats(experimental),
            'ocr_oracle_included': mocked,
            'warning': ('mock text is not OCR evidence; compare localization only'
                        if mocked else None),
            'manual_found_experimental_missed': [left['query_id'] for left, right in pairs
                                                  if left.get('roi') and not right.get('roi')],
            'experimental_changed_top1': [left['query_id'] for left, right in pairs
                                           if [c['slug'] for c in left.get('candidates', [])[:1]] !=
                                              [c['slug'] for c in right.get('candidates', [])[:1]]]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('manual_jsonl')
    p.add_argument('experimental_jsonl')
    p.add_argument('--split', choices=['train', 'validation', 'test'], default='validation')
    args = p.parse_args()
    manual, experimental = (jsonl(path) for path in (args.manual_jsonl, args.experimental_jsonl))
    print(json.dumps(compare([r for r in manual if r.get('split') == args.split],
                             [r for r in experimental if r.get('split') == args.split]),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
