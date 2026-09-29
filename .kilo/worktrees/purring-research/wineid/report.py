"""Compute transparent denominators; rejects count against known exact-slug accuracy."""
import argparse
import json
from collections import Counter


def report(rows, k_values=(1, 5, 20)):
    known = [r for r in rows if r.get('gt_slug')]
    unknown = [r for r in rows if r.get('gt_slug') is None and r.get('gt_status') == 'unknown']
    labeled = [r for r in rows if r.get('gt_slug') or r.get('gt_status') == 'unknown']
    accepted = [r for r in labeled if r.get('status') == 'accepted']
    correct = sum(r.get('status') == 'accepted' and r.get('slug') == r['gt_slug'] for r in known)
    def fraction(n, d):
        return {'numerator': n, 'denominator': d, 'value': n/d if d else None}
    latency = [r.get('timings_ms', {}).get('total') for r in rows]
    latency = sorted(x for x in latency if isinstance(x, (int, float)))
    return {
        'total': len(rows), 'known': len(known), 'unknown': len(unknown),
        'exact_slug_accuracy': fraction(correct, len(known)),
        'accepted_precision': fraction(sum(r.get('gt_slug') is not None and r.get('slug') == r['gt_slug']
                                           for r in accepted), len(accepted)),
        'coverage': fraction(sum(r.get('status') == 'accepted' for r in rows), len(rows)),
        'unknown_false_accept': fraction(sum(r.get('status') == 'accepted' for r in unknown), len(unknown)),
        'text_recall': {str(k): fraction(sum(r['gt_slug'] in [c['slug'] for c in r.get('candidates', [])[:k]] for r in known), len(known)) for k in k_values},
        'statuses': dict(Counter(r.get('status') for r in rows)),
        'latency_ms': {f'p{p}': latency[min(len(latency)-1, int((len(latency)-1)*p/100))] if latency else None for p in (50, 95, 99)},
        'independent_source_groups': len({r['source_group'] for r in rows if r.get('source_group')}),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('jsonl')
    p.add_argument('--split', choices=['train', 'validation', 'test'])
    args = p.parse_args()
    rows = [json.loads(line) for line in open(args.jsonl, encoding='utf-8')]
    if args.split:
        rows = [r for r in rows if r.get('split') == args.split]
    print(json.dumps(report(rows), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
