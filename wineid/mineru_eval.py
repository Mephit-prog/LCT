"""Independent per-font CER/WER for *documents*, not wine SKU accuracy.

JSONL: key, gt_path, gt_sha256, font_class, split, gt_source. GT must be
human-verified, from a different source than MinerU outputs. No OCR calls.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .mineru_cloud import normalize_ru
from .ocr_eval import edit_distance


def evaluate(manifest, results_dir, *, split='validation'):
    root = Path(manifest).resolve().parent
    results = Path(results_dir).resolve()
    if split not in ('validation', 'test'):
        raise ValueError('invalid split')
    rows = [json.loads(line) for line in Path(manifest).read_text(encoding='utf-8').splitlines()
            if line.strip()]
    if not rows or len({row.get('key') for row in rows}) != len(rows):
        raise ValueError('empty or duplicate manifest')
    checked = []
    for row in rows:  # audit ALL splits before filtering
        if (not isinstance(row.get('key'), str) or len(row['key']) != 64
                or any(c not in '0123456789abcdef' for c in row['key'])
                or row.get('split') not in ('validation', 'test')
                or not isinstance(row.get('font_class'), str) or not row['font_class']
                or not isinstance(row.get('gt_source'), str) or not row['gt_source']
                or not isinstance(row.get('gt_sha256'), str)):
            raise ValueError('invalid ground truth manifest')
        path = (root / str(row.get('gt_path', ''))).resolve()
        if (not path.is_relative_to(root) or not path.is_file() or path.stat().st_size > 10_000_000
                or hashlib.sha256(path.read_bytes()).hexdigest() != row['gt_sha256']):
            raise ValueError('missing, unsafe or changed ground truth')
        checked.append((row, normalize_ru(path.read_text(encoding='utf-8'))))
    counters = {}
    for row, gt in checked:
        if row['split'] != split:
            continue
        path = results / row['key'] / 'normalized.txt'
        if not path.resolve().is_relative_to(results) or (path.is_file() and path.stat().st_size > 64_000_000):
            raise ValueError('unsafe result path')
        predicted = normalize_ru(path.read_text(encoding='utf-8')) if path.is_file() else ''
        if len(gt) > 20000 or len(predicted) > 20000:
            raise ValueError('CER/WER require page-level transcripts (max 20000 characters)')
        group = counters.setdefault(row['font_class'], dict(count=0, missing=0, chars=0,
                                                             char_errors=0, words=0, word_errors=0))
        group['count'] += 1
        group['missing'] += not path.is_file()
        group['chars'] += len(gt)
        group['char_errors'] += edit_distance(gt, predicted)
        group['words'] += len(gt.split())
        group['word_errors'] += edit_distance(gt.split(), predicted.split())
    if not counters:
        raise ValueError('empty split')
    return {'split': split, 'documents': sum(x['count'] for x in counters.values()),
            'by_font_class': {name: {**c, 'cer': c['char_errors'] / c['chars'] if c['chars'] else None,
                                      'wer': c['word_errors'] / c['words'] if c['words'] else None}
                              for name, c in sorted(counters.items())},
            'note': 'Requires independent literal transcripts; no automated fallback threshold.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('--results-dir', required=True)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    args = parser.parse_args()
    print(json.dumps(evaluate(args.manifest, args.results_dir, split=args.split),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
