"""Paired diagnostic comparison of two *unverified candidate-link* packshot runs.

This is NOT accuracy or independent validation. Never use it to calibrate an
acceptance policy or select an untouched test checkpoint.
"""
import argparse
import json
from pathlib import Path


def compare_candidate_reports(left, right):
    if any(report.get('data_status') != 'candidate_upload_links_not_ground_truth'
           for report in (left, right)):
        raise ValueError('requires candidate-link diagnostic reports')
    keys = ('catalog_sha256', 'catalog_wines', 'config', 'fusion_version',
            'search_version', 'ocr_mode', 'class_prompt')
    if any(left.get(key) != right.get(key) for key in keys):
        raise ValueError('incompatible catalog/prompt/fusion configuration')
    def base_encoder(report):
        encoder = report.get('encoder') or {}
        return {k: v for k, v in encoder.items() if k != 'adapter_sha256'}
    if base_encoder(left) != base_encoder(right):
        raise ValueError('incompatible base encoder')
    a, b = left['candidate_link_diagnostic'], right['candidate_link_diagnostic']
    if any(a.get(key) != b.get(key) for key in
           ('images_csv_sha256', 'seed', 'sample_requested', 'sampled')):
        raise ValueError('candidate source or sampling mismatch')

    def aligned(report):
        pairs = report['candidate_link_diagnostic']['pairs']
        outputs = report['results']
        if len(pairs) != len(outputs) or len(pairs) != a['sampled']:
            raise ValueError('incomplete paired reports')
        if len({r['query_id'] for r in pairs}) != len(pairs):
            raise ValueError('duplicate query IDs')
        if any(p['query_id'] != r['query_id'] or p['predicted_slug'] != r['predicted_slug']
               for p, r in zip(pairs, outputs)):
            raise ValueError('candidate/result mismatch')
        for k in (1, 5):
            actual = sum((p['candidate_slug'] == r['predicted_slug'] if k == 1 else
                          p['candidate_slug'] in r['top5'][:5])
                         for p, r in zip(pairs, outputs))
            if report['candidate_link_diagnostic'][f'agreement_at_{k}'] != {
                    'numerator': actual, 'denominator': len(pairs)}:
                raise ValueError('candidate agreement/result mismatch')
        return {p['query_id']: (p['candidate_slug'], r['image_sha256'], r['predicted_slug'],
                                r['top5']) for p, r in zip(pairs, outputs)}
    before, after = aligned(left), aligned(right)
    if before.keys() != after.keys() or any(x[:2] != after[q][:2] for q, x in before.items()):
        raise ValueError('not the same candidate links and image bytes')
    helped = sorted(q for q, x in before.items()
                    if x[2] != x[0] and after[q][2] == x[0])
    harmed = sorted(q for q, x in before.items()
                    if x[2] == x[0] and after[q][2] != x[0])
    return {'data_status': 'candidate_upload_links_not_ground_truth',
            'note': 'Agreement with unverified CSV/upload mappings on studio photos; NOT accuracy.',
            'paired_images': len(before),
            'left_agreement_at_1': a['agreement_at_1'],
            'right_agreement_at_1': b['agreement_at_1'],
            'left_agreement_at_5': a['agreement_at_5'],
            'right_agreement_at_5': b['agreement_at_5'],
            'changed_top1': sum(x[2] != after[q][2] for q, x in before.items()),
            'helped_candidate_link': helped, 'harmed_candidate_link': harmed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left')
    parser.add_argument('right')
    args = parser.parse_args()
    result = compare_candidate_reports(json.loads(Path(args.left).read_text(encoding='utf-8')),
                                       json.loads(Path(args.right).read_text(encoding='utf-8')))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
