"""Independent manual-ROI OCR -> text retrieval diagnostic (not a SKU accuracy claim).

No model is loaded before the complete manifest, hashes and split are audited.
Raw text is only written to the explicit private --output file, never to stdout.
"""
import argparse
import hashlib
import json
import os
import time
import unicodedata
from pathlib import Path

from .catalog import load_catalog
from .data import jsonl, query_manifest_errors
from .roi import MAX_BYTES, InvalidImage, manual_roi
from .search import Index


def words(text):
    text = unicodedata.normalize('NFKC', text).lower().replace('ё', 'е')
    return ' '.join(text.split())


def edit_distance(a, b):
    # Same Levenshtein definition for characters and token sequences.
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j-1] + (x != y)))
        previous = current
    return previous[-1]


def prepare_manifest(manifest, index):
    manifest = Path(manifest).resolve()
    rows = jsonl(manifest)
    errors, _ = query_manifest_errors(rows, catalog_ids={w.slug for w in index.wines})
    sources = {}
    for row in rows:
        qid = row.get('query_id')
        transcription = row.get('transcription')
        if not isinstance(transcription, str) or not transcription.strip():
            errors.append(f'missing literal transcription: {qid}')
        bbox = row.get('bbox')
        if not (isinstance(bbox, list) and len(bbox) == 4 and
                all(type(x) is int for x in bbox)):
            errors.append(f'invalid oriented bbox: {qid}')
        path = (manifest.parent / str(row.get('image_path', ''))).resolve()
        if not path.is_relative_to(manifest.parent) or not path.is_file():
            errors.append(f'missing/unsafe image: {qid}')
            continue
        if path.stat().st_size > MAX_BYTES:
            errors.append(f'oversized image: {qid}')
            continue
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row.get('image_sha256'):
            errors.append(f'image hash mismatch: {qid}')
        sources[qid] = data
    if errors:
        raise ValueError('; '.join(sorted(set(errors))))
    return rows, sources


def evaluate(manifest, index, ocr, *, split='validation', seconds=8.5, output=None, prepared=None):
    rows, sources = prepared if prepared is not None else prepare_manifest(manifest, index)
    selected = [row for row in rows if row['split'] == split]
    if not selected:
        raise ValueError('empty split')
    total_chars = total_words = char_errors = word_errors = 0
    successes = recall = oracle_recall = 0
    statuses = {}
    durations = []
    details = []
    for row in selected:
        started = time.monotonic()
        try:
            roi = manual_roi(sources[row['query_id']], tuple(row['bbox']), started + seconds)
            result = ocr.recognize(roi, started + seconds)
            status, raw, code, model = result.status, result.raw_text, result.error_code, result.model
        except TimeoutError:
            status, raw, code, model = 'timeout', None, 'deadline', getattr(ocr, 'model', None)
        except InvalidImage:
            status, raw, code, model = 'unreadable', None, 'invalid_roi', getattr(ocr, 'model', None)
        if time.monotonic() >= started + seconds:
            status, raw, code = 'timeout', None, 'deadline'
        elapsed = round((time.monotonic() - started) * 1000)
        durations.append(elapsed)
        statuses[status] = statuses.get(status, 0) + 1
        gt = words(row['transcription'])
        pred = words(raw or '') if status == 'ok' else ''
        total_chars += len(gt)
        total_words += len(gt.split())
        char_errors += edit_distance(gt, pred)
        word_errors += edit_distance(gt.split(), pred.split())
        successes += status == 'ok'
        slug = row.get('gt_slug')
        if slug:
            oracle_recall += slug in [c.slug for c in index.search(row['transcription'], k=5)]
            if pred and time.monotonic() < started + seconds:
                recall += slug in [c.slug for c in index.search(raw, k=5)]
        details.append({'query_id': row['query_id'], 'split': split, 'source_group': row['source_group'],
                        'gt_slug': slug, 'status': status, 'error_code': code,
                        'raw_text': raw, 'transcription': row['transcription'],
                        'model': model, 'latency_ms': elapsed})
    if output is not None:
        # New file only; 0600 protects permitted raw text from accidental reuse.
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            for row in details:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    known = sum(bool(r.get('gt_slug')) for r in selected)
    return {'split': split, 'model': getattr(ocr, 'model', None),
            'count': len(selected), 'known': known,
            'independent_source_groups': len({r['source_group'] for r in selected}),
            'statuses': statuses, 'usable_text': {'numerator': successes, 'denominator': len(selected)},
            'cer': {'errors': char_errors, 'characters': total_chars,
                    'value': char_errors / total_chars if total_chars else None},
            'wer': {'errors': word_errors, 'words': total_words,
                    'value': word_errors / total_words if total_words else None},
            'recall_at_5': {'ocr': recall, 'manual_text_oracle': oracle_recall, 'denominator': known},
            'latency_ms': {'p50': sorted(durations)[(len(durations)-1)//2],
                           'p95': sorted(durations)[int((len(durations)-1)*.95)]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', help='private JSONL with independent GT, bbox, transcription')
    parser.add_argument('--csv', default='strapi_output0709.csv')
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--output', help='private JSONL raw transcripts (0600, must not exist)')
    args = parser.parse_args()
    wines, _ = load_catalog(args.csv)
    from .ocr import configured_ocr
    ocr = configured_ocr(os.environ)
    if ocr is None or os.environ.get('WINE_OCR_PROVIDER') in ('mock', 'mistral'):
        raise ValueError('evaluation requires an explicitly selected local OCR engine')
    index = Index(wines)
    prepared = prepare_manifest(args.manifest, index)  # validate *all* splits before inference
    try:
        ocr.start()
        print(json.dumps(evaluate(args.manifest, index, ocr, split=args.split,
                                  output=args.output, prepared=prepared), ensure_ascii=False, indent=2))
    finally:
        if hasattr(ocr, 'close'):
            ocr.close()


if __name__ == '__main__':
    main()
