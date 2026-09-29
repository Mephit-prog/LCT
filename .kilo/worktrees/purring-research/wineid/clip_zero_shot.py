"""Jina CLIP text-to-image retrieval and closed-set zero-shot classification, no OCR.

Image indexes need no wine labels. Class indexes contain only Russian descriptions,
not reference photos. Cosines/ranks are not calibrated probabilities or acceptance.
"""
import argparse
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np

from .catalog import load_catalog
from .data import jsonl, retrieval_manifest_errors
from .jina_clip import JinaCLIPEmbedder, REVISION, unit_vectors, IncompatibleEmbeddings
from .canonical import canonical_text
from .roi import decode, MAX_BYTES

SCHEMA = 'jina-zero-shot-v1'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
# The same templates are applied to EVERY class. No inferred label colours,
# country of origin, translation, OCR text, filename or slug enters the encoder.
TEMPLATES = {
    'title': ('{name}',),
    'name': ('Фотография бутылки вина «{name}»; видна лицевая этикетка.',
             'Фотография лицевой этикетки вина «{name}».'),
    'name-producer': (
        'Фотография бутылки вина «{name}» производителя {producer}; видна лицевая этикетка.',
        'Фотография лицевой этикетки вина «{name}» производителя {producer}.'),
    'name-producer-visual': (
        'Фотография бутылки вина «{name}» производителя {producer}: {visual}.',
        'Фотография лицевой этикетки вина «{name}» производителя {producer}: {visual}.'),
    # tz-clip-text-ranking.md §3: one canonical template; built from
    # name_ru/producer/color/vintage via wineid.canonical.canonical_text.
    'canonical': ('{canonical}',),
}


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def read_image(path):
    """Bounded local decode: content sniffing, EXIF orientation and white alpha background."""
    with Path(path).open('rb') as source:
        data = source.read(MAX_BYTES + 1)
    image, _ = decode(data, time.monotonic() + 30)
    return image, hashlib.sha256(data).hexdigest()


def class_prompts(rows, variant='name', ensemble=False):
    if variant not in TEMPLATES:
        raise ValueError('unknown prompt variant')
    if not rows:
        raise ValueError('empty classes')
    templates = TEMPLATES[variant] if ensemble else TEMPLATES[variant][:1]
    required = ['name']
    if 'producer' in variant:
        required.append('producer')
    if 'visual' in variant:
        required.append('visual')
    items, ids = [], set()
    for row in rows:
        cid = row.get('class_id')
        if not isinstance(cid, str) or not cid.strip() or cid in ids:
            raise ValueError('missing or duplicate class_id')
        if variant == 'canonical':
            values = {'name_ru': (row.get('name_ru') or '').strip(),
                      'producer': (row.get('producer') or '').strip(),
                      'color': (row.get('color') or '').strip(),
                      'vintage': (row.get('vintage') or '').strip()}
            if not values['name_ru'] or not values['producer']:
                raise ValueError(f'{cid}: canonical requires name_ru and producer for ALL classes')
            prompt = canonical_text(name_ru=values['name_ru'], producer=values['producer'],
                                    color=values['color'] or None,
                                    vintage=values['vintage'] or None)
            items.append({'id': cid, **values, 'prompts': [prompt]})
            continue
        if any(not isinstance(row.get(key), str) or not row[key].strip() for key in required):
            raise ValueError(f'{cid}: every class needs {required}; choose a simpler template for ALL classes')
        ids.add(cid)
        values = {key: row[key].strip() for key in required}
        items.append({'id': cid, **values,
                      'prompts': [template.format(**values) for template in templates]})
    return items


def prompt_collisions(items):
    """Classes with identical input prompts cannot be distinguished by this encoder."""
    groups = {}
    for item in items:
        groups.setdefault(tuple(item['prompts']), []).append(item['id'])
    return {cid: group for group in groups.values() if len(group) > 1 for cid in group}


class EmbeddingIndex:
    def __init__(self, vectors, metadata):
        if not isinstance(metadata, dict):
            raise IncompatibleEmbeddings('invalid zero-shot index metadata')
        # The validated state must not be mutable through a caller's metadata object.
        self.meta = json.loads(json.dumps(metadata, ensure_ascii=False, allow_nan=False))
        metadata = self.meta
        self.kind = metadata.get('kind')
        self.items = metadata.get('items')
        config = metadata.get('encoder')
        if (metadata.get('schema') != SCHEMA or self.kind not in ('images', 'classes')
                or not isinstance(config, dict) or type(config.get('dimensions')) is not int
                or config['dimensions'] < 1 or not isinstance(self.items, list) or not self.items
                or any(not isinstance(item, dict) or not isinstance(item.get('id'), str)
                       or not item['id'].strip() for item in self.items)
                or len({item['id'] for item in self.items}) != len(self.items)):
            raise IncompatibleEmbeddings('invalid zero-shot index metadata')
        if self.kind == 'images':
            if (not isinstance(metadata.get('image_root'), str) or not metadata['image_root']
                    or metadata.get('image_mode') != 'exif-rgb-full-frame-v1'
                    or any(not isinstance(item.get('image_sha256'), str)
                           or not re.fullmatch(r'[0-9a-f]{64}', item['image_sha256'])
                           or Path(item['id']).is_absolute() or '..' in Path(item['id']).parts
                           or item['id'] in ('.', '..') or '\\' in item['id']
                           for item in self.items)):
                raise IncompatibleEmbeddings('invalid image index metadata')
        else:
            if (metadata.get('prompt_variant') not in TEMPLATES
                    or type(metadata.get('ensemble')) is not bool
                    or any(not isinstance(item.get('prompts'), list)
                           or len(item['prompts']) != (len(TEMPLATES[metadata['prompt_variant']])
                                                      if metadata['ensemble'] else 1)
                           or any(not isinstance(p, str) or not p.strip() for p in item['prompts'])
                           for item in self.items)
                    or (metadata.get('prompt_collisions') is not None
                        and metadata['prompt_collisions'] != prompt_collisions(self.items))):
                raise IncompatibleEmbeddings('invalid class index metadata')
        vectors = np.asarray(vectors, dtype='float32')
        self.vectors = unit_vectors(vectors, len(self.items), config['dimensions'])
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
            raise IncompatibleEmbeddings('unnormalized index')

    def check_encoder(self, encoder):
        if self.meta['encoder'] != encoder.config:
            raise IncompatibleEmbeddings('index/encoder mismatch; rebuild index')

    def save(self, path):
        out = Path(path)
        if out.exists():
            raise FileExistsError(out)
        with tempfile.NamedTemporaryFile(dir=out.parent, suffix='.npz', delete=False) as file:
            temporary = Path(file.name)
        try:
            np.savez_compressed(temporary, embeddings=self.vectors,
                                metadata=np.array(json.dumps(self.meta, ensure_ascii=False, allow_nan=False)))
            try:
                os.link(temporary, out)  # exclusive atomic publish, no overwrite
            except OSError:
                # Filesystems without hardlink support (FAT/exFAT, some network
                # mounts): rename is still atomic within the directory (P2-14).
                if out.exists():
                    raise FileExistsError(out)
                os.rename(temporary, out)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        try:
            if Path(path).stat().st_size > 2 * 1024**3:
                raise IncompatibleEmbeddings('index exceeds archive size limit')
            with np.load(path, allow_pickle=False) as archive:
                return cls(archive['embeddings'], json.loads(str(archive['metadata'].item())))
        except (KeyError, TypeError, AttributeError, OSError, ValueError) as exc:
            raise IncompatibleEmbeddings(f'invalid zero-shot index: {exc}') from exc

    def audit_images(self):
        """Explicit snapshot audit; never silently claim that a stale index is current."""
        if self.kind != 'images':
            raise ValueError('requires an image index')
        root = Path(self.meta['image_root']).resolve()
        problems = []
        for item in self.items:
            path = root / item['id']
            if not path.resolve().is_relative_to(root) or not path.is_file():
                problems.append({'id': item['id'], 'reason': 'missing_or_unsafe'})
                continue
            with path.open('rb') as source:
                data = source.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES or hashlib.sha256(data).hexdigest() != item['image_sha256']:
                problems.append({'id': item['id'], 'reason': 'changed_or_oversized'})
        return {'checked': len(self.items), 'problems': problems, 'snapshot_valid': not problems}

    def rank(self, vectors, top_k=5):
        if type(top_k) is not int or top_k < 1:
            raise ValueError('top_k must be positive')
        vectors = np.asarray(vectors, dtype='float32')
        if vectors.ndim != 2:
            raise IncompatibleEmbeddings('expected a matrix of query embeddings')
        vectors = unit_vectors(vectors, len(vectors), self.vectors.shape[1])
        results = []
        # One score vector at a time: don't allocate queries x entire corpus.
        for vector in vectors:
            scores = np.clip(self.vectors @ vector, -1, 1)
            order = np.argsort(-scores, kind='stable')[:top_k]
            results.append([{**self.items[i], 'rank': rank, 'cosine': float(scores[i])}
                            for rank, i in enumerate(order, 1)])
        return results

    def search(self, query, encoder, top_k=5):
        if self.kind != 'images':
            raise ValueError('text-to-image search requires an image index')
        if not isinstance(query, str) or not query.strip():
            raise ValueError('nonempty query required')
        self.check_encoder(encoder)
        return self.rank(encoder.encode_text([query]), top_k)[0]

    def classify(self, images, encoder, top_k=5):
        if self.kind != 'classes':
            raise ValueError('zero-shot classification requires a class index')
        self.check_encoder(encoder)
        return self.rank(encoder.encode_images(images), top_k)


def build_images(directory, encoder, batch_size=8):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be positive')
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError('image directory does not exist')
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError('no JPEG/PNG/WEBP images found')
    items, chunks = [], []
    for start in range(0, len(paths), batch_size):
        images = []
        try:
            for path in paths[start:start + batch_size]:
                if not path.resolve().is_relative_to(root):
                    raise ValueError(f'image symlink outside directory: {path}')
                image, digest = read_image(path)
                images.append(image)
                items.append({'id': path.relative_to(root).as_posix(),
                              'image_sha256': digest})
            chunks.append(unit_vectors(encoder.encode_images(images), len(images),
                                       encoder.config['dimensions']))
        finally:
            for image in images:
                image.close()
    return EmbeddingIndex(np.concatenate(chunks), {
        'schema': SCHEMA, 'kind': 'images', 'encoder': encoder.config,
        'image_mode': 'exif-rgb-full-frame-v1', 'image_root': str(root), 'items': items,
    })


def build_classes(rows, encoder, *, variant='name', ensemble=False, source_sha256=None,
                  reduction='mean'):
    """Build a class index. ``reduction`` over per-class prompts: 'mean' (old
    default) or 'max' (tz-clip-text-ranking.md §3.1 allows a fixed template set
    with max-similarity; start with a single template and add on val evidence).
    """
    if reduction not in ('mean', 'max'):
        raise ValueError("reduction must be 'mean' or 'max'")
    items = class_prompts(rows, variant, ensemble)
    count = len(items[0]['prompts'])
    texts = [prompt for item in items for prompt in item['prompts']]
    vectors = unit_vectors(encoder.encode_text(texts), len(texts), encoder.config['dimensions'])
    per_class = vectors.reshape(len(items), count, -1)
    reduced = per_class.mean(axis=1) if reduction == 'mean' else per_class.max(axis=1)
    # Normalize each prompt, reduce within a class, then normalize again.
    vectors = unit_vectors(reduced, len(items))
    return EmbeddingIndex(vectors, {
        'schema': SCHEMA, 'kind': 'classes', 'encoder': encoder.config,
        'prompt_variant': variant, 'ensemble': ensemble, 'reduction': reduction,
        'source_sha256': source_sha256,
        'prompt_collisions': prompt_collisions(items), 'items': items,
    })


def retrieval_report(index, queries, encoder):
    """Hit-style Recall@1/5: at least one independently labelled relevant image in Top-K.

    Query JSONL: query_id, text, relevant_image_ids, optional hard_negative (bool).
    This does not infer GT from filenames or calibrate open-set rejection.
    """
    if index.kind != 'images':
        raise ValueError('retrieval evaluation requires an image index')
    index.check_encoder(encoder)
    available = {item['id'] for item in index.items}
    if not queries:
        raise ValueError('empty evaluation queries')
    errors = retrieval_manifest_errors(queries, available)
    if errors:
        raise ValueError(errors[0])
    results = []
    batch_size = getattr(encoder, 'batch_size', 8)
    for start in range(0, len(queries), batch_size):
        batch = queries[start:start + batch_size]
        rankings = index.rank(encoder.encode_text([row['text'] for row in batch]), top_k=5)
        for row, ranking in zip(batch, rankings):
            relevant = set(row['relevant_image_ids'])
            results.append({'query_id': row['query_id'], 'hard_negative': row.get('hard_negative', False),
                            'candidates': ranking,
                            **{f'hit_at_{k}': any(item['id'] in relevant for item in ranking[:k])
                               for k in (1, 5)}})
    def summary(rows):
        return {f'recall_at_{k}': {'numerator': sum(r[f'hit_at_{k}'] for r in rows),
                                 'denominator': len(rows),
                                 'value': sum(r[f'hit_at_{k}'] for r in rows)/len(rows) if rows else None}
                for k in (1, 5)}
    return {'all': summary(results),
            'hard_negatives': summary([r for r in results if r['hard_negative']]),
            'queries': results, 'definition': 'at least one relevant image in Top-K'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build-images', help='index local photos; no labels or OCR required')
    b.add_argument('directory')
    b.add_argument('output')
    c = sub.add_parser('build-classes', help='encode Russian class prompts; no reference photos')
    c.add_argument('output')
    source = c.add_mutually_exclusive_group()
    source.add_argument('--csv', default=None)
    source.add_argument('--classes', help='JSONL: class_id, name, optional producer, visual')
    c.add_argument('--prompt', choices=TEMPLATES, default='name')
    c.add_argument('--ensemble', action='store_true')
    c.add_argument('--reduction', choices=('mean', 'max'), default='mean',
                   help='per-class prompt reduction for ensemble indexes')
    s = sub.add_parser('search', help='Russian text -> ranked photos')
    s.add_argument('index')
    s.add_argument('query')
    p = sub.add_parser('predict', help='photos -> ranked classes (closed set)')
    p.add_argument('index')
    p.add_argument('images', nargs='+')
    e = sub.add_parser('evaluate', help='Recall@1/5 on independently labelled text queries')
    e.add_argument('index')
    e.add_argument('queries', help='JSONL: query_id, text, relevant_image_ids, hard_negative optional')
    a = sub.add_parser('audit-images', help='check indexed photo paths and SHA256; no model needed')
    a.add_argument('index')
    ce = sub.add_parser('evaluate-classes', help='audited photo -> SKU ranking, no acceptance policy')
    ce.add_argument('index')
    ce.add_argument('queries', help='JSONL: query_id, image_path, image_sha256, source_group, split, gt_status, gt_slug, gt_source')
    ce.add_argument('--split', choices=('validation', 'test'), default='validation')
    cc = sub.add_parser('compare-classes', help='paired comparison of two evaluate-classes JSON reports; no model')
    cc.add_argument('left')
    cc.add_argument('right')
    for command in (b, c, s, p, e, ce):
        command.add_argument('--revision', default=REVISION, help='full HF weights commit SHA')
        command.add_argument('--device', default=None, help='cpu, cuda, mps; default cuda if available, else cpu')
        command.add_argument('--batch-size', type=positive_int, default=8)
        command.add_argument('--allow-remote-code', action='store_true',
                             help='approve HF code execution/downloads after license/code review')
    for command in (s, p):
        command.add_argument('--top-k', type=positive_int, default=5)
    args = parser.parse_args()
    if args.command == 'audit-images':
        index = EmbeddingIndex.load(args.index)
        if index.kind != 'images':
            parser.error('audit-images requires an image index')
        print(json.dumps(index.audit_images(), ensure_ascii=False, indent=2))
        return
    if args.command == 'compare-classes':
        from .clip_class_eval import compare_class_reports
        left = json.loads(Path(args.left).read_text(encoding='utf-8'))
        right = json.loads(Path(args.right).read_text(encoding='utf-8'))
        print(json.dumps(compare_class_reports(left, right), ensure_ascii=False, indent=2, allow_nan=False))
        return
    index, rows, digest = None, None, None
    if args.command == 'evaluate-classes':
        from .clip_class_eval import audit_class_queries
        # Do not load remote code or weights if any split has missing GT, leakage or stale bytes.
        index = EmbeddingIndex.load(args.index)
        audit_class_queries(index, args.queries)
    if not args.allow_remote_code:
        parser.error('review CC BY-NC 4.0 and HF remote code, then pass --allow-remote-code')
    if args.command.startswith('build-'):
        output = Path(args.output)
        if output.exists() or not output.parent.is_dir():
            parser.error('output must be new and its parent directory must exist')
        if args.command == 'build-classes':
            if args.classes:
                rows = jsonl(args.classes)
                digest = hashlib.sha256(Path(args.classes).read_bytes()).hexdigest()
            else:
                from .catalog import structured_fields
                wines, stats = load_catalog(args.csv or 'strapi_output0709.csv')
                if args.prompt == 'canonical':
                    rows = [{'class_id': w.slug,
                             'name_ru': (structured_fields(w.raw or {})['name_ru'] or w.name),
                             'producer': (structured_fields(w.raw or {})['producer'] or w.producer),
                             'color': structured_fields(w.raw or {})['color'],
                             'vintage': structured_fields(w.raw or {})['vintage']}
                            for w in wines]
                else:
                    rows = [{'class_id': w.slug, 'name': w.name, 'producer': w.producer}
                            for w in wines]
                digest = stats['csv_sha256']
            class_prompts(rows, args.prompt, args.ensemble)  # fail before loading weights
    else:
        if index is None:
            index = EmbeddingIndex.load(args.index)
        expected = 'classes' if args.command in ('predict', 'evaluate-classes') else 'images'
        if index.kind != expected:
            parser.error(f'{args.command} requires a {expected} index')
    if index is not None and index.meta['encoder'].get('revision') != args.revision:
        parser.error('index/encoder revision mismatch; rebuild index')
    encoder = JinaCLIPEmbedder(revision=args.revision, device=args.device,
                               batch_size=args.batch_size, allow_remote_code=True)
    if index is not None:
        index.check_encoder(encoder)
    if args.command == 'build-images':
        index = build_images(args.directory, encoder, args.batch_size)
    elif args.command == 'build-classes':
        index = build_classes(rows, encoder, variant=args.prompt, ensemble=args.ensemble,
                              source_sha256=digest, reduction=args.reduction)
    elif args.command == 'search':
        result = {'query': args.query, 'image_root': index.meta['image_root'],
                  'candidates': index.search(args.query, encoder, args.top_k)}
    elif args.command == 'predict':
        predictions = []
        for start in range(0, len(args.images), args.batch_size):
            images, sources = [], []
            try:
                for path in args.images[start:start + args.batch_size]:
                    image, digest = read_image(path)
                    images.append(image)
                    sources.append((path, digest))
                for (path, digest), ranking in zip(sources, index.classify(images, encoder, args.top_k)):
                    collisions = index.meta.get('prompt_collisions', {})
                    # Recompute for older archives rather than trusting an absent field.
                    if not collisions:
                        collisions = prompt_collisions(index.items)
                    tied = collisions.get(ranking[0]['id'], [])
                    predictions.append({'image_path': path, 'image_sha256': digest,
                                        'predicted_class_id': None if tied else ranking[0]['id'],
                                        'indistinguishable_class_ids': tied,
                                        'status': 'ambiguous_prompt' if tied else 'ranked_only',
                                        'confidence_calibrated': None, 'candidates': ranking})
            finally:
                for image in images:
                    image.close()
        result = {'predictions': predictions}
    elif args.command == 'evaluate':
        result = retrieval_report(index, jsonl(args.queries), encoder)
    elif args.command == 'evaluate-classes':
        from .clip_class_eval import class_report, sha256_file
        result = class_report(index, args.queries, encoder, split=args.split,
                              archive_sha256=sha256_file(args.index))
    if args.command.startswith('build-'):
        index.save(args.output)
        result = {'index': args.output, 'kind': index.kind, 'count': len(index.items)}
    print(json.dumps({**result, 'encoder': encoder.config,
                      'note': 'Cosines are not probabilities; no open-set acceptance.'},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
