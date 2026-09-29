"""Offline frozen Jina CLIP v2 image-to-image features on manually verified ROI only.

The checkpoint must be licensed/reviewed before use. Cosines are not probabilities.
"""
import hashlib
import io
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image
from .data import jsonl
from .roi import manual_roi, decode, MAX_BYTES
from .jina_clip import (JinaCLIPEmbedder, MODEL, REVISION, PREPROCESS,
                        IncompatibleEmbeddings, unit_vectors)

# Retain the gallery API/metadata key; pretrained now means an HF weights revision.
PRETRAINED = REVISION
CROP_VERSION = 'oriented-rgb-bbox-v1'
ROLES = ('front', 'back', 'neck')


IncompatibleGallery = IncompatibleEmbeddings
_unit_vectors = unit_vectors


class CLIPEmbedder(JinaCLIPEmbedder):
    """Compatibility adapter for PNG crops; also exposes Jina text/image encoders."""
    def __init__(self, model_name=MODEL, pretrained=PRETRAINED, **kwargs):
        super().__init__(model_name, pretrained, **kwargs)
        self.pretrained = pretrained

    def __call__(self, pngs, *, model_name, pretrained):
        if (model_name, pretrained) != (self.model_name, self.pretrained):
            raise IncompatibleGallery('embedding checkpoint mismatch')
        images = []
        try:
            for data in pngs:
                images.append(decode(data, time.monotonic() + 30)[0])
            return self.encode_images(images)
        finally:
            for image in images:
                image.close()


def embed_pngs(pngs, model_name=MODEL, pretrained=PRETRAINED, **kwargs):
    """Convenience entry point; requires explicit allow_remote_code=True approval."""
    return CLIPEmbedder(model_name, pretrained, **kwargs)(
        pngs, model_name=model_name, pretrained=pretrained)


def build_gallery(manifest, catalog, output, embed=None,
                  model_name=MODEL, pretrained=PRETRAINED, batch_size=8,
                  allow_remote_code=False, device=None, adapter_path=None, synthetic_views=0):
    """Build a verified, versioned gallery in bounded batches; no filename inference.

    Passing `embed` permits deterministic test fixtures without downloading weights.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be positive')
    if type(synthetic_views) is not int or not 0 <= synthetic_views <= 5:
        raise ValueError('synthetic_views must be in [0,5]')
    path = Path(manifest).resolve()
    records = jsonl(path)
    known = {w.wine_id for w in catalog}
    if not catalog or len({w.csv_sha256 for w in catalog}) != 1:
        raise IncompatibleGallery('catalog must have one version')
    ids, image_hashes, sources, chunks, pending = set(), set(), [], [], []
    if adapter_path and embed is not None:
        raise ValueError('adapter_path cannot be combined with a custom embedder')
    adapter_sha256 = None
    if embed is None and adapter_path:
        from .fusion_vision import AdaptedEncoder
        if (model_name, pretrained) != (MODEL, REVISION):
            raise IncompatibleGallery('adapter requires pinned Jina encoder')
        encoder = AdaptedEncoder(adapter=adapter_path, revision=pretrained,
                                 batch_size=batch_size, device=device,
                                 allow_remote_code=allow_remote_code)
        adapter_sha256 = encoder.adapter_sha256
        def embed(pngs, *, model_name, pretrained):
            images = []
            try:
                for png in pngs:
                    images.append(decode(png, time.monotonic() + 30)[0])
                return encoder.encode_images(images)
            finally:
                for image in images:
                    image.close()
    if embed is None:
        embed = CLIPEmbedder(model_name, pretrained, batch_size=batch_size,
                            device=device, allow_remote_code=allow_remote_code)
    dim = None

    def flush():
        nonlocal dim
        if pending:
            vectors = _unit_vectors(embed(pending, model_name=model_name, pretrained=pretrained),
                                    len(pending), dim)
            dim = vectors.shape[1]
            chunks.append(vectors)
            pending.clear()

    for row in records:
        asset = row['asset_id']
        if (not asset or asset in ids or row.get('wine_id') not in known
                or row.get('role') not in ROLES or not row.get('source_group')):
            raise IncompatibleGallery('duplicate asset, unknown wine, role or source group')
        ids.add(asset)
        if (row.get('mapping_status') != 'verified' or not row.get('verified_by')
                or not row.get('verification_method')):
            raise IncompatibleGallery('gallery must be manually verified')
        photo = (path.parent / row['image_path']).resolve()
        if not photo.is_relative_to(path.parent):
            raise IncompatibleGallery('unsafe asset path')
        with photo.open('rb') as source:
            data = source.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise IncompatibleGallery('asset exceeds image size limit')
        if hashlib.sha256(data).hexdigest() != row.get('image_sha256'):
            raise IncompatibleGallery('asset hash mismatch')
        if row['image_sha256'] in image_hashes:
            raise IncompatibleGallery('duplicate reference image hash')
        image_hashes.add(row['image_sha256'])
        roi = manual_roi(data, tuple(row['bbox']), time.monotonic()+8)
        pending.append(roi.crop)
        sources.append({key: row[key] for key in ('asset_id', 'wine_id', 'role', 'source_group')})
        sources[-1]['image_sha256'] = row['image_sha256']
        if len(pending) >= batch_size:
            flush()
        if synthetic_views:
            from .augment import field_view
            with Image.open(io.BytesIO(roi.crop)) as reference:
                for view_number in range(synthetic_views):
                    seed = int.from_bytes(hashlib.sha256(
                        f"{row['asset_id']}:{view_number}".encode()).digest()[:8], 'big')
                    variant = field_view(reference, np.random.default_rng(seed))
                    buf = io.BytesIO()
                    variant.save(buf, format='PNG')
                    variant.close()
                    view_bytes = buf.getvalue()
                    digest = hashlib.sha256(view_bytes).hexdigest()
                    asset_id = f"synthetic:{row['asset_id']}:{view_number}"
                    if digest in image_hashes or asset_id in ids:
                        raise IncompatibleGallery('duplicate synthetic view')
                    image_hashes.add(digest)
                    ids.add(asset_id)
                    pending.append(view_bytes)
                    sources.append({'asset_id': asset_id, 'wine_id': row['wine_id'],
                                    'role': row['role'], 'source_group': row['source_group'],
                                    'image_sha256': digest, 'reference_asset_id': row['asset_id'],
                                    'variant_seed': seed})
                    if len(pending) >= batch_size:
                        flush()
    if not pending and not chunks:
        raise IncompatibleGallery('empty gallery')
    flush()
    vectors = np.concatenate(chunks)
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    meta = {'model': model_name, 'pretrained': pretrained, 'preprocess': PREPROCESS,
            'crop_version': CROP_VERSION, 'csv_sha256': catalog[0].csv_sha256,
            'manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'adapter_sha256': adapter_sha256, 'synthetic_views': synthetic_views,
            'sources': sources, 'embedding_dim': dim}
    # Complete archive first; never leave a partially written output.
    with tempfile.NamedTemporaryFile(dir=out.parent, suffix='.npz', delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        np.savez_compressed(temporary, embeddings=vectors,
                            metadata=np.array(json.dumps(meta, ensure_ascii=False)))
        if out.exists():
            raise FileExistsError(out)
        try:
            os.link(temporary, out)  # exclusive creation; no overwrite of existing gallery
        except OSError:
            # Filesystems without hardlink support (FAT/exFAT, some network
            # mounts): rename is still atomic within the directory (P2-14).
            if out.exists():
                raise FileExistsError(out)
            os.rename(temporary, out)
    finally:
        temporary.unlink(missing_ok=True)
    return meta


class Gallery:
    def __init__(self, path, *, model, pretrained, csv_sha256, preprocess=PREPROCESS):
        try:
            with np.load(path, allow_pickle=False) as bundle:
                meta = json.loads(str(bundle['metadata'].item()))
                vectors = bundle['embeddings'].copy()
        except (KeyError, ValueError, TypeError, OSError) as exc:
            raise IncompatibleGallery('invalid gallery archive') from exc
        for key, expected in (('model', model), ('pretrained', pretrained),
                              ('csv_sha256', csv_sha256), ('preprocess', preprocess),
                              ('crop_version', CROP_VERSION)):
            if meta.get(key) != expected:
                raise IncompatibleGallery(f'mismatch: {key}')
        sources = meta.get('sources')
        if (not isinstance(sources, list) or not sources or vectors.ndim != 2
                or vectors.shape != (len(sources), meta.get('embedding_dim'))
                or not np.isfinite(vectors).all()):
            raise IncompatibleGallery('invalid embedding dimensions')
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
            raise IncompatibleGallery('unnormalized embeddings')
        if any(s.get('role') not in ROLES or not s.get('asset_id') or not s.get('wine_id')
               or not s.get('source_group') or not s.get('image_sha256') for s in sources):
            raise IncompatibleGallery('invalid reference provenance')
        if any(len({s[key] for s in sources}) != len(sources)
               for key in ('asset_id', 'image_sha256')):
            raise IncompatibleGallery('duplicate reference provenance')
        if not isinstance(meta.get('manifest_sha256'), str) or len(meta['manifest_sha256']) != 64:
            raise IncompatibleGallery('invalid manifest fingerprint')
        self.archive_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        self.vectors, self.sources, self.meta = vectors, sources, meta
        self._by_wine_role = {}
        for i, s in enumerate(sources):
            self._by_wine_role.setdefault((s['wine_id'], s['role']), []).append(i)

    def scores(self, query_vector, candidate_ids, role='front'):
        """Per-wine best verified role cosine, reference and count; missing is None."""
        if role not in ROLES:
            raise ValueError('invalid role')
        q = np.asarray(query_vector, dtype='float32')
        if q.shape != (self.vectors.shape[1],) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
            raise IncompatibleGallery('invalid query vector')
        cosines = self.vectors @ (q / np.linalg.norm(q))
        out = {}
        for wine_id in candidate_ids:
            indices = self._by_wine_role.get((wine_id, role), ())
            best = max(indices, key=lambda i: cosines[i]) if indices else None
            out[wine_id] = {'cosine': float(cosines[best]) if best is not None else None,
                            'reference_id': self.sources[best]['asset_id'] if best is not None else None,
                            'reference_count': len(indices)}
        return out
