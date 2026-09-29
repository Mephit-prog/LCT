"""Optional image sources for the online blender; no weights downloaded on import."""
import hashlib
import time
from pathlib import Path
from threading import BoundedSemaphore

import numpy as np

from .blend import CandidateEvidence
from .jina_clip import JinaCLIPEmbedder, MODEL, REVISION, CODE_REVISION, PREPROCESS, unit_vectors
from .clip_zero_shot import EmbeddingIndex
from .text_policy import fingerprint
from .visual import Gallery, IncompatibleGallery


class AdaptedEncoder(JinaCLIPEmbedder):
    def __init__(self, *, adapter=None, **kwargs):
        super().__init__(**kwargs)
        self.adapter_sha256 = None
        if adapter:
            from .train_clip import load_adapter
            path = Path(adapter)
            with path.open('rb') as source:
                self.adapter_sha256 = hashlib.file_digest(source, 'sha256').hexdigest()
            meta = load_adapter(self.model, path, device=self.device)
            if meta.get('revision') != self.revision:
                raise ValueError('adapter revision does not match encoder')
            self.model.eval().requires_grad_(False)

    def close(self):
        # Drop the heavy torch graph at shutdown; do not keep a model alive via
        # a pipeline reference after its runtime context has been closed.
        self.model = None

    @property
    def config(self):
        return {**super().config, **({'adapter_sha256': self.adapter_sha256}
                                     if self.adapter_sha256 else {})}


class VisionSources:
    """One shared image encoding; class cosines + per-SKU max verified photo cosine.

    Refuses stale indexes/checkpoints at startup. Missing photo is not a zero
    similarity. Runtime may be injected with a fake encoder in offline tests.
    """
    def __init__(self, encoder, wines, *, classes=None, gallery=None, max_concurrent=1,
                 query_views=1):
        if classes is None and gallery is None:
            raise ValueError('no vision index')
        if type(max_concurrent) is not int or max_concurrent < 1:
            raise ValueError('max_concurrent must be positive')
        if query_views not in (1, 3):
            raise ValueError('query_views must be 1 or 3')
        catalog = {w.slug: w for w in wines}
        by_id = {w.wine_id: w for w in wines}
        if not catalog or len(catalog) != len(wines) or len(by_id) != len(wines) or len({w.csv_sha256 for w in wines}) != 1:
            raise ValueError('one nonempty catalog version required')
        sha = wines[0].csv_sha256
        if classes is not None:
            if classes.kind != 'classes' or classes.meta.get('source_sha256') != sha:
                raise ValueError('classes/catalog mismatch')
            classes.check_encoder(encoder)
            if any(item['id'] not in catalog for item in classes.items):
                raise ValueError('unknown class slug')
        if gallery is not None:
            config = encoder.config
            if (gallery.meta.get('csv_sha256') != sha
                    or gallery.meta['model'] != config.get('model')
                    or gallery.meta['pretrained'] != config.get('revision')
                    or gallery.meta['preprocess'] != config.get('preprocess')
                    or gallery.meta.get('adapter_sha256') != config.get('adapter_sha256')
                    or gallery.vectors.shape[1] != config['dimensions']
                    or any(s['wine_id'] not in by_id for s in gallery.sources)):
                raise IncompatibleGallery('gallery/encoder/catalog mismatch')
        self.encoder, self.classes, self.gallery = encoder, classes, gallery
        self.catalog, self.by_id = catalog, by_id
        self.slots = BoundedSemaphore(max_concurrent)
        self.query_views = query_views
        self.class_fingerprint = (getattr(classes, 'archive_sha256', fingerprint(classes.meta))
                                  if classes else None)

    @property
    def ready(self):
        return self.encoder is not None and getattr(self.encoder, 'model', True) is not None

    def close(self):
        encoder, self.encoder = self.encoder, None
        if hasattr(encoder, 'close'):
            encoder.close()

    def observe(self, image, deadline):
        """Encode PIL image once. Caller owns image and the shared deadline."""
        if not self.slots.acquire(blocking=False):
            raise RuntimeError('vision_capacity_exceeded')
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError('deadline')
            views = [('full', image)]
            if self.query_views == 3:
                w, h = image.size
                def crop(x0, y0, x1, y1):
                    return image.crop((round(w*x0), round(h*y0),
                                       max(round(w*x0)+1, round(w*x1)),
                                       max(round(h*y0)+1, round(h*y1))))
                views += [('center_80', crop(.1, .1, .9, .9)),
                          ('lower_center', crop(.22, .35, .78, .82))]
            try:
                images = [img for _, img in views]
                if hasattr(self.encoder, 'encode_images_deadline'):
                    encoded = self.encoder.encode_images_deadline(images, deadline)
                else:
                    encoded = self.encoder.encode_images(images)
                vectors = unit_vectors(encoded, len(views), self.encoder.config['dimensions'])
                if time.monotonic() >= deadline:
                    raise TimeoutError('deadline')
                out = []
                if self.classes is not None:
                    similarities = np.clip(self.classes.vectors @ vectors.T, -1, 1)
                    best_view = similarities.argmax(axis=1)
                    cos = similarities.max(axis=1)
                    order = np.argsort(-cos, kind='stable')
                    for rank, i in enumerate(order, 1):
                        out.append(CandidateEvidence(self.classes.items[i]['id'], 'class',
                                                    float(cos[i]), rank, 'cosine', self.class_fingerprint,
                                                    query_view=views[best_view[i]][0]))
                if self.gallery is not None:
                    scores = {}
                    for (view, _), vector in zip(views, vectors):
                        for wine_id, info in self.gallery.scores(vector, self.by_id).items():
                            if info['cosine'] is not None and (wine_id not in scores or
                                    info['cosine'] > scores[wine_id][0]['cosine']):
                                scores[wine_id] = (info, view)
                    ranked = sorted(((self.by_id[wine_id].slug, info, view)
                                     for wine_id, (info, view) in scores.items()),
                                    key=lambda row: (-row[1]['cosine'], row[0]))
                    for rank, (slug, info, view) in enumerate(ranked, 1):
                        out.append(CandidateEvidence(slug, 'photo', info['cosine'], rank,
                                                    'cosine', self.gallery.archive_sha256,
                                                    info['reference_id'], query_view=view))
                return out
            finally:
                for _, view in views[1:]:
                    view.close()
        finally:
            self.slots.release()


def load_vision(wines, *, classes_path=None, gallery_path=None, adapter_path=None,
                allow_remote_code=False, device=None, query_views=1, isolated=False,
                startup_timeout=120):
    """Explicit startup configuration; never construct a gallery from candidate links."""
    if not classes_path and not gallery_path:
        if adapter_path:
            raise ValueError('adapter needs a class index or verified gallery')
        return None
    if not wines:
        raise ValueError('empty catalog')
    if not allow_remote_code:
        raise ValueError('approve license and HF remote code explicitly')
    # Read metadata before expensive model initialization; fail fast on bad archives.
    sha = wines[0].csv_sha256
    classes = EmbeddingIndex.load(classes_path) if classes_path else None
    gallery = (Gallery(gallery_path, model=MODEL, pretrained=REVISION, csv_sha256=sha,
                       preprocess=PREPROCESS) if gallery_path else None)
    if classes and (classes.kind != 'classes' or classes.meta.get('source_sha256') != sha):
        raise ValueError('class index/catalog mismatch; rebuild index')
    adapter_sha = None
    if adapter_path:
        with Path(adapter_path).open('rb') as source:
            adapter_sha = hashlib.file_digest(source, 'sha256').hexdigest()
    if gallery and gallery.meta.get('adapter_sha256') != adapter_sha:
        raise IncompatibleGallery('gallery/adapter mismatch; rebuild gallery')
    if classes and (classes.meta['encoder'].get('adapter_sha256') != adapter_sha
                    or classes.meta['encoder'].get('model') != MODEL
                    or classes.meta['encoder'].get('revision') != REVISION
                    or classes.meta['encoder'].get('preprocess') != PREPROCESS):
        raise ValueError('class index/encoder mismatch; rebuild index')
    if isolated:
        from .vision_ipc import IsolatedVisionEncoder
        # The child also hashes the adapter. A changed checkpoint or model
        # config cannot be accepted by the already-loaded parent indexes.
        expected = {'model': MODEL, 'revision': REVISION,
                    'code_revision': CODE_REVISION,
                    'preprocess': PREPROCESS, 'text_task': None, 'normalize': 'l2',
                    'dimensions': 1024}
        if adapter_sha:
            expected['adapter_sha256'] = adapter_sha
        encoder = IsolatedVisionEncoder(adapter_path=adapter_path, device=device,
                                        startup_timeout=startup_timeout,
                                        expected_config=expected)
    else:
        encoder = AdaptedEncoder(adapter=adapter_path, revision=REVISION, device=device,
                                 allow_remote_code=True)
    try:
        if isolated:
            encoder.start()  # load weights/canary before the HTTP service starts
        return VisionSources(encoder, wines, classes=classes, gallery=gallery,
                             query_views=query_views)
    except BaseException:
        encoder.close()
        raise
