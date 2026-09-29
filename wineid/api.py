"""FastAPI backend for the wine-label recogniser (frontend integration layer).

This module is a thin HTTP adapter around the existing :class:`wineid.pipeline.Pipeline`
and :class:`wineid.search.Index`. It deliberately does **not** change recognition
semantics: the pipeline still abstains by default (``roi_mode='refuse'``), OCR is
only called for an explicitly enabled ROI mode, and ``score``/cosine are never
presented as probabilities.

Endpoints
---------
Frontend-facing (prefix ``/api``):
  * ``GET  /api/health``            readiness + active configuration;
  * ``GET  /api/wines``             catalog list / autocomplete (``q``, ``limit``, ``offset``);
  * ``GET  /api/wines/{slug}``      one catalog card;
  * ``POST /api/search``            text branch -> ranked candidates (JSON);
  * ``POST /api/recognize``         multipart ``image`` (optional manual ``bbox``) -> pipeline result;
  * ``GET  /api/media/{filename}``  candidate upload image (only filenames from the
                                    candidate-links artifact are served);
  * ``GET  /api/catalog/collisions`` catalog identifiability diagnostics.

Compatibility with ``eval/participant_test.sh`` and the legacy stdlib server:
  * ``GET  /ready``                 ``{"ready": ...}`` (503 when not ready);
  * ``POST /v1/recognize``          same result shape as the stdlib server;
  * ``POST /v1/eval/predict``       forced Top-1 for valid images, independent of acceptance.

Auth and limits
---------------
``WINE_TOKEN`` (or ``token=``) enables ``Authorization: Bearer`` on the POST
endpoints (``/ready`` and ``/api/health`` stay open). Concurrent recognition
requests are bounded by ``WINE_MAX_WORKERS`` (or ``max_workers=``); saturation
returns ``503 busy`` instead of queueing. Request bodies larger than
``MAX_BYTES + 65536`` are rejected with ``413``. Internal errors never leak
details.

Run with ``python -m wineid.api`` (uvicorn); see ``requirements-api.txt``.
"""
import asyncio
import hmac
import json
import math
import os
import time
from pathlib import Path

import anyio
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from .catalog import Wine, structured_fields
from .pipeline import Pipeline
from .roi import MAX_BYTES
from .search import Index

MAX_BODY = MAX_BYTES + 65536
MAX_SEARCH_BODY = 16 * 1024
DEFAULT_CORS = ('http://localhost:5173', 'http://localhost:3000',
                'http://127.0.0.1:5173', 'http://127.0.0.1:3000')
DEFAULT_REQUEST_TIMEOUT = 8.5

class BoundedBody:
    """ASGI upload admission before FastAPI spools multipart files to disk.

    Buffers at most MAX_BODY bytes, including requests without Content-Length.
    The same monotonic start is passed on to the recognition pipeline.
    """
    def __init__(self, app, timeout, *, token=None, max_uploads=8):
        self.app, self.timeout, self.token = app, timeout, token
        self.slots = asyncio.Semaphore(max_uploads)

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('method') != 'POST':
            return await self.app(scope, receive, send)
        started = time.monotonic()
        raw_headers = scope.get('headers', ())
        headers = dict(raw_headers)
        path = scope.get('path')
        if path not in ('/api/search', '/api/recognize', '/v1/recognize', '/v1/eval/predict'):
            return await JSONResponse(status_code=404, content=_error('not_found'))(scope, receive, send)
        if self.token is not None and (sum(key == b'authorization' for key, _ in raw_headers) != 1
                or not hmac.compare_digest(
                    headers.get(b'authorization', b''), ('Bearer ' + self.token).encode())):
            return await JSONResponse(status_code=401, content=_error('unauthorized'))(scope, receive, send)
        if self.slots.locked():
            return await JSONResponse(status_code=503, content=_error('busy'))(scope, receive, send)
        await self.slots.acquire()  # no intervening await after locked(): fail fast, no queue
        try:
            limit = MAX_SEARCH_BODY if path == '/api/search' else MAX_BODY
            length = headers.get(b'content-length', b'')
            encoding = headers.get(b'transfer-encoding', b'')
            if (sum(key == b'content-length' for key, _ in raw_headers) > 1
                    or (encoding and (encoding.lower() != b'chunked' or length))
                    or (length and (not length.isdigit() or int(length) > limit))):
                return await JSONResponse(status_code=413, content=_error('invalid_body_size'))(scope, receive, send)
            if headers.get(b'content-encoding', b'identity').lower() != b'identity':
                return await JSONResponse(status_code=400, content=_error('invalid_body_encoding'))(scope, receive, send)
            chunks, total = [], 0
            while True:
                remaining = self.timeout - (time.monotonic() - started)
                if remaining <= 0:
                    return await JSONResponse(status_code=408, content=_error('deadline'))(scope, receive, send)
                try:
                    message = await asyncio.wait_for(receive(), timeout=remaining)
                except asyncio.TimeoutError:
                    return await JSONResponse(status_code=408, content=_error('deadline'))(scope, receive, send)
                if message['type'] == 'http.disconnect':
                    return
                if message['type'] != 'http.request':
                    return await JSONResponse(status_code=400, content=_error('invalid_body'))(scope, receive, send)
                part = message.get('body', b'')
                total += len(part)
                if total > limit:
                    return await JSONResponse(status_code=413, content=_error('invalid_body_size'))(scope, receive, send)
                chunks.append(part)
                if not message.get('more_body', False):
                    break
            scope.setdefault('state', {})['started'] = started
            body = b''.join(chunks)
            served = False
            async def replay():
                nonlocal served
                if not served:
                    served = True
                    return {'type': 'http.request', 'body': body, 'more_body': False}
                return await receive()
            return await self.app(scope, replay, send)
        finally:
            self.slots.release()


# ---------------------------------------------------------------------------
# Response schemas (also the OpenAPI contract consumed by the frontend)
# ---------------------------------------------------------------------------


class ErrorOut(BaseModel):
    status: str = 'error'
    slug: str | None = None
    reason_codes: list[str]


class WineCard(BaseModel):
    slug: str
    name: str
    producer: str
    color: str = ''
    vintage: str = ''
    grape: str = ''
    category: str = ''
    sweetness: str = ''
    media_url: str | None = None
    media_mapping_status: str | None = None


class CandidateOut(BaseModel):
    wine_id: str
    slug: str
    rank: int
    score: float = Field(description='retrieval score [0,100], NOT a probability')
    fuzzy_score: int | None = None
    edit_distance: int | None = None
    normalized_distance: float | None = None
    match: str | None = None
    blend_score: float | None = None
    attribute_score: float | None = None
    family_size: int | None = None
    signals: dict | None = None
    wine: WineCard | None = None


class WineListOut(BaseModel):
    total: int
    count: int
    limit: int
    offset: int
    items: list[WineCard]


class SearchIn(BaseModel):
    text: str = Field(max_length=4096)
    k: int = 20
    vintage_mode: str = 'soft'


class SearchOut(BaseModel):
    text: str
    backend: str
    candidates: list[CandidateOut]


class HealthOut(BaseModel):
    ready: bool
    catalog_ready: bool
    recognition_ready: bool
    acceptance_ready: bool
    catalog_sha256: str | None = None
    wine_count: int
    ocr_configured: bool
    ocr_ready: bool
    roi_mode: str
    policy_configured: bool
    search_backend: str
    media_configured: bool


class RecognizeOut(BaseModel):
    status: str
    slug: str | None = None
    reason_codes: list[str] = []
    versions: dict = {}
    timings_ms: dict = {}
    candidates: list[CandidateOut] = []
    ocr: dict | None = None
    roi: dict | None = None
    margin: float | None = None
    confidence: dict | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_media_links(path: str | Path) -> dict[str, dict]:
    """Load the candidate slug->media artifact; never treats links as verified."""
    path = Path(path)
    links: dict[str, dict] = {}
    if not path.is_file():
        return links
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        slug = row.get('slug')
        if not slug:
            continue
        links[slug] = {'image_path': row.get('image_path'),
                       'mapping_status': row.get('mapping_status'),
                       'image_sha256': row.get('image_sha256')}
    return links


def _error(reason: str) -> dict:
    return {'status': 'error', 'slug': None, 'reason_codes': [reason]}


def _parse_bbox(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    parts = [part.strip() for part in value.split(',')]
    if len(parts) != 4:
        raise HTTPException(status_code=400, detail='invalid_bbox')
    try:
        box = tuple(int(part) for part in parts)
    except ValueError:
        raise HTTPException(status_code=400, detail='invalid_bbox')
    return box  # type: ignore[return-value]


def create_app(pipeline: Pipeline, *, media_links: dict[str, dict] | None = None,
               uploads_root: str | Path | None = None, media_base_url: str | None = None,
               token: str | None = None, max_workers: int = 8, cors_origins=(),
               request_timeout: float = DEFAULT_REQUEST_TIMEOUT) -> FastAPI:
    """Build the FastAPI application around an already constructed pipeline.

    ``media_links`` maps slug -> ``{"image_path", "mapping_status", ...}`` from
    ``artifacts/csv-media-links.jsonl`` (candidates, not verified relations).
    ``uploads_root`` is the directory the candidate filenames are served from;
    ``media_base_url`` overrides the served URL prefix (e.g. an external CDN),
    otherwise ``/api/media/<filename>`` is used when ``uploads_root`` exists.
    ``token`` enables Bearer auth on POST endpoints; ``cors_origins`` configures
    the frontend dev-server origins (``('*',)`` allows all).
    """
    if type(max_workers) is not int or max_workers < 1:
        raise ValueError('max_workers must be a positive integer')
    if not isinstance(request_timeout, (int, float)) or not math.isfinite(request_timeout) or request_timeout <= 0:
        raise ValueError('invalid request_timeout')
    if pipeline.policy is not None and request_timeout != pipeline.request_timeout:
        raise ValueError('request timeout differs from calibrated policy profile')
    media_links = media_links or {}
    root = Path(uploads_root).resolve() if uploads_root else None
    if root is not None and not root.is_dir():
        root = None
    allowed_media = {link['image_path'] for link in media_links.values()
                     if link.get('image_path')}
    media_prefix = media_base_url.rstrip('/') + '/' if media_base_url else None

    index: Index = pipeline.index
    wines: list[Wine] = index.wines
    by_slug = {wine.slug: wine for wine in wines}

    app = FastAPI(title='Wine Label Recogniser API', version='0.1.0',
                  description='Experimental research prototype. Recognition is '
                              'not production-ready and abstains by default. '
                              'Scores are retrieval scores, not probabilities.')
    slots = anyio.Semaphore(max_workers)

    def require_auth(request: Request) -> None:
        if token is None:
            return
        if request.headers.get('authorization') != 'Bearer ' + token:
            raise HTTPException(status_code=401, detail='unauthorized')

    def card(wine: Wine) -> dict:
        link = media_links.get(wine.slug) or {}
        path = link.get('image_path')
        if not path:
            media_url = None
        elif media_prefix:
            media_url = media_prefix + path
        elif root is not None:
            media_url = '/api/media/' + path
        else:
            media_url = None
        fields = structured_fields(wine.raw or {})
        return {'slug': wine.slug, 'name': wine.name, 'producer': wine.producer,
                'color': fields['color'], 'vintage': fields['vintage'],
                'grape': fields['grape'], 'category': fields['category'],
                'sweetness': fields['sweetness'],
                'media_url': media_url,
                'media_mapping_status': link.get('mapping_status')}

    def candidate_out(candidate: dict) -> dict:
        slug = candidate.get('slug')
        wine = by_slug.get(slug) if slug else None
        return {**candidate, 'wine': card(wine) if wine else None}

    def enrich(result: dict) -> dict:
        out = dict(result)
        out['candidates'] = [candidate_out(c) for c in out.get('candidates', [])]
        return out

    async def run_recognition(data: bytes, bbox, roi_mode=None, force_top1=False,
                              started=None) -> dict:
        try:
            slots.acquire_nowait()
        except anyio.WouldBlock:
            raise HTTPException(status_code=503, detail='busy')
        try:
            remaining = max(0, request_timeout - (time.monotonic() - started)) if started else request_timeout
            if remaining <= 0:
                raise HTTPException(status_code=408, detail='deadline')
            result = await run_in_threadpool(pipeline.predict, data, bbox=bbox,
                                             seconds=remaining, roi_mode=roi_mode,
                                             force_top1=force_top1)
        finally:
            slots.release()
        if not force_top1 and result['status'] == 'error':
            if any(code in result['reason_codes'] for code in
                   ('ocr_capacity_exceeded', 'vision_capacity_exceeded', 'busy')):
                raise HTTPException(status_code=503, detail='busy')
            if any(code in result['reason_codes'] for code in
                   ('vision_worker_failed', 'vision_worker_unavailable')):
                raise HTTPException(status_code=503, detail='vision_unavailable')
            if any(code in result['reason_codes'] for code in
                   ('worker_unavailable', 'worker_failed', 'inference_failed')):
                raise HTTPException(status_code=503, detail='ocr_unavailable')
        return enrich(result)

    async def check_parts(request: Request, allowed: set[str]) -> None:
        form = await request.form()
        parts = form.multi_items()
        if (len(parts) > len(allowed) or len({key for key, _ in parts}) != len(parts)
                or any(key not in allowed for key, _ in parts)
                or sum(key == 'image' for key, _ in parts) != 1):
            raise HTTPException(status_code=400, detail='invalid_multipart')

    async def read_image(image: UploadFile) -> bytes:
        data = await image.read()
        if not data or len(data) > MAX_BYTES:
            raise HTTPException(status_code=413, detail='invalid_size')
        return data

    # -- exception shaping (no internal details leak) -----------------------
    @app.exception_handler(StarletteHTTPException)
    async def _http_exception(_request: Request, exc: StarletteHTTPException):
        reason = exc.detail if isinstance(exc.detail, str) else 'error'
        return JSONResponse(status_code=exc.status_code, content=_error(reason))

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, _exc: Exception):
        return JSONResponse(status_code=500, content=_error('internal'))

    @app.middleware('http')
    async def _limit_body(request: Request, call_next):
        if not hasattr(request.state, 'started'):
            request.state.started = time.monotonic()
        length = request.headers.get('content-length')
        if request.method == 'POST' and length is not None:
            try:
                size = int(length)
            except ValueError:
                return JSONResponse(status_code=413, content=_error('invalid_body_size'))
            if size > MAX_BODY:
                return JSONResponse(status_code=413, content=_error('invalid_body_size'))
        return await call_next(request)

    # -- health -------------------------------------------------------------
    def health_payload() -> dict:
        catalog_ready = bool(wines) and pipeline.csv_sha256 == wines[0].csv_sha256
        # Catalog readiness is not recognition readiness (STATUS.md B06).
        return {'ready': catalog_ready, 'catalog_ready': catalog_ready,
                'recognition_ready': pipeline.recognition_ready,
                'acceptance_ready': pipeline.acceptance_ready,
                'catalog_sha256': pipeline.csv_sha256,
                'wine_count': len(wines), 'ocr_configured': pipeline.ocr is not None,
                'ocr_ready': pipeline.ocr_ready, 'roi_mode': pipeline.roi_mode,
                'policy_configured': pipeline.policy is not None,
                'search_backend': index.backend,
                'media_configured': root is not None}

    @app.get('/api/health', response_model=HealthOut)
    async def health():
        return health_payload()

    @app.get('/api/health/recognition')
    async def recognition_health():
        payload = health_payload()
        return JSONResponse(status_code=200 if payload['recognition_ready'] else 503, content=payload)

    @app.get('/ready')
    async def ready():
        payload = health_payload()
        return JSONResponse(status_code=200 if payload['ready'] else 503, content=payload)

    # -- catalog ------------------------------------------------------------
    @app.get('/api/wines', response_model=WineListOut)
    async def list_wines(q: str | None = None, limit: int = 20, offset: int = 0):
        if not 1 <= limit <= 200 or offset < 0:
            raise HTTPException(status_code=400, detail='invalid_pagination')
        if q and q.strip():
            wanted = min(offset + limit, 200)
            candidates = await run_in_threadpool(index.search, q, wanted, 100, 'soft')
            sliced = candidates[offset:offset + limit]
            items = [card(by_slug[c.slug]) for c in sliced if c.slug in by_slug]
            total = len(candidates)
        else:
            items = [card(wine) for wine in wines[offset:offset + limit]]
            total = len(wines)
        return {'total': total, 'count': len(items), 'limit': limit,
                'offset': offset, 'items': items}

    @app.get('/api/wines/{slug}', response_model=WineCard)
    async def get_wine(slug: str):
        wine = by_slug.get(slug)
        if wine is None:
            raise HTTPException(status_code=404, detail='not_found')
        return card(wine)

    @app.get('/api/catalog/collisions')
    async def collisions():
        stats = await run_in_threadpool(index.collision_stats)
        return {'note': 'Indistinguishable canonical-text groups; exact Top-1 is '
                        'not justified inside a group.',
                **stats}

    @app.get('/api/aliases')
    async def producer_aliases():
        """Known winery aliases mapping canonical producer to recognized variants."""
        if index.aliases is None:
            return {'count': 0, 'producers': {}}
        variants_map = {k: sorted(v) for k, v in index.aliases.variants_by_canonical().items()}
        return {'count': len(variants_map), 'producers': variants_map}

    # -- text search --------------------------------------------------------
    @app.post('/api/search', response_model=SearchOut,
              dependencies=[Depends(require_auth)])
    async def search(payload: SearchIn):
        if payload.vintage_mode not in ('soft', 'hard'):
            raise HTTPException(status_code=400, detail='invalid_vintage_mode')
        if not 1 <= payload.k <= 100:
            raise HTTPException(status_code=400, detail='invalid_k')
        if not payload.text.strip():
            return {'text': payload.text, 'backend': index.backend, 'candidates': []}
        candidates = await run_in_threadpool(index.search, payload.text,
                                             payload.k, 100, payload.vintage_mode)
        return {'text': payload.text, 'backend': index.backend,
                'candidates': [candidate_out(c.__dict__) for c in candidates]}

    # -- recognition --------------------------------------------------------
    @app.post('/api/recognize', response_model=RecognizeOut,
              dependencies=[Depends(require_auth)])
    @app.post('/v1/recognize', response_model=RecognizeOut,
              dependencies=[Depends(require_auth)])
    async def recognize(request: Request, image: UploadFile = File(...), bbox: str | None = Form(None),
                        roi_mode: str | None = Form(None)):
        if roi_mode is not None and roi_mode not in (
                'refuse', 'full_frame_experimental', 'auto_bbox_experimental', 'center_80_crop'):
            raise HTTPException(status_code=400, detail='invalid_roi_mode')
        await check_parts(request, {'image', 'bbox', 'roi_mode'})
        data = await read_image(image)
        return await run_recognition(data, _parse_bbox(bbox), roi_mode=roi_mode,
                                     started=request.state.started)

    @app.post('/v1/eval/predict', dependencies=[Depends(require_auth)])
    async def eval_predict(request: Request, image: UploadFile = File(...)):
        await check_parts(request, {'image'})
        data = await read_image(image)
        result = await run_recognition(data, None, force_top1=True,
                                       started=request.state.started)
        if result.get('slug') is None:
            raise HTTPException(status_code=422 if result['status'] == 'unreadable'
                                else 503, detail=result['reason_codes'][0])
        return {'slug': result['slug']}

    # -- candidate media ----------------------------------------------------
    @app.get('/api/media/{name}')
    async def media(name: str):
        if root is None or name not in allowed_media or Path(name).name != name:
            raise HTTPException(status_code=404, detail='not_found')
        path = root / name
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise HTTPException(status_code=404, detail='not_found')
        return FileResponse(path)

    app.add_middleware(BoundedBody, timeout=request_timeout, token=token,
                       max_uploads=max_workers)
    if cors_origins:
        # Outermost: attach CORS headers even to pre-body 401/413/503 responses.
        app.add_middleware(CORSMiddleware, allow_origins=list(cors_origins),
                           allow_methods=['*'], allow_headers=['*'],
                           allow_credentials=False)
    return app


def main() -> None:
    import uvicorn

    from .runtime import open_runtime
    with open_runtime(os.environ) as runtime:
        media_links = load_media_links(os.environ.get('WINE_MEDIA_LINKS',
                                                      'artifacts/csv-media-links.jsonl'))
        uploads = os.environ.get('WINE_UPLOADS',
                                 'prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads')
        origins = tuple(o.strip() for o in os.environ.get(
            'WINE_CORS_ORIGINS', ','.join(DEFAULT_CORS)).split(',') if o.strip())
        app = create_app(runtime.pipeline, media_links=media_links, uploads_root=uploads,
                         media_base_url=os.environ.get('WINE_MEDIA_BASE_URL') or None,
                         token=os.environ.get('WINE_TOKEN') or None,
                         max_workers=runtime.settings.max_workers, cors_origins=origins,
                         request_timeout=runtime.settings.timeout)
        uvicorn.run(app, host=runtime.settings.host, port=runtime.settings.port)


if __name__ == '__main__':
    main()
