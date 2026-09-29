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
  * ``POST /v1/eval/predict``       ``{"slug": "<exact slug>"}`` or ``{"slug": null}``.

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
import json
import os
from pathlib import Path

import anyio
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from .catalog import load_catalog, Wine
from .catalog import structured_fields
from .canonical import ProducerAliases
from .ocr import MistralOCR, MockOCR
from .pipeline import Pipeline
from .text_policy import load_policy
from .roi import MAX_BYTES
from .search import Index

MAX_BODY = MAX_BYTES + 65536
DEFAULT_CORS = ('http://localhost:5173', 'http://localhost:3000',
                'http://127.0.0.1:5173', 'http://127.0.0.1:3000')
DEFAULT_REQUEST_TIMEOUT = 8.5

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
    fuzzy_score: int
    edit_distance: int
    normalized_distance: float
    match: str
    wine: WineCard | None = None


class WineListOut(BaseModel):
    total: int
    count: int
    limit: int
    offset: int
    items: list[WineCard]


class SearchIn(BaseModel):
    text: str
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
    catalog_sha256: str | None = None
    wine_count: int
    ocr_configured: bool
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
    if cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=list(cors_origins),
                           allow_methods=['*'], allow_headers=['*'],
                           allow_credentials=False)

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

    async def run_recognition(data: bytes, bbox, roi_mode=None) -> dict:
        try:
            slots.acquire_nowait()
        except anyio.WouldBlock:
            raise HTTPException(status_code=503, detail='busy')
        try:
            result = await run_in_threadpool(pipeline.predict, data, bbox=bbox,
                                             seconds=request_timeout, roi_mode=roi_mode)
        finally:
            slots.release()
        return enrich(result)

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
        recognition_ready = (catalog_ready and pipeline.ocr is not None
                             and pipeline.roi_mode != 'refuse')
        return {'ready': catalog_ready, 'catalog_ready': catalog_ready,
                'recognition_ready': recognition_ready,
                'catalog_sha256': pipeline.csv_sha256,
                'wine_count': len(wines), 'ocr_configured': pipeline.ocr is not None,
                'roi_mode': pipeline.roi_mode,
                'policy_configured': pipeline.policy is not None,
                'search_backend': index.backend,
                'media_configured': root is not None}

    @app.get('/api/health', response_model=HealthOut)
    async def health():
        return health_payload()

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
    async def recognize(image: UploadFile = File(...), bbox: str | None = Form(None),
                        roi_mode: str | None = Form(None)):
        if roi_mode is not None and roi_mode not in (
                'refuse', 'full_frame_experimental', 'auto_bbox_experimental', 'center_80_crop'):
            raise HTTPException(status_code=400, detail='invalid_roi_mode')
        data = await read_image(image)
        return await run_recognition(data, _parse_bbox(bbox), roi_mode=roi_mode)

    @app.post('/v1/eval/predict', dependencies=[Depends(require_auth)])
    async def eval_predict(image: UploadFile = File(...)):
        data = await read_image(image)
        result = await run_recognition(data, None)
        return {'slug': result.get('slug')}

    # -- candidate media ----------------------------------------------------
    @app.get('/api/media/{name}')
    async def media(name: str):
        if root is None or name not in allowed_media or Path(name).name != name:
            raise HTTPException(status_code=404, detail='not_found')
        path = root / name
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise HTTPException(status_code=404, detail='not_found')
        return FileResponse(path)

    return app


def main() -> None:
    import uvicorn

    catalog = os.environ.get('WINE_CSV', 'strapi_output0709.csv')
    wines, report = load_catalog(catalog)
    policy_path = os.environ.get('WINE_POLICY')
    policy = load_policy(policy_path) if policy_path else None
    key = os.environ.get('MISTRAL_API_KEY')
    mock_ocr = os.environ.get('WINE_MOCK_OCR')
    ocr = MistralOCR(key) if key else (MockOCR(mock_ocr) if mock_ocr else None)
    aliases_path = os.environ.get('WINE_ALIASES', 'producer_aliases.txt')
    aliases = ProducerAliases.load(aliases_path) if Path(aliases_path).is_file() else None
    pipeline = Pipeline(Index(wines, aliases=aliases), ocr, policy,
                        report['csv_sha256'], os.environ.get('WINE_ROI_MODE', 'refuse'))

    media_links = load_media_links(os.environ.get('WINE_MEDIA_LINKS',
                                                  'artifacts/csv-media-links.jsonl'))
    uploads = os.environ.get('WINE_UPLOADS',
                             'prod-svoe-vino-strapi/prod-svoe-vino/strapi/uploads')
    origins = tuple(o.strip() for o in os.environ.get(
        'WINE_CORS_ORIGINS', ','.join(DEFAULT_CORS)).split(',') if o.strip())

    app = create_app(pipeline, media_links=media_links, uploads_root=uploads,
                     media_base_url=os.environ.get('WINE_MEDIA_BASE_URL') or None,
                     token=os.environ.get('WINE_TOKEN') or None,
                     max_workers=int(os.environ.get('WINE_MAX_WORKERS', 8)),
                     cors_origins=origins,
                     request_timeout=float(os.environ.get('WINE_REQUEST_TIMEOUT',
                                                          DEFAULT_REQUEST_TIMEOUT)))
    uvicorn.run(app, host=os.environ.get('WINE_HOST', '127.0.0.1'),
                port=int(os.environ.get('WINE_PORT', 8080)))


if __name__ == '__main__':
    main()
