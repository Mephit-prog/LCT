"""One Mistral OCR request per permitted ROI; no retries beyond the shared deadline."""
import base64
import time
from dataclasses import dataclass
import httpx
from .catalog import normalize

MODEL = 'mistral-ocr-latest'


@dataclass(frozen=True)
class OCRResult:
    roi_id: str
    status: str
    raw_text: str | None
    normalized_text: str | None
    model: str
    latency_ms: int
    error_code: str | None = None
    # Reserved contract fields: deliberately not populated by MistralOCR (the
    # endpoint returns page-level markdown only). Kept for response schema
    # stability; consumers must treat them as "not filled".
    boxes: None = None
    confidence: None = None


class MistralOCR:
    def __init__(self, key: str, model: str = MODEL, endpoint: str = 'https://api.mistral.ai/v1/ocr'):
        self.key, self.model, self.endpoint = key, model, endpoint

    @property
    def ready(self):
        # Remote availability cannot be inferred from a configured key.
        return False

    def recognize(self, roi, deadline: float) -> OCRResult:
        start = time.monotonic()
        status, raw, code = 'provider_error', None, None
        try:
            remaining = deadline - start
            if remaining <= 0:
                raise httpx.TimeoutException('deadline')
            # Mistral OCR image_url accepts base64 data URI in a single-page document.
            uri = 'data:' + roi.mime + ';base64,' + base64.b64encode(roi.crop).decode('ascii')
            with httpx.Client(timeout=httpx.Timeout(remaining, connect=min(2, remaining))) as client:
                response = client.post(self.endpoint,
                    headers={'Authorization': 'Bearer ' + self.key},
                    json={'model': self.model, 'document': {'type': 'image_url', 'image_url': uri},
                          'include_image_base64': False})
            if time.monotonic() >= deadline:
                raise httpx.TimeoutException('deadline')
            if response.status_code != 200:
                code = 'rate_limited' if response.status_code == 429 else (
                    'upstream_5xx' if response.status_code >= 500 else 'upstream_4xx')
            else:
                pages = response.json()['pages']
                if not isinstance(pages, list) or not pages or not all(
                        isinstance(p, dict) and isinstance(p.get('markdown'), str) for p in pages):
                    code = 'invalid_response'
                else:
                    raw = '\n'.join(p['markdown'] for p in pages)
                    status = 'ok' if normalize(raw) else 'no_text'
        except (httpx.TimeoutException, TimeoutError):
            status, code = 'timeout', 'deadline'
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            code = 'invalid_response_or_network'
        return OCRResult(roi.roi_id, status, raw,
                         normalize(raw) if raw is not None else None, self.model,
                         round((time.monotonic()-start)*1000), code)


def configured_ocr(env):
    """Choose one explicit provider; never send photos to a remote API implicitly."""
    provider = env.get('WINE_OCR_PROVIDER', 'none')
    key = env.get('MISTRAL_API_KEY')
    if provider == 'none':
        return None
    if provider == 'mock':
        if env.get('WINE_ENABLE_MOCK_OCR') != '1':
            raise ValueError('mock OCR is only allowed with WINE_ENABLE_MOCK_OCR=1')
        return MockOCR(env.get('WINE_MOCK_OCR', ''))
    if provider == 'mistral':
        if not key:
            raise ValueError('MISTRAL_API_KEY required for mistral OCR')
        return MistralOCR(key)
    if provider == 'mineru':
        from .mineru_http import MinerUImageOCR
        return MinerUImageOCR(env.get('MINERU_TOKEN'))
    from .local_ocr import LocalOCR, SUPPORTED
    if provider == 'mineru_local':
        provider = 'mineru'
    if provider not in SUPPORTED:
        raise ValueError('invalid WINE_OCR_PROVIDER')
    mode = env.get('WINE_OCR_MODE', 'cold')
    if mode not in ('cold', 'warm') or (mode == 'warm' and provider != 'paddle'):
        raise ValueError('warm OCR mode only supports paddle')
    if mode == 'warm':
        from .warm_ocr import WarmPaddleOCR
        return WarmPaddleOCR(python=env.get('WINE_OCR_PYTHON'),
                             det_dir=env.get('WINE_PADDLE_DET_DIR'),
                             rec_dir=env.get('WINE_PADDLE_REC_DIR'),
                             manifest=env.get('WINE_OCR_MANIFEST'),
                             startup_timeout=float(env.get('WINE_OCR_STARTUP_TIMEOUT', 120)))
    return LocalOCR(provider, python=env.get('WINE_OCR_PYTHON'),
                    tesseract=env.get('WINE_TESSERACT', 'tesseract'),
                    tessdata_dir=env.get('WINE_TESSDATA_DIR'))


class MockOCR:
    """Explicit offline fixture; never activated implicitly for HTTP requests."""
    def __init__(self, text: str):
        self.text = text

    ready = True  # explicit offline fixture, not a model canary
    model = 'mock'

    def recognize(self, roi, deadline):
        if time.monotonic() >= deadline:
            return OCRResult(roi.roi_id, 'timeout', None, None, 'mock', 0, 'deadline')
        return OCRResult(roi.roi_id, 'ok' if normalize(self.text) else 'no_text',
                         self.text, normalize(self.text), 'mock', 0)
