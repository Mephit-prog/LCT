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


class MockOCR:
    """Explicit offline fixture; never activated implicitly for HTTP requests."""
    def __init__(self, text: str):
        self.text = text

    def recognize(self, roi, deadline):
        if time.monotonic() >= deadline:
            return OCRResult(roi.roi_id, 'timeout', None, None, 'mock', 0, 'deadline')
        return OCRResult(roi.roi_id, 'ok' if normalize(self.text) else 'no_text',
                         self.text, normalize(self.text), 'mock', 0)
