"""Request-scoped MinerU image adapter. No implicit alternative provider or acceptance policy."""
import hashlib
import tempfile
import time
import zipfile
from pathlib import Path

from .catalog import normalize
from .mineru_cloud import MinerUBackend, MinerUError
from .ocr import OCRResult


class MinerUImageOCR:
    model = 'mineru-cloud-pipeline-cyrillic'
    ready = True  # Configured for requests; not a remote health/acceptance probe.

    def __init__(self, token, *, backend_factory=None):
        if not token:
            raise ValueError('MINERU_TOKEN required')
        self.token = token
        self.backend_factory = backend_factory or (lambda: MinerUBackend(token, retries=0, max_zip_bytes=20_000_000))

    def recognize(self, roi, deadline):
        start = time.monotonic()
        raw, code = None, None
        status = 'provider_error'
        try:
            suffix = {'image/png': '.png', 'image/jpeg': '.jpg', 'image/webp': '.webp'}.get(roi.mime)
            if suffix is None or not roi.crop or len(roi.crop) > 10_000_000:
                raise MinerUError('invalid_image')
            with tempfile.TemporaryDirectory(prefix='wine-mineru-') as directory:
                source = Path(directory) / ('photo' + suffix)
                archive = Path(directory) / 'result.zip'
                if time.monotonic() >= deadline:
                    raise MinerUError('deadline')
                source.write_bytes(roi.crop)
                backend = self.backend_factory()
                try:
                    backend.deadline = deadline
                    digest = hashlib.sha256(roi.crop).hexdigest()
                    batch, url = backend.reserve(source.name, digest)
                    backend.upload(url, source)
                    while True:
                        backend._remaining(1)
                        row = backend.poll(batch, digest)
                        if row['state'] == 'done':
                            backend.download(row['full_zip_url'], archive)
                            break
                        if row['state'] == 'failed':
                            raise MinerUError('extract_failed')
                        backend._sleep(min(1.0, max(0, deadline - time.monotonic())))
                    backend._remaining(1)
                    with zipfile.ZipFile(archive) as z:
                        names = [n for n in z.namelist() if n.split('/')[-1] == 'full.md']
                        if len(names) != 1 or z.getinfo(names[0]).file_size > 1_000_000:
                            raise MinerUError('invalid_zip_outputs')
                        raw = z.read(names[0]).decode('utf-8')
                    backend._remaining(1)
                finally:
                    backend.close()
            status = 'ok' if normalize(raw) else 'no_text'
        except MinerUError as exc:
            code = exc.code
            status = 'timeout' if code == 'deadline' or time.monotonic() >= deadline else 'provider_error'
        except (OSError, ValueError, UnicodeError, zipfile.BadZipFile):
            code = 'invalid_response'
            status = 'timeout' if time.monotonic() >= deadline else 'provider_error'
        return OCRResult(roi.roi_id, status, raw, normalize(raw) if raw is not None else None,
                         self.model, round((time.monotonic() - start) * 1000), code)
