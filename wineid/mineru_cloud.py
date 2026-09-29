"""MinerU Precision document and opt-in wine-label image OCR.

No token, URL or document contents are logged. POST reservations are never
retried after an ambiguous network failure: data_id is a correlation ID, NOT
an API guarantee of idempotency. All remote URLs are checked before fetching.
"""
import hashlib
import hmac
import io
import json
import ipaddress
import math
import os
import re
import stat
import tempfile
import time
import unicodedata
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

import httpx

BASE = 'https://mineru.net/api/v4'
MAX_FILE = 200_000_000
MAX_ZIP = 300_000_000
MAX_UNPACKED = 512_000_000
MAX_SOURCE = 1_000_000_000
UPLOAD_HOSTS = frozenset({'mineru.oss-cn-shanghai.aliyuncs.com'})
DOWNLOAD_HOSTS = frozenset({'cdn-mineru.openxlab.org.cn'})
SOURCE_EXTENSIONS = frozenset({'.pdf', '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.jp2'})
STATES = frozenset({'waiting-file', 'pending', 'running', 'converting', 'done', 'failed'})


class MinerUError(Exception):
    """Safe error code only, no upstream body, signed URL or token."""
    def __init__(self, code, *, retryable=False):
        super().__init__(code)
        self.code, self.retryable = code, retryable


@dataclass(frozen=True)
class MinerUConfig:
    model_version: str = 'pipeline'
    language: str = 'cyrillic'
    is_ocr: bool = True
    enable_formula: bool = False
    enable_table: bool = True
    callback: str | None = None
    seed: str | None = None

    def __post_init__(self):
        if self.model_version not in ('pipeline', 'vlm') or self.language not in ('cyrillic', 'east_slavic'):
            raise ValueError('invalid MinerU model/language')
        if any(type(v) is not bool for v in (self.is_ocr, self.enable_formula, self.enable_table)):
            raise ValueError('invalid OCR flags')
        if bool(self.callback) != bool(self.seed):
            raise ValueError('callback and seed must be provided together')
        if self.callback:
            # Public callbacks must be HTTPS and live behind authenticated ingress.
            host = urlsplit(self.callback).hostname
            if not host or host in ('localhost',) or host.endswith(('.local', '.internal')):
                raise ValueError('callback must use a public HTTPS host')
            try:
                ip = ipaddress.ip_address(host)
            except ValueError:
                ip = None
            if ip is not None and not ip.is_global:
                raise ValueError('callback must use a public HTTPS host')
            safe_url(self.callback, {host}, allow_query=False)
            if not re.fullmatch(r'[A-Za-z0-9_]{32,64}', self.seed):
                raise ValueError('invalid callback seed')

    @property
    def sha256(self):
        # Exclude callback address/seed: they do not change OCR output and are
        # secrets/operational endpoints, not cache contents.
        values = {k: v for k, v in asdict(self).items() if k not in ('callback', 'seed')}
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()

    def request(self, name, digest):
        data = {'files': [{'name': name, 'data_id': digest, 'is_ocr': self.is_ocr}],
                'model_version': self.model_version, 'language': self.language,
                'enable_formula': self.enable_formula, 'enable_table': self.enable_table}
        if self.callback:
            data.update(callback=self.callback, seed=self.seed)
        return data


def safe_url(value, hosts, *, allow_query=True):
    try:
        url = urlsplit(value)
        host = url.hostname
        port = url.port
    except (ValueError, TypeError, AttributeError) as exc:
        raise MinerUError('unsafe_remote_url') from exc
    if (url.scheme != 'https' or not host or host not in hosts or port not in (None, 443)
            or url.username or url.password or url.fragment or not url.path.startswith('/')
            or (url.query and not allow_query)):
        raise MinerUError('unsafe_remote_url')
    return value


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_file(path):
    path = Path(path).resolve(strict=True)
    if path.suffix.lower() not in SOURCE_EXTENSIONS or not path.is_file() or not 0 < path.stat().st_size <= MAX_FILE:
        raise ValueError('unsupported or oversized document part')
    return path


def prepare(path, spool, *, max_pages=200, max_source_bytes=MAX_SOURCE):
    """Split a PDF by both pages and byte size, never silently truncate pages.

    Output parts live in a private persistent spool until jobs are complete.
    Whole source PDFs are capped to limit pypdf's in-memory parser. Rendering /
    image DPI changes are deliberately NOT done without an annotated policy.
    """
    source = Path(path).resolve(strict=True)
    if source.suffix.lower() != '.pdf':
        return [(check_file(source), None)]
    if not 0 < source.stat().st_size <= max_source_bytes:
        raise ValueError('source PDF exceeds configured size limit')
    if not 1 <= max_pages <= 200:
        raise ValueError('invalid page limit')
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:
        raise ValueError('PDF splitting requires requirements-mineru.txt') from exc
    try:
        reader = PdfReader(str(source), strict=True)
        if reader.is_encrypted:
            raise ValueError('encrypted PDF is not supported')
        count = len(reader.pages)
    except ValueError:
        raise
    except Exception:
        raise ValueError('invalid source PDF') from None
    if count < 1 or count > 5000:
        raise ValueError('invalid PDF page count')
    if count <= max_pages and source.stat().st_size <= MAX_FILE:
        return [(source, (1, count))]
    spool = Path(spool)
    spool.mkdir(mode=0o700, parents=True, exist_ok=True)
    source_sha = digest_file(source)
    output = []
    start = 0
    while start < count:
        def render(end):
            writer = PdfWriter()
            for page in reader.pages[start:end]:
                writer.add_page(page)
            buffer = io.BytesIO()
            writer.write(buffer)
            return buffer.getvalue()
        low, high = start + 1, min(count, start + max_pages)
        chosen, payload = None, None
        while low <= high:
            mid = (low + high) // 2
            candidate = render(mid)
            if len(candidate) <= MAX_FILE:
                chosen, payload, low = mid, candidate, mid + 1
            else:
                high = mid - 1
        if chosen is None:
            raise ValueError('a PDF page exceeds MinerU file size limit')
        target = spool / f'{source_sha}-p{start+1:05d}-{chosen:05d}.pdf'
        if not target.exists() or digest_file(target) != hashlib.sha256(payload).hexdigest():
            # mkstemp + explicit close: Windows cannot os.replace an open file.
            fd, name = tempfile.mkstemp(dir=spool)
            temporary = Path(name)
            try:
                with os.fdopen(fd, 'wb') as f:
                    f.write(payload)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        output.append((target, (start+1, chosen)))
        start = chosen
    return output


class OcrBackend(Protocol):
    """Queue-facing async document OCR contract (other providers need a shim)."""
    config: MinerUConfig
    download_hosts: frozenset[str]
    def reserve(self, name: str, digest: str) -> tuple[str, str]: ...
    def upload(self, url: str, path: Path) -> None: ...
    def poll(self, batch_id: str, data_id: str) -> dict: ...
    def download(self, url: str, path: Path) -> None: ...


class MinerUBackend:
    """Single-file reservations; no unbounded retries or hidden external fallback."""
    def __init__(self, token, config=MinerUConfig(), *, client=None,
                 min_interval=0.2, retries=2, upload_hosts=UPLOAD_HOSTS,
                 download_hosts=DOWNLOAD_HOSTS, max_zip_bytes=MAX_ZIP):
        if not token or not token.strip() or not isinstance(config, MinerUConfig):
            raise ValueError('MINERU_TOKEN and MinerUConfig are required')
        if not math.isfinite(min_interval) or min_interval < 0 or type(retries) is not int or not 0 <= retries <= 5:
            raise ValueError('invalid rate/retry settings')
        self.token, self.config = token, config
        self.min_interval, self.retries = min_interval, retries
        if not isinstance(max_zip_bytes, int) or not 0 < max_zip_bytes <= MAX_ZIP:
            raise ValueError('invalid zip limit')
        self.max_zip_bytes = max_zip_bytes
        self.upload_hosts, self.download_hosts = frozenset(upload_hosts), frozenset(download_hosts)
        self.client = client or httpx.Client(timeout=httpx.Timeout(30, connect=5), follow_redirects=False)
        self.owns_client = client is None
        self._next_call = 0.
        self.deadline = None  # Request-scoped instances only.

    def _remaining(self, cap):
        if self.deadline is None:
            return cap
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise MinerUError('deadline')
        return min(cap, left)

    def _sleep(self, seconds):
        time.sleep(self._remaining(seconds))
        self._remaining(1)

    def close(self):
        if self.owns_client:
            self.client.close()

    def _rate(self):
        wait = self._next_call - time.monotonic()
        if wait > 0:
            self._sleep(wait)
        self._next_call = time.monotonic() + self.min_interval

    def _json(self, method, path, *, payload=None, repeat=False):
        headers = {'Authorization': 'Bearer ' + self.token, 'Accept': 'application/json'}
        attempts = (self.retries + 1) if repeat else 1
        for i in range(attempts):
            self._rate()
            try:
                with self.client.stream(method, BASE + path, headers=headers, json=payload,
                                        timeout=httpx.Timeout(self._remaining(30),
                                                              connect=self._remaining(5),
                                                              read=self._remaining(3))) as response:
                    status = response.status_code
                    if status == 200:
                        content = bytearray()
                        for chunk in response.iter_bytes(65536):
                            self._remaining(1)
                            content.extend(chunk)
                            if len(content) > 2_000_000:
                                raise MinerUError('invalid_api_response')
            except httpx.RequestError:
                if repeat and i + 1 < attempts:
                    self._sleep(min(2**i, 4))
                    continue
                raise MinerUError('network_failure', retryable=repeat) from None
            if status in (429, 500, 502, 503, 504):
                if repeat and i + 1 < attempts:
                    self._sleep(min(2**i, 4))
                    continue
                raise MinerUError('upstream_busy', retryable=repeat)
            if status != 200:
                raise MinerUError('invalid_api_response')
            try:
                body = json.loads(content)
                if not isinstance(body, dict) or type(body['code']) is not int:
                    raise ValueError()
            except (ValueError, KeyError, TypeError):
                raise MinerUError('invalid_api_response') from None
            if body['code'] != 0:
                code = body['code']
                if code == -60018:
                    raise MinerUError('daily_quota')
                if code == -60009:
                    raise MinerUError('queue_full', retryable=True)
                if code == -60010:
                    raise MinerUError('extract_failed')
                raise MinerUError('upstream_rejected')
            if not isinstance(body.get('data'), dict):
                raise MinerUError('invalid_api_response')
            return body['data']
        raise MinerUError('network_failure')

    def reserve(self, name, digest):
        if (Path(name).name != name or not re.fullmatch(r'[a-f0-9]{64}', digest)
                or Path(name).suffix.lower() not in SOURCE_EXTENSIONS):
            raise ValueError('invalid file name or digest')
        data = self._json('POST', '/file-urls/batch', payload=self.config.request(name, digest))
        batch = data.get('batch_id')
        urls = data.get('file_urls')
        if (not isinstance(batch, str) or not re.fullmatch(r'[\w-]{1,128}', batch)
                or not isinstance(urls, list) or len(urls) != 1):
            raise MinerUError('invalid_api_response')
        return batch, safe_url(urls[0], self.upload_hosts)

    def upload(self, url, path):
        safe_url(url, self.upload_hosts)
        path = check_file(path)
        # Never send our bearer token or content-type to a signed OSS URL.
        for attempt in range(self.retries + 1):
            try:
                with path.open('rb') as stream:
                    response = self.client.put(url, content=stream,
                        timeout=httpx.Timeout(self._remaining(120), connect=self._remaining(5),
                                              write=self._remaining(3)), follow_redirects=False)
            except httpx.RequestError:
                response = None
            if response is not None and response.status_code == 200:
                return
            if response is not None and response.status_code not in (429, 500, 502, 503, 504):
                raise MinerUError('upload_failed')
            if attempt < self.retries:
                self._sleep(min(2**attempt, 4))
        raise MinerUError('upload_retry_later', retryable=True)

    def poll(self, batch_id, data_id):
        if not re.fullmatch(r'[\w-]{1,128}', batch_id):
            raise ValueError('invalid batch ID')
        data = self._json('GET', '/extract-results/batch/' + batch_id, repeat=True)
        return match_result(data, batch_id, data_id, hosts=self.download_hosts)

    def download(self, url, path):
        safe_url(url, self.download_hosts)
        path = Path(path)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # mkstemp + explicit close: Windows cannot os.replace an open file.
        fd, name = tempfile.mkstemp(dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, 'wb') as f:
                with self.client.stream('GET', url, timeout=httpx.Timeout(self._remaining(120),
                                        connect=self._remaining(5), read=self._remaining(3)),
                                        follow_redirects=False) as response:
                    if response.status_code != 200:
                        raise MinerUError('zip_download_failed', retryable=response.status_code >= 500)
                    total = 0
                    for block in response.iter_bytes(65536):
                        total += len(block)
                        self._remaining(1)
                        if total > self.max_zip_bytes:
                            raise MinerUError('zip_too_large')
                        f.write(block)
            if total == 0:
                raise MinerUError('invalid_zip')
            os.replace(temporary, path)
        except httpx.RequestError:
            raise MinerUError('zip_download_failed', retryable=True) from None
        finally:
            temporary.unlink(missing_ok=True)


def match_result(data, batch_id, data_id, *, hosts=DOWNLOAD_HOSTS):
    if not isinstance(data, dict) or data.get('batch_id') != batch_id:
        raise MinerUError('result_batch_mismatch')
    rows = data.get('extract_result')
    if not isinstance(rows, list):
        raise MinerUError('invalid_api_response')
    matches = [r for r in rows if isinstance(r, dict) and r.get('data_id') == data_id]
    if len(matches) != 1 or matches[0].get('state') not in STATES:
        raise MinerUError('result_data_mismatch')
    row = matches[0]
    if row['state'] == 'done':
        safe_url(row.get('full_zip_url'), hosts)
    return row


def verify_callback(payload, *, uid, seed):
    """Signature is SHA256(uid + seed + the exact content string), not re-JSON."""
    if (not isinstance(payload, dict) or set(payload) != {'content', 'checksum'}
            or not isinstance(payload['content'], str) or len(payload['content'].encode()) > 65536
            or not isinstance(payload['checksum'], str) or not re.fullmatch(r'[a-f0-9]{64}', payload['checksum'])):
        raise MinerUError('invalid_callback')
    expected = hashlib.sha256((uid + seed + payload['content']).encode('utf-8')).hexdigest()
    if not hmac.compare_digest(expected, payload['checksum']):
        raise MinerUError('invalid_callback')
    try:
        return json.loads(payload['content'])
    except (TypeError, ValueError):
        raise MinerUError('invalid_callback') from None


def normalize_ru(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().replace('ё', 'е').split())


def unpack_result(archive, directory, *, data_id, config_sha):
    """Keep the zip, extract only bounded, recognised files. No extractall."""
    archive, directory = Path(archive), Path(directory)
    extracted = {}
    total = 0
    try:
        with zipfile.ZipFile(archive) as z:
            if len(z.infolist()) > 10000:
                raise MinerUError('unsafe_zip')
            for info in z.infolist():
                name = info.filename
                parts = name.split('/')
                mode = info.external_attr >> 16
                total += info.file_size
                if (name.startswith('/') or '\\' in name or any(p in ('', '.', '..') for p in parts if not info.is_dir())
                        or stat.S_ISLNK(mode) or total > MAX_UNPACKED):
                    raise MinerUError('unsafe_zip')
                if info.is_dir():
                    continue
                basename = parts[-1]
                key = ('full.md' if basename == 'full.md' else
                       'content_list.json' if basename.endswith('_content_list.json') else
                       'layout.json' if basename == 'layout.json' else
                       'model.json' if basename.endswith('_model.json') else None)
                if key:
                    if key in extracted or info.file_size > 64_000_000 or (info.file_size and not info.compress_size):
                        raise MinerUError('unsafe_zip')
                    with z.open(info) as stream:
                        content = stream.read(64_000_001)
                    if len(content) != info.file_size or len(content) > 64_000_000:
                        raise MinerUError('unsafe_zip')
                    extracted[key] = content
    except (zipfile.BadZipFile, OSError, EOFError, RuntimeError):
        raise MinerUError('invalid_zip') from None
    if 'full.md' not in extracted or 'content_list.json' not in extracted:
        raise MinerUError('missing_zip_outputs')
    try:
        text = extracted['full.md'].decode('utf-8')
        blocks = json.loads(extracted['content_list.json'])
        if not isinstance(blocks, list):
            raise ValueError()
        for k in ('layout.json', 'model.json'):
            if k in extracted:
                json.loads(extracted[k])
    except (UnicodeError, ValueError):
        raise MinerUError('invalid_zip_outputs') from None
    if directory.exists():
        try:
            previous = json.loads((directory / 'qa.json').read_text(encoding='utf-8'))
            if (previous['data_id'] == data_id and previous['config_sha256'] == config_sha
                    and previous['zip_sha256'] == digest_file(archive)):
                return previous  # crash after atomic rename, before DB commit
        except (OSError, ValueError, KeyError):
            pass
        raise MinerUError('result_already_exists')
    directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mineru-', dir=directory.parent) as temp:
        root = Path(temp)
        for name, content in extracted.items():
            (root / name).write_bytes(content)
        (root / 'normalized.txt').write_text(normalize_ru(text), encoding='utf-8')
        metadata = {'data_id': data_id, 'config_sha256': config_sha,
                    'zip_sha256': digest_file(archive), 'empty_text': not bool(normalize_ru(text)),
                    'empty_structure': not bool(blocks),
                    'outputs': sorted(extracted)}
        (root / 'qa.json').write_text(json.dumps(metadata, sort_keys=True), encoding='utf-8')
        os.replace(root, directory)
    return metadata
