"""Single-slot, deadline-bound local Paddle worker over private pipes.

One instance per API process. Do not use multiple uvicorn workers with a GPU model;
process-wide admission control across independent API processes is NOT provided.
"""
import base64
import hashlib
import json
import os
import select
import subprocess
import sys
import threading
from pathlib import Path
import time

from .catalog import normalize
from .ipc_util import kill_tree, with_deadline
from .ocr import OCRResult
from .roi import MAX_CROP_BYTES

MAX_REPLY = 2 * 1024 * 1024
RECOVERY_DELAY = 1.0
RECOVERY_COOLDOWN = 30.0


class WorkerFailure(Exception):
    pass


def verify_manifest(path, det_dir, rec_dir):
    """Check *every* byte of both read-only model trees before loading weights."""
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    if (manifest.get('format') != 'wineid-paddle-v1' or
            manifest.get('ocr_version') != 'PP-OCRv5' or manifest.get('lang') != 'ru' or
            not isinstance(manifest.get('packages'), dict) or
            set(manifest['packages']) not in ({'paddleocr', 'paddlepaddle'},
                                             {'paddleocr', 'paddlepaddle-gpu'}) or
            not all(isinstance(v, str) and v for v in manifest['packages'].values())):
        raise ValueError('incompatible OCR manifest')
    for key, root in [('det', det_dir), ('rec', rec_dir)]:
        directory = Path(root)
        expected = manifest.get(key)
        if directory.is_symlink() or not directory.is_dir() or not isinstance(expected, dict) or not expected:
            raise ValueError('missing OCR model files')
        actual = {p.relative_to(directory).as_posix(): p for p in directory.rglob('*') if p.is_file()}
        if set(actual) != set(expected) or any(p.is_symlink() for p in directory.rglob('*')):
            raise ValueError('OCR model manifest file list mismatch')
        for name, file in actual.items():
            digest = hashlib.sha256()
            with file.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != expected[name]:
                raise ValueError('OCR model hash mismatch')
    return manifest, hashlib.sha256(Path(path).read_bytes()).hexdigest()


class WarmPaddleOCR:
    model = 'PP-OCRv5-ru-local'

    def __init__(self, *, python=None, det_dir=None, rec_dir=None, manifest=None,
                 startup_timeout=120, command=None):
        # command is only for offline IPC contract tests; normal startup always
        # uses the private worker module with explicit local weights.
        self.command = command or [python or sys.executable, '-m', 'wineid.warm_ocr_worker']
        self.env = os.environ.copy()
        if det_dir is not None:
            self.env['WINE_PADDLE_DET_DIR'] = det_dir
        if rec_dir is not None:
            self.env['WINE_PADDLE_REC_DIR'] = rec_dir
        self.manifest = manifest
        if command is None and (not manifest or not all(self.env.get(k) for k in
                               ('WINE_PADDLE_DET_DIR', 'WINE_PADDLE_REC_DIR'))):
            raise ValueError('warm Paddle requires a model manifest and two local model directories')
        self.env['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
        self.env['HF_HUB_OFFLINE'] = '1'
        self.startup_timeout = float(startup_timeout)
        if not 0 < self.startup_timeout <= 600:
            raise ValueError('invalid OCR startup timeout')
        self._lock = threading.Lock()
        self._process = None
        self._manifest_digest = None
        self._owner = os.getpid()
        self._closed = threading.Event()
        self._recovery = None
        self._retry_after = 0.0
        self._incompatible = False

    @property
    def ready(self):
        p = self._process
        return (not self._closed.is_set() and not self._incompatible
                and os.getpid() == self._owner and p is not None and p.poll() is None)

    def _stop(self):
        p, self._process = self._process, None
        if p is None:
            return
        # Never kill a worker inherited across fork from another API process.
        if os.getpid() != self._owner:
            return
        # Descendants may keep pipes open even after the leader exits.
        try:
            kill_tree(p)
        except (ProcessLookupError, PermissionError, FileNotFoundError):
            if p.poll() is None:
                p.kill()
        try:
            p.wait(timeout=.05)
        except subprocess.TimeoutExpired:
            # Do not consume the request deadline waiting on a stuck kernel wait.
            threading.Thread(target=p.wait, daemon=True).start()
        p.stdin.close()
        p.stdout.close()

    def close(self):
        self._closed.set()
        # Interrupt a blocked read/canary before acquiring the inference lock.
        p = self._process
        if p is not None and os.getpid() == self._owner:
            try:
                kill_tree(p)
            except (ProcessLookupError, PermissionError, FileNotFoundError):
                pass
        with self._lock:
            self._stop()
        recovery = self._recovery
        if recovery is not None and recovery is not threading.current_thread():
            recovery.join(timeout=.1)

    @staticmethod
    def _write_blocking(p, payload, cancelled):
        fd = p.stdin.fileno()
        view = memoryview(payload)
        while view:
            if cancelled.is_set():
                raise WorkerFailure('worker closed')
            try:
                sent = os.write(fd, view[:65536])
            except BrokenPipeError as exc:
                raise WorkerFailure('worker exited') from exc
            if sent <= 0:
                raise WorkerFailure('worker exited')
            view = view[sent:]

    @staticmethod
    def _read_blocking(p, cancelled):
        fd = p.stdout.fileno()
        buf = bytearray()
        while True:
            if cancelled.is_set():
                raise WorkerFailure('worker closed')
            try:
                chunk = os.read(fd, min(65536, MAX_REPLY + 1 - len(buf)))
            except OSError as exc:
                raise WorkerFailure('worker exited') from exc
            if not chunk:
                raise WorkerFailure('worker exited')
            buf.extend(chunk)
            if len(buf) > MAX_REPLY:
                raise WorkerFailure('reply too large')
            if b'\n' in buf:
                line, tail = bytes(buf).split(b'\n', 1)
                if tail:
                    raise WorkerFailure('multiple replies')
                try:
                    return json.loads(line)
                except (UnicodeError, ValueError) as exc:
                    raise WorkerFailure('invalid reply') from exc

    @staticmethod
    def _write(p, payload, deadline, cancelled):
        if os.name == 'nt':
            return with_deadline(lambda: WarmPaddleOCR._write_blocking(p, payload, cancelled),
                                deadline)
        fd = p.stdin.fileno()
        sent = 0
        while sent < len(payload):
            if cancelled.is_set():
                raise WorkerFailure('worker closed')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('deadline')
            if not select.select([], [fd], [], min(remaining, .1))[1]:
                continue
            try:
                sent += os.write(fd, payload[sent:sent + 65536])
            except BlockingIOError:
                continue
            except BrokenPipeError as exc:
                raise WorkerFailure('worker exited') from exc

    @staticmethod
    def _read(p, deadline, cancelled):
        if os.name == 'nt':
            return with_deadline(lambda: WarmPaddleOCR._read_blocking(p, cancelled), deadline)
        fd = p.stdout.fileno()
        buf = bytearray()
        while True:
            if cancelled.is_set():
                raise WorkerFailure('worker closed')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('deadline')
            if not select.select([fd], [], [], min(remaining, .1))[0]:
                continue
            try:
                chunk = os.read(fd, min(65536, MAX_REPLY + 1 - len(buf)))
            except BlockingIOError:
                continue
            if not chunk:
                raise WorkerFailure('worker exited')
            buf.extend(chunk)
            if len(buf) > MAX_REPLY:
                raise WorkerFailure('reply too large')
            if b'\n' in buf:
                line, tail = bytes(buf).split(b'\n', 1)
                if tail:
                    raise WorkerFailure('multiple replies')
                try:
                    return json.loads(line)
                except (UnicodeError, ValueError) as exc:
                    raise WorkerFailure('invalid reply') from exc

    def _verify(self):
        if self.manifest:
            try:
                manifest, digest = verify_manifest(self.manifest, self.env['WINE_PADDLE_DET_DIR'],
                                                   self.env['WINE_PADDLE_REC_DIR'])
                if self._manifest_digest is not None and digest != self._manifest_digest:
                    raise ValueError('model manifest changed during process lifetime')
            except (ValueError, OSError, KeyError):
                # Never restart with different or unverifiable weights. Requires
                # operator intervention and a fresh runtime, not retry traffic.
                self._incompatible = True
                raise
            self._manifest_digest = digest
            self.env['WINE_OCR_EXPECTED_PACKAGES'] = json.dumps(manifest['packages'])
            self.model = 'PP-OCRv5-ru-local:' + digest

    def _spawn(self, deadline):
        if self._closed.is_set() or self._incompatible or os.getpid() != self._owner:
            raise WorkerFailure('worker unavailable')
        self._verify()  # bootstrap or background recovery only
        if self._closed.is_set():
            raise WorkerFailure('worker closed')
        self._process = subprocess.Popen(self.command, env=self.env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=0, start_new_session=True,
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0))
        if os.name != 'nt':
            os.set_blocking(self._process.stdin.fileno(), False)
            os.set_blocking(self._process.stdout.fileno(), False)
        if self._read(self._process, deadline, self._closed) != {'ready': True}:
            raise WorkerFailure('startup canary failed')

    def _recover(self):
        for attempt in range(3):
            if self._closed.wait(RECOVERY_DELAY * (attempt + 1)):
                return
            with self._lock:
                if self._closed.is_set() or self._incompatible or self.ready:
                    return
                try:
                    self._stop()
                    self._spawn(time.monotonic() + self.startup_timeout)
                    return
                except (OSError, TimeoutError, ValueError, KeyError, WorkerFailure):
                    self._stop()
                    if self._incompatible:
                        return
        self._retry_after = time.monotonic() + RECOVERY_COOLDOWN

    def _schedule_recovery(self):
        if (not self._closed.is_set() and not self._incompatible
                and os.getpid() == self._owner and time.monotonic() >= self._retry_after
                and (self._recovery is None or not self._recovery.is_alive())):
            self._recovery = threading.Thread(target=self._recover, daemon=True)
            self._recovery.start()

    def start(self):
        """Load weights and complete one canary inference before serving HTTP."""
        with self._lock:
            if self.ready:
                return
            if os.getpid() != self._owner:
                raise RuntimeError('OCR worker cannot be shared across forked API processes')
            self._stop()
            try:
                self._spawn(time.monotonic() + self.startup_timeout)
            except (OSError, TimeoutError, WorkerFailure, ValueError, KeyError):
                self._stop()
                raise RuntimeError('OCR worker failed to start') from None

    def recognize(self, roi, deadline):
        start = time.monotonic()
        if time.monotonic() >= deadline:
            return OCRResult(roi.roi_id, 'timeout', None, None, self.model, 0, 'deadline')
        if not self._lock.acquire(blocking=False):
            return OCRResult(roi.roi_id, 'provider_error', None, None, self.model, 0, 'busy')
        try:
            status, code, raw = 'provider_error', None, None
            restart = False
            if not self.ready:
                # Hashing weights and loading the model never run in a request.
                self._schedule_recovery()
                code = 'worker_unavailable'
            else:
                try:
                    if len(roi.crop) > MAX_CROP_BYTES:
                        raise WorkerFailure('crop too large')
                    request = (json.dumps({'op': 'recognize', 'image':
                               base64.b64encode(roi.crop).decode('ascii')}) + '\n').encode('utf-8')
                    self._write(self._process, request, deadline, self._closed)
                    reply = self._read(self._process, deadline, self._closed)
                    if not isinstance(reply, dict) or set(reply) not in ({'text'}, {'error'}):
                        raise WorkerFailure('invalid reply')
                    if 'error' in reply:
                        code = 'inference_failed'
                        restart = True
                    elif not isinstance(reply['text'], str):
                        raise WorkerFailure('invalid reply')
                    else:
                        raw = reply['text']
                        status = 'ok' if normalize(raw) else 'no_text'
                    if time.monotonic() >= deadline:
                        raise TimeoutError('deadline')
                except TimeoutError:
                    status, code, raw = 'timeout', 'deadline', None
                    restart = True
                except (OSError, WorkerFailure, ValueError, KeyError):
                    status, code, raw = 'provider_error', 'worker_failed', None
                    restart = True
                if restart:
                    self._stop()
                    self._schedule_recovery()
            return OCRResult(roi.roi_id, status, raw,
                             normalize(raw) if raw is not None else None,
                             self.model, round((time.monotonic()-start)*1000), code)
        finally:
            self._lock.release()
