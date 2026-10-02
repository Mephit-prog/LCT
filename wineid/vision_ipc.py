"""Single persistent, killable CLIP encoder for HTTP and offline fusion runs.

Only image encoding runs in the child; verified index scoring remains in the
parent. The child is started at bootstrap, never on a user request. Failed
workers recover asynchronously and cannot be shared between API processes.
"""
import base64
import concurrent.futures
import io
import json
import os
import signal
if os.name != 'nt':
    import select
import subprocess
import sys
import threading
import time
from PIL import Image

from .jina_clip import unit_vectors

MAX_REQUEST = 128 * 1024 * 1024  # max three lossless RGB views, including base64
MAX_VISION_PIXELS = 6_000_000
MAX_REPLY = 256 * 1024
RECOVERY_DELAY = 1.0


class VisionWorkerError(RuntimeError):
    """Safe, typed failure; do not expose subprocess stderr to HTTP."""


def _windows_io(fn, deadline):
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn)
    try:
        return future.result(timeout=max(0, deadline - time.monotonic()))
    except concurrent.futures.TimeoutError:
        raise TimeoutError('deadline') from None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _write(fd, data, deadline):
    if os.name == 'nt':
        return _windows_io(lambda: _write_blocking(fd, data), deadline)
    return _write_posix(fd, data, deadline)


def _write_blocking(fd, data):
    data = memoryview(data)
    while data:
        written = os.write(fd, data[:65536])
        if written <= 0:
            raise OSError('pipe closed')
        data = data[written:]


def _write_posix(fd, data, deadline):
    offset = 0
    data = memoryview(data)
    while offset < len(data):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
            raise TimeoutError('deadline')
        try:
            offset += os.write(fd, data[offset:offset + 65536])
        except BlockingIOError:
            continue


def _read(fd, deadline, cancelled=None):
    if os.name == 'nt':
        return _windows_io(lambda: _read_blocking(fd), deadline)
    return _read_posix(fd, deadline, cancelled)


def _read_blocking(fd):
    buf = bytearray()
    while True:
        chunk = os.read(fd, min(65536, MAX_REPLY + 1 - len(buf)))
        if not chunk or len(buf) + len(chunk) > MAX_REPLY:
            raise VisionWorkerError('vision_worker_failed')
        buf.extend(chunk)
        if b'\n' in buf:
            line, tail = bytes(buf).split(b'\n', 1)
            if tail:
                raise VisionWorkerError('vision_worker_failed')
            try:
                return json.loads(line)
            except (ValueError, UnicodeError) as exc:
                raise VisionWorkerError('vision_worker_failed') from exc


def _read_posix(fd, deadline, cancelled=None):
    buf = bytearray()
    while True:
        if cancelled is not None and cancelled.is_set():
            raise VisionWorkerError('vision_worker_unavailable')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('deadline')
        if not select.select([fd], [], [], min(remaining, .1) if cancelled else remaining)[0]:
            continue
        chunk = os.read(fd, min(65536, MAX_REPLY + 1 - len(buf)))
        if not chunk:
            raise VisionWorkerError('vision_worker_failed')
        buf.extend(chunk)
        if len(buf) > MAX_REPLY:
            raise VisionWorkerError('vision_worker_failed')
        if b'\n' in buf:
            line, tail = bytes(buf).split(b'\n', 1)
            if tail:
                raise VisionWorkerError('vision_worker_failed')
            try:
                return json.loads(line)
            except (UnicodeError, ValueError) as exc:
                raise VisionWorkerError('vision_worker_failed') from exc


def _kill_tree(p):
    if os.name == 'nt':
        try:
            subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            if p.poll() is None:
                p.kill()
    else:
        os.killpg(p.pid, signal.SIGKILL)


class IsolatedVisionEncoder:
    def __init__(self, *, adapter_path=None, device=None, startup_timeout=120,
                 command=None, expected_config=None):
        if not 0 < startup_timeout <= 600:
            raise ValueError('invalid vision startup timeout')
        self.command = command or [sys.executable, '-m', 'wineid.vision_worker',
                                   str(adapter_path or ''), str(device or '')]
        self.env = os.environ.copy()
        for secret in ('WINE_TOKEN', 'MISTRAL_API_KEY', 'MINERU_TOKEN',
                       'HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN'):
            self.env.pop(secret, None)
        # Fail at startup/recovery rather than downloading unreviewed code or
        # weights during a request. Provision the pinned HF cache in advance.
        self.env['HF_HUB_OFFLINE'] = '1'
        self.env['TRANSFORMERS_OFFLINE'] = '1'
        from pathlib import Path
        cache = Path('artifacts/hf-cache').resolve()
        if cache.is_dir():
            self.env['HF_HOME'] = str(cache)
        self.startup_timeout = startup_timeout
        self.expected_config = expected_config
        self.config = None
        self.device = None
        self._process = None
        self._verified = False
        self._owner = os.getpid()
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._recovery = None

    @property
    def ready(self):
        p = self._process
        return (self._verified and not self._closed.is_set() and os.getpid() == self._owner
                and p is not None and p.poll() is None)

    @property
    def model(self):
        # VisionSources.ready expects an encoder.model sentinel.
        return True if self.ready else None

    def _stop(self):
        p, self._process = self._process, None
        self._verified = False
        if p is None or os.getpid() != self._owner:
            return
        try:
            _kill_tree(p)
        except (ProcessLookupError, FileNotFoundError):
            pass
        except PermissionError:
            if p.poll() is None:
                p.kill()
        try:
            p.wait(timeout=.05)
        except subprocess.TimeoutExpired:
            # A stuck kernel wait must not consume the request budget.
            threading.Thread(target=p.wait, daemon=True).start()
        p.stdin.close()
        p.stdout.close()

    def _spawn(self):
        if self._closed.is_set() or os.getpid() != self._owner:
            raise VisionWorkerError('vision_worker_unavailable')
        self._process = subprocess.Popen(self.command, env=self.env, stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         bufsize=0, start_new_session=True,
                                         creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0))
        try:
            if os.name != 'nt':
                os.set_blocking(self._process.stdin.fileno(), False)
                os.set_blocking(self._process.stdout.fileno(), False)
            reply = _read(self._process.stdout.fileno(), time.monotonic() + self.startup_timeout,
                          self._closed)
            if (not isinstance(reply, dict) or set(reply) != {'ready', 'config', 'device'}
                    or reply['ready'] is not True or not isinstance(reply['config'], dict)
                    or not isinstance(reply['device'], str)
                    or (self.expected_config is not None and reply['config'] != self.expected_config)
                    or (self.config is not None and reply['config'] != self.config)
                    or (self.device is not None and reply['device'] != self.device)):
                raise VisionWorkerError('vision_worker_incompatible')
            self.config, self.device = reply['config'], reply['device']
            self._verified = True
        except BaseException:
            self._stop()
            raise

    def start(self):
        """Bootstrap/canary only, outside the request deadline."""
        with self._lock:
            if not self.ready:
                self._stop()
                self._spawn()

    def _recover(self):
        # One recovery at a time, with bounded retries and backoff. The next
        # request during recovery fails fast; it never loads weights itself.
        for attempt in range(3):
            if self._closed.wait(RECOVERY_DELAY * (attempt + 1)):
                return
            with self._lock:
                if self._closed.is_set() or self.ready:
                    return
                try:
                    self._stop()
                    self._spawn()
                    return
                except (OSError, TimeoutError, ValueError, VisionWorkerError):
                    pass

    def _schedule_recovery(self):
        if not self._closed.is_set() and (self._recovery is None or not self._recovery.is_alive()):
            self._recovery = threading.Thread(target=self._recover, daemon=True)
            self._recovery.start()

    def encode_images_deadline(self, images, deadline):
        if not self._lock.acquire(blocking=False):
            raise VisionWorkerError('vision_capacity_exceeded')
        try:
            if not self.ready:
                self._stop()
                self._schedule_recovery()
                raise VisionWorkerError('vision_worker_unavailable')
            if time.monotonic() >= deadline:
                raise TimeoutError('deadline')
            payload = []
            size = 32  # JSON framing overhead
            for image in images:
                if time.monotonic() >= deadline:
                    raise TimeoutError('deadline')
                if len(images) > 3:
                    raise VisionWorkerError('vision_input_too_large')
                if image.width * image.height > MAX_VISION_PIXELS:
                    # Jina resizes to 512 px anyway: send a downscaled copy of a
                    # phone-sized frame instead of refusing it.
                    scale = (MAX_VISION_PIXELS / (image.width * image.height)) ** 0.5
                    image = image.resize((max(1, int(image.width * scale)),
                                          max(1, int(image.height * scale))), Image.BICUBIC)
                output = io.BytesIO()
                image.save(output, format='PNG')  # lossless; same RGB pixels as in-process Jina
                if time.monotonic() >= deadline:
                    raise TimeoutError('deadline')
                if output.tell() > MAX_REQUEST // 2:
                    raise VisionWorkerError('vision_input_too_large')
                encoded = base64.b64encode(output.getvalue()).decode('ascii')
                size += len(encoded) + 4
                if size > MAX_REQUEST:
                    raise VisionWorkerError('vision_input_too_large')
                payload.append(encoded)
            if time.monotonic() >= deadline:
                raise TimeoutError('deadline')
            request = (json.dumps({'images': payload}, separators=(',', ':')) + '\n').encode('ascii')
            try:
                p = self._process
                _write(p.stdin.fileno(), request, deadline)
                reply = _read(p.stdout.fileno(), deadline, self._closed)
                if time.monotonic() >= deadline:
                    raise TimeoutError('deadline')
                if not isinstance(reply, dict) or set(reply) != {'vectors'}:
                    raise VisionWorkerError('vision_worker_failed')
                return unit_vectors(reply['vectors'], len(images), self.config['dimensions'])
            except TimeoutError:
                self._stop()
                self._schedule_recovery()
                raise
            except (OSError, ValueError, KeyError, TypeError, VisionWorkerError) as exc:
                self._stop()
                self._schedule_recovery()
                raise VisionWorkerError('vision_worker_failed') from exc
        finally:
            self._lock.release()

    def close(self):
        self._closed.set()
        # Wake a blocked inference/slow canary even when it holds _lock.
        p = self._process
        if p is not None and os.getpid() == self._owner:
            try:
                _kill_tree(p)
            except (ProcessLookupError, PermissionError, FileNotFoundError):
                pass
        with self._lock:
            self._stop()
        recovery = self._recovery
        if recovery is not None and recovery is not threading.current_thread():
            recovery.join(timeout=.1)
