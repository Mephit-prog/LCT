"""Opt-in local OCR backends. Heavy engines run in an isolated, killable process.

This per-request worker is a correctness baseline, NOT a production warm model pool.
Never send input bytes or model stderr to application logs.
"""
import io
import os
import select
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from PIL import Image

from .catalog import normalize
from .ipc_util import kill_tree
from .ocr import OCRResult

SUPPORTED = ('tesseract', 'paddle', 'mineru')
MAX_OUTPUT = 2 * 1024 * 1024


class LocalFailure(OSError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


class LocalOCR:
    def __init__(self, backend: str, *, python: str | None = None,
                 tesseract: str = 'tesseract', tessdata_dir: str | None = None):
        if backend not in SUPPORTED:
            raise ValueError('unsupported OCR backend')
        self.backend = backend
        self.model = {'tesseract': 'tesseract-rus+eng', 'paddle': 'PP-OCRv5-ru',
                      'mineru': 'mineru-pipeline-cyrillic'}[backend]
        self.python = python or sys.executable
        self.tesseract = tesseract
        self.tessdata_dir = tessdata_dir
        self._ready = False

    @property
    def ready(self):
        return self._ready

    def start(self):
        """Real local canary; construction alone never implies readiness."""
        from PIL import ImageDraw
        from .roi import full_frame_roi
        buf = io.BytesIO()
        image = Image.new('RGB', (320, 80), 'white')
        ImageDraw.Draw(image).text((10, 10), 'WINE 2024', fill='black')
        image.save(buf, 'PNG')
        deadline = time.monotonic() + 120  # cold baseline; request deadlines stay short
        result = self.recognize(full_frame_roi(buf.getvalue(), deadline), deadline)
        if result.status not in ('ok', 'no_text'):
            raise RuntimeError('local OCR canary failed: ' + (result.error_code or result.status))
        self._ready = True

    def recognize(self, roi, deadline: float) -> OCRResult:
        start = time.monotonic()
        status, raw, code = 'provider_error', None, None
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(self.backend, 0)
            # roi.crop is already an oriented, metadata-free bounded PNG from roi.py.
            # Re-decoding and PNG-encoding it here can consume the request budget
            # synchronously before we even start the killable OCR process.
            if self.backend == 'tesseract':
                # stdin works even on hosts where leptonica cannot reopen a named file.
                cmd = [self.tesseract, 'stdin', 'stdout']
                if self.tessdata_dir:
                    cmd += ['--tessdata-dir', self.tessdata_dir]
                cmd += ['-l', 'rus+eng', '--psm', '3']
                raw = _execute(cmd, roi.crop, deadline)
            else:
                # No persistent ROI files; worker writes only inside this temporary dir.
                with tempfile.TemporaryDirectory(prefix='wine-ocr-') as tmp:
                    path = Path(tmp) / 'roi.png'
                    path.write_bytes(roi.crop)
                    if time.monotonic() >= deadline:
                        raise subprocess.TimeoutExpired(self.backend, 0)
                    cmd = [self.python, '-m', 'wineid.local_ocr_worker', self.backend,
                           str(path), str(Path(tmp) / 'output')]
                    raw = _execute(cmd, None, deadline)
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(self.backend, 0)
            status = 'ok' if normalize(raw) else 'no_text'
        except subprocess.TimeoutExpired:
            status, code = 'timeout', 'deadline'
        except FileNotFoundError:
            code = 'engine_missing'
        except LocalFailure as exc:
            code = exc.code
        except (OSError, ValueError, UnicodeError):
            code = 'local_ocr_error'
        return OCRResult(roi.roi_id, status, raw, normalize(raw) if raw is not None else None,
                         self.model, round((time.monotonic() - start) * 1000), code)


def _execute(cmd, payload, deadline, cwd=None):
    """Kill the whole process group on timeout, including MinerU descendants."""
    if os.name == 'nt':
        return _execute_windows(cmd, payload, deadline, cwd)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired(cmd, 0)
    p = subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.PIPE if payload is not None else subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, start_new_session=True)
    out = bytearray()
    sent = 0
    try:
        if payload is not None:
            os.set_blocking(p.stdin.fileno(), False)
            if not payload:
                p.stdin.close()
        os.set_blocking(p.stdout.fileno(), False)
        reading = True
        while reading or (payload is not None and sent < len(payload)):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(cmd, remaining)
            readers = [p.stdout] if reading else []
            writers = [p.stdin] if payload is not None and sent < len(payload) else []
            r, w, _ = select.select(readers, writers, [], remaining)
            if not r and not w:
                raise subprocess.TimeoutExpired(cmd, remaining)
            if w:
                try:
                    sent += os.write(p.stdin.fileno(), payload[sent:sent + 65536])
                except BlockingIOError:
                    pass
                except BrokenPipeError:
                    sent = len(payload)
                if sent == len(payload):
                    p.stdin.close()
            if r:
                try:
                    chunk = os.read(p.stdout.fileno(), min(65536, MAX_OUTPUT + 1 - len(out)))
                except BlockingIOError:
                    continue
                if chunk:
                    out.extend(chunk)
                    if len(out) > MAX_OUTPUT:
                        raise LocalFailure('output_too_large')
                else:
                    reading = False
        p.wait(timeout=max(0, deadline - time.monotonic()))
        if p.returncode:
            raise LocalFailure('process_failed')
        try:
            return out.decode('utf-8')
        except UnicodeError as exc:
            raise LocalFailure('invalid_response') from exc
    except (subprocess.TimeoutExpired, OSError, ValueError):
        # A child (MinerU) can outlive its parent and keep stdout open.
        try:
            kill_tree(p)
        except (ProcessLookupError, PermissionError, FileNotFoundError):
            if p.poll() is None:
                p.kill()
        try:
            p.wait(timeout=.05)
        except subprocess.TimeoutExpired:
            threading.Thread(target=p.wait, daemon=True).start()
        raise
    finally:
        if payload is not None and not p.stdin.closed:
            p.stdin.close()
        p.stdout.close()


def _execute_windows(cmd, payload, deadline, cwd=None):
    """Windows variant: pipes cannot be select()ed, so use bounded helper threads.

    The reader caps output and the whole tree is killed on output overflow or
    deadline overrun, exactly like the POSIX process-group path.
    """
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired(cmd, 0)
    p = subprocess.Popen(cmd, cwd=cwd,
                         stdin=subprocess.PIPE if payload is not None else subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    out = bytearray()
    failure = []

    def writer():
        try:
            if payload:
                p.stdin.write(payload)
            p.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    def reader():
        try:
            while True:
                chunk = os.read(p.stdout.fileno(), 65536)
                if not chunk:
                    return
                out.extend(chunk)
                if len(out) > MAX_OUTPUT:
                    failure.append(LocalFailure('output_too_large'))
                    return
        except OSError:
            return

    if payload is not None:
        threading.Thread(target=writer, daemon=True).start()
    read_thread = threading.Thread(target=reader, daemon=True)
    read_thread.start()
    try:
        read_thread.join(timeout=max(0, deadline - time.monotonic()))
        if failure:
            raise failure[0]
        if read_thread.is_alive():
            raise subprocess.TimeoutExpired(cmd, max(0, deadline - time.monotonic()))
        p.wait(timeout=max(0, deadline - time.monotonic()))
        if p.returncode:
            raise LocalFailure('process_failed')
        try:
            return bytes(out).decode('utf-8')
        except UnicodeError as exc:
            raise LocalFailure('invalid_response') from exc
    except (subprocess.TimeoutExpired, OSError, ValueError):
        try:
            kill_tree(p)
        except (ProcessLookupError, PermissionError, FileNotFoundError):
            if p.poll() is None:
                p.kill()
        try:
            p.wait(timeout=.05)
        except subprocess.TimeoutExpired:
            threading.Thread(target=p.wait, daemon=True).start()
        raise
    finally:
        if payload is not None and p.stdin is not None and not p.stdin.closed:
            p.stdin.close()
        if p.stdout is not None:
            p.stdout.close()
