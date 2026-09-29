"""Stdlib HTTP adapter: no external parser dependency; one bounded image part.

Optional bearer auth: set ``WINE_TOKEN`` to require ``Authorization: Bearer``
on the POST endpoints (401 otherwise); ``/ready`` stays open. Concurrency is
bounded by :class:`BoundedHTTPServer` (``WINE_MAX_WORKERS``, default 8): a
saturated server answers 503 on the connection instead of spawning unbounded
threads (review P1-4). Internal failures never leak details to the client
(review P2-16).
"""
import json
import math
import os
import socket
import threading
import time
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .roi import MAX_BYTES

MAX_BODY = MAX_BYTES + 65536


def multipart_image(content_type, body):
    if not content_type.lower().startswith('multipart/form-data;') or 'boundary=' not in content_type.lower():
        raise ValueError('expected_multipart')
    msg = BytesParser(policy=default).parsebytes(
        b'MIME-Version: 1.0\r\nContent-Type: ' + content_type.encode('ascii') +
        b'\r\n\r\n' + body)
    if not msg.is_multipart():
        raise ValueError('invalid_multipart')
    parts = [p for p in msg.iter_parts() if p.get_param('name', header='content-disposition') == 'image']
    if len(parts) != 1 or len(list(msg.iter_parts())) != 1:
        raise ValueError('expected_one_image')
    data = parts[0].get_payload(decode=True)
    if not data or len(data) > MAX_BYTES:
        raise ValueError('invalid_size')
    return data


class BoundedHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer whose concurrent request processing is bounded.

    A semaphore is acquired before a worker thread is spawned and released when
    that worker finishes, so slow requests count towards the limit. When all
    slots are taken, the connection is answered 503 directly (no new thread).
    """
    daemon_threads = True

    def __init__(self, *args, max_workers=8, **kwargs):
        super().__init__(*args, **kwargs)
        if type(max_workers) is not int or max_workers < 1:
            raise ValueError('max_workers must be a positive integer')
        self._slots = threading.BoundedSemaphore(max_workers)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            try:
                self._respond_busy(request)
            finally:
                self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()

    @staticmethod
    def _respond_busy(request):
        body = json.dumps({'status': 'error', 'slug': None,
                           'reason_codes': ['busy']}).encode('utf-8')
        try:
            request.sendall(b'HTTP/1.1 503 Service Unavailable\r\n'
                            b'Content-Type: application/json; charset=utf-8\r\n'
                            b'Content-Length: ' + str(len(body)).encode('ascii') +
                            b'\r\nConnection: close\r\n\r\n' + body)
            # Graceful TCP shutdown on Windows: send FIN, drain incoming data so
            # the peer does not see an immediate TCP RST (WSAECONNRESET).
            request.shutdown(socket.SHUT_WR)
            request.settimeout(1.0)
            while request.recv(65536):
                pass
        except OSError:
            pass


def create_handler(pipeline, *, timeout=8.5, token=None):
    """Build a request handler.

    ``timeout`` bounds the per-connection body read (and the pipeline deadline);
    ``token`` enables Bearer auth for the POST endpoints (None disables it).
    """
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('invalid request timeout')
    if pipeline.policy is not None and timeout != pipeline.request_timeout:
        raise ValueError('request timeout differs from calibrated policy profile')

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            # no images, OCR text or authorization in logs
            pass

        def send_json(self, code, obj):
            data = json.dumps(obj, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass
            if self.close_connection and os.name == 'nt':
                self._graceful_close()

        def _graceful_close(self):
            # Windows raises WSAECONNRESET when a socket with unread inbound bytes
            # closes. Half-close and drain a bounded amount so the peer finishes.
            try:
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_WR)
            except OSError:
                return
            try:
                self.connection.settimeout(0.5)
                budget = 65536
                while budget > 0:
                    chunk = self.connection.recv(min(65536, budget))
                    if not chunk:
                        break
                    budget -= len(chunk)
            except OSError:
                pass

        def _authorized(self):
            if token is None:
                return True
            return self.headers.get('Authorization', '') == 'Bearer ' + token

        def do_GET(self):
            if self.path in ('/ready', '/api/health/recognition'):
                catalog_ready = bool(pipeline.index.wines) and pipeline.csv_sha256 == pipeline.index.wines[0].csv_sha256
                # Catalog readiness must not be mistaken for recognition readiness
                # (STATUS.md B06): recognition needs OCR and a non-refusing ROI mode.
                recognition_ready = pipeline.recognition_ready
                self.send_json(200 if (recognition_ready if self.path != '/ready' else catalog_ready) else 503, {
                    'ready': catalog_ready, 'catalog_ready': catalog_ready,
                    'recognition_ready': recognition_ready,
                    'acceptance_ready': pipeline.acceptance_ready,
                    'catalog_sha256': pipeline.csv_sha256,
                    'wine_count': len(pipeline.index.wines), 'ocr_configured': pipeline.ocr is not None,
                    'ocr_ready': pipeline.ocr_ready,
                    'automatic_roi': pipeline.roi_mode == 'auto_bbox_experimental',
                    'roi_mode': pipeline.roi_mode,
                    'policy_configured': pipeline.policy is not None})
            else:
                self.send_json(404, {'error': 'not_found'})

        def _read_body(self, length, started, timeout):
            chunks = []
            while length:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise socket.timeout('deadline')
                self.connection.settimeout(remaining)
                # read1 returns after one socket read; read(n) may wait for n
                # bytes while a trickle client repeatedly resets socket timeout.
                chunk = self.rfile.read1(min(length, 65536))
                if not chunk:
                    raise socket.timeout('deadline')
                chunks.append(chunk)
                length -= len(chunk)
            return b''.join(chunks)

        def do_POST(self):
            started = time.monotonic()
            if self.path not in ('/v1/recognize', '/v1/eval/predict'):
                return self.send_json(404, {'error': 'not_found'})
            eval_mode = self.path == '/v1/eval/predict'
            def reply(code, result):
                self.send_json(code, {'slug': result.get('slug')} if eval_mode else result)
            if not self._authorized():
                # Reject from headers, never buffer an unauthenticated upload.
                # Close instead of draining attacker-controlled body bytes.
                self.close_connection = True
                return reply(401, {'status': 'error', 'slug': None, 'reason_codes': ['unauthorized']})
            length = self.headers.get('Content-Length', '')
            if (self.headers.get('Transfer-Encoding') or not length.isdecimal()
                    or not 0 < int(length) <= MAX_BODY):
                self.close_connection = True
                return reply(413, {'status': 'error', 'slug': None, 'reason_codes': ['invalid_body_size']})
            content_length = int(length)
            self.connection.settimeout(timeout)
            try:
                content_type = self.headers.get('Content-Type', '')
                image = multipart_image(content_type, self._read_body(content_length, started, timeout))
                result = pipeline.predict(image, seconds=max(0, timeout-(time.monotonic()-started)),
                                          force_top1=eval_mode)
                if eval_mode and result.get('slug') is None:
                    reply(422 if result['status'] == 'unreadable' else 503, result)
                elif not eval_mode and result['status'] == 'error' and (
                        'ocr_capacity_exceeded' in result['reason_codes'] or
                        'vision_capacity_exceeded' in result['reason_codes'] or
                        'busy' in result['reason_codes']):
                    reply(503, {'status': 'error', 'slug': None, 'reason_codes': ['busy']})
                elif not eval_mode and result['status'] == 'error' and any(
                        code in result['reason_codes'] for code in
                        ('vision_worker_failed', 'vision_worker_unavailable')):
                    reply(503, {'status': 'error', 'slug': None,
                                'reason_codes': ['vision_unavailable']})
                elif not eval_mode and result['status'] == 'error' and any(
                        code in result['reason_codes'] for code in
                        ('worker_unavailable', 'worker_failed', 'inference_failed')):
                    reply(503, {'status': 'error', 'slug': None,
                                'reason_codes': ['ocr_unavailable']})
                else:
                    reply(200, result)
            except (ValueError, UnicodeError) as exc:
                reason = str(exc)
                if reason not in ('expected_multipart', 'invalid_multipart',
                                  'expected_one_image', 'invalid_size'):
                    reason = 'invalid_multipart'
                reply(400, {'status': 'error', 'slug': None, 'reason_codes': [reason]})
            except socket.timeout:
                # Slow/missing body: the request-level read timed out.
                reply(400, {'status': 'error', 'slug': None, 'reason_codes': ['deadline']})
            except Exception:
                # Never leak internal details to the client (review P2-16).
                try:
                    reply(500, {'status': 'error', 'slug': None, 'reason_codes': ['internal']})
                except (BrokenPipeError, ConnectionResetError):
                    pass

    return Handler


def main():
    from .runtime import open_runtime
    with open_runtime(os.environ) as runtime:
        cfg = runtime.settings
        server = BoundedHTTPServer((cfg.host, cfg.port),
                                   create_handler(runtime.pipeline, timeout=cfg.timeout,
                                                  token=os.environ.get('WINE_TOKEN') or None),
                                   max_workers=cfg.max_workers)
        try:
            server.serve_forever()
        finally:
            server.server_close()


if __name__ == '__main__':
    main()
