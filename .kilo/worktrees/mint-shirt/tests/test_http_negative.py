"""Negative HTTP/OCR/pipeline coverage (review P2-13).

Covers: 413 on oversized/invalid/missing Content-Length, chunked bodies, 404 on
unknown paths, /ready -> 503, slow-body timeout, bearer auth (401), bounded
concurrency (503 busy), OCR provider errors / invalid_response / no_text /
timeout, ocr_not_configured and deadline between pipeline steps.
"""
import http.client
import io
import json
import socket
import threading
import time
import unittest
from http.server import ThreadingHTTPServer

import httpx
from PIL import Image

from wineid.catalog import Wine, normalize
from wineid.search import Index
from wineid.pipeline import Pipeline
from wineid.ocr import MockOCR, OCRResult
from wineid.server import BoundedHTTPServer, create_handler, MAX_BODY


def wine(slug, name, producer):
    return Wine(slug, slug, name, producer, normalize(name),
                normalize(producer), (2,), {}, 'test')


def picture(fmt='WEBP'):
    buffer = io.BytesIO()
    Image.new('RGB', (120, 90), 'white').save(buffer, format=fmt)
    return buffer.getvalue()


def multipart(image, boundary='abc'):
    return (b'--' + boundary.encode('ascii') +
            b'\r\nContent-Disposition: form-data; name="image"; filename="a.jpg"\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + image +
            b'\r\n--' + boundary.encode('ascii') + b'--\r\n')


class FakeOCR:
    """Deterministic OCR stub returning a fixed OCRResult."""

    def __init__(self, result):
        self._result = result

    def recognize(self, roi, deadline):
        return self._result


class BlockingOCR:
    """OCR stub that blocks until released (concurrency tests)."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def recognize(self, roi, deadline):
        self.entered.set()
        self.release.wait(3)
        return OCRResult(roi.roi_id, 'ok', 'Вино 2020', 'вино 2020', 'blocking', 0)


class SlowOCR:
    """OCR stub that consumes the shared deadline."""

    def __init__(self, seconds):
        self._seconds = seconds

    def recognize(self, roi, deadline):
        time.sleep(self._seconds)
        return OCRResult(roi.roi_id, 'ok', 'Вино 2020', 'вино 2020', 'slow', 0)


class HttpNegativeTests(unittest.TestCase):
    def setUp(self):
        self.index = Index([wine('a-2020', 'Вино 2020', 'Завод А')])
        self.image = picture()

    def _url(self, server):
        return f'http://127.0.0.1:{server.server_address[1]}'

    def _start(self, server_type, pipeline, **handler_kwargs):
        server = server_type(('127.0.0.1', 0),
                             create_handler(pipeline, **handler_kwargs))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        # addCleanup runs in LIFO order: shutdown -> close -> join.
        self.addCleanup(thread.join, 3)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def _raw(self, port, request_bytes, read_timeout=3.0):
        with socket.create_connection(('127.0.0.1', port), timeout=3.0) as sock:
            sock.sendall(request_bytes)
            sock.settimeout(read_timeout)
            data = b''
            try:
                while True:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    data += chunk
            except socket.timeout:
                pass
        return data

    @staticmethod
    def _status(raw):
        return int(raw.split(b' ', 2)[1])

    # ------------------------------------------------------------------
    # HTTP request validation
    # ------------------------------------------------------------------
    def test_413_oversized_content_length(self):
        pipe = Pipeline(self.index, MockOCR('Вино 2020'), csv_sha256='test')
        server = self._start(ThreadingHTTPServer, pipe)
        conn = http.client.HTTPConnection('127.0.0.1', server.server_address[1], timeout=3)
        conn.request('POST', '/v1/recognize', body=multipart(self.image),
                     headers={'Content-Type': 'multipart/form-data; boundary=abc',
                              'Content-Length': str(MAX_BODY + 1)})
        response = conn.getresponse()
        self.assertEqual(response.status, 413)
        self.assertEqual(json.loads(response.read())['reason_codes'], ['invalid_body_size'])
        conn.close()

    def test_413_missing_content_length(self):
        pipe = Pipeline(self.index, MockOCR('Вино 2020'), csv_sha256='test')
        server = self._start(ThreadingHTTPServer, pipe)
        request = (b'POST /v1/recognize HTTP/1.1\r\nHost: test\r\n'
                   b'Content-Type: multipart/form-data; boundary=abc\r\n'
                   b'Connection: close\r\n\r\n' + multipart(self.image))
        self.assertEqual(self._status(self._raw(server.server_address[1], request)), 413)

    def test_413_invalid_content_length(self):
        pipe = Pipeline(self.index, MockOCR('Вино 2020'), csv_sha256='test')
        server = self._start(ThreadingHTTPServer, pipe)
        request = (b'POST /v1/recognize HTTP/1.1\r\nHost: test\r\n'
                   b'Content-Type: multipart/form-data; boundary=abc\r\n'
                   b'Content-Length: nope\r\nConnection: close\r\n\r\n')
        self.assertEqual(self._status(self._raw(server.server_address[1], request)), 413)

    def test_413_chunked_body_without_content_length(self):
        pipe = Pipeline(self.index, MockOCR('Вино 2020'), csv_sha256='test')
        server = self._start(ThreadingHTTPServer, pipe)
        chunked = b'4\r\nwiki\r\n5\r\npedia\r\n0\r\n\r\n'
        request = (b'POST /v1/recognize HTTP/1.1\r\nHost: test\r\n'
                   b'Content-Type: multipart/form-data; boundary=abc\r\n'
                   b'Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n' + chunked)
        self.assertEqual(self._status(self._raw(server.server_address[1], request)), 413)

    def test_404_unknown_paths(self):
        pipe = Pipeline(self.index, MockOCR('Вино 2020'), csv_sha256='test')
        server = self._start(ThreadingHTTPServer, pipe)
        url = self._url(server)
        with httpx.Client() as client:
            self.assertEqual(client.get(url + '/nope').status_code, 404)
            self.assertEqual(client.post(url + '/nope').status_code, 404)

    def test_ready_503_empty_index(self):
        empty = Pipeline(Index([]), None, csv_sha256='anything')
        server = self._start(ThreadingHTTPServer, empty)
        response = httpx.get(self._url(server) + '/ready')
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json()['ready'])

    def test_slow_body_read_times_out(self):
        pipe = Pipeline(self.index, MockOCR('Вино 2020'), csv_sha256='test')
        server = self._start(ThreadingHTTPServer, pipe, timeout=0.3)
        request = (b'POST /v1/recognize HTTP/1.1\r\nHost: test\r\n'
                   b'Content-Type: multipart/form-data; boundary=abc\r\n'
                   b'Content-Length: ' + str(MAX_BODY).encode('ascii') + b'\r\n'
                   b'Connection: close\r\n\r\n--abc\r\npartial body')
        raw = self._raw(server.server_address[1], request, read_timeout=2.0)
        self.assertEqual(self._status(raw), 400)

    # ------------------------------------------------------------------
    # Authentication (P1-4)
    # ------------------------------------------------------------------
    def test_bearer_auth_required_when_token_configured(self):
        pipe = Pipeline(self.index, MockOCR('Вино 2020'), csv_sha256='test')
        server = self._start(ThreadingHTTPServer, pipe, token='sekret')
        url = self._url(server)
        with httpx.Client() as client:
            no_auth = client.post(url + '/v1/recognize',
                                  files={'image': ('a.jpg', self.image)})
            self.assertEqual(no_auth.status_code, 401)
            self.assertEqual(no_auth.json()['reason_codes'], ['unauthorized'])
            bad = client.post(url + '/v1/recognize',
                              files={'image': ('a.jpg', self.image)},
                              headers={'Authorization': 'Bearer wrong'})
            self.assertEqual(bad.status_code, 401)
            good = client.post(url + '/v1/recognize',
                               files={'image': ('a.jpg', self.image)},
                               headers={'Authorization': 'Bearer sekret'})
            self.assertEqual(good.status_code, 200)
            eval_no = client.post(url + '/v1/eval/predict',
                                  files={'image': ('a.jpg', self.image)})
            self.assertEqual(eval_no.status_code, 401)
            self.assertIsNone(eval_no.json()['slug'])
        # /ready stays open without auth.
        self.assertEqual(httpx.get(url + '/ready').status_code, 200)

    # ------------------------------------------------------------------
    # Bounded concurrency (P1-4)
    # ------------------------------------------------------------------
    def test_busy_returns_503_without_new_worker(self):
        ocr = BlockingOCR()
        # full-frame experimental mode reaches OCR without a manual bbox, so the
        # first request holds the single worker slot inside recognize().
        pipe = Pipeline(self.index, ocr, csv_sha256='test',
                        roi_mode='full_frame_experimental')
        server = BoundedHTTPServer(('127.0.0.1', 0), create_handler(pipe),
                                   max_workers=1)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 3)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        url = self._url(server)
        results = []
        worker = threading.Thread(target=lambda: results.append(
            httpx.post(url + '/v1/recognize',
                       files={'image': ('a.jpg', self.image)})))
        worker.start()
        try:
            self.assertTrue(ocr.entered.wait(2))
            with httpx.Client() as client:
                busy = client.post(url + '/v1/recognize',
                                   files={'image': ('a.jpg', self.image)})
            self.assertEqual(busy.status_code, 503)
            self.assertEqual(busy.json()['reason_codes'], ['busy'])
        finally:
            ocr.release.set()
            worker.join(3)
        self.assertEqual(results[0].status_code, 200)

    # ------------------------------------------------------------------
    # OCR and pipeline negatives
    # ------------------------------------------------------------------
    def test_ocr_provider_errors_and_statuses(self):
        base = {'csv_sha256': 'test'}
        for code in ('upstream_5xx', 'upstream_4xx', 'rate_limited',
                     'invalid_response'):
            pipe = Pipeline(self.index, FakeOCR(OCRResult(
                'r', 'provider_error', None, None, 'fake', 0, code)), **base)
            result = pipe.predict(self.image, bbox=(0, 0, 100, 80))
            self.assertEqual(result['status'], 'error')
            self.assertEqual(result['reason_codes'],
                             ['ocr_provider_error', code])
        timeout = Pipeline(self.index, FakeOCR(OCRResult(
            'r', 'timeout', None, None, 'fake', 0, 'deadline')), **base)
        self.assertEqual(timeout.predict(self.image, bbox=(0, 0, 100, 80))['reason_codes'],
                         ['ocr_timeout', 'deadline'])
        no_text = Pipeline(self.index, FakeOCR(OCRResult(
            'r', 'no_text', None, None, 'fake', 0)), **base)
        result = no_text.predict(self.image, bbox=(0, 0, 100, 80))
        self.assertEqual(result['status'], 'unreadable')
        self.assertEqual(result['reason_codes'], ['no_text'])

    def test_ocr_not_configured(self):
        result = Pipeline(self.index, None, csv_sha256='test').predict(
            self.image, bbox=(0, 0, 100, 80))
        self.assertEqual(result['reason_codes'], ['ocr_not_configured'])

    def test_deadline_between_pipeline_steps(self):
        # OCR overruns the shared deadline; the search step must not run.
        result = Pipeline(self.index, SlowOCR(0.25), csv_sha256='test').predict(
            self.image, bbox=(0, 0, 100, 80), seconds=0.05)
        self.assertEqual(result['reason_codes'], ['deadline'])


if __name__ == '__main__':
    unittest.main()