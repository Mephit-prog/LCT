import time
import io
import zipfile

import httpx

from wineid.mineru_cloud import MinerUBackend, MinerUError
from wineid.mineru_http import MinerUImageOCR
from wineid.roi import ROI


class Backend:
    def __init__(self, failure=None):
        self.failure = failure
        self.deadline = None
        self.closed = False

    def reserve(self, name, digest):
        if self.failure:
            raise MinerUError(self.failure)
        return 'batch', 'upload'

    def upload(self, url, path):
        pass

    def _remaining(self, cap):
        if time.monotonic() >= self.deadline:
            raise MinerUError('deadline')

    def poll(self, batch, digest):
        return {'state': 'done', 'full_zip_url': 'download'}

    def download(self, url, path):
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr('result/full.md', 'Каберне 2022')

    def close(self):
        self.closed = True


def test_cloud_image_and_errors():
    roi = ROI('photo', 'sha', (10, 10), (0, 0, 10, 10), 'full', 'image/png', b'png')
    for failure, status, code in [(None, 'ok', None), ('upstream_busy', 'provider_error', 'upstream_busy'),
                                  ('upstream_rejected', 'provider_error', 'upstream_rejected')]:
        backend = Backend(failure)
        result = MinerUImageOCR('test', backend_factory=lambda: backend).recognize(roi, time.monotonic() + 5)
        assert (result.status, result.error_code) == (status, code)
        assert backend.closed
        if failure is None:
            assert 'каберне' in result.normalized_text
    result = MinerUImageOCR('test', backend_factory=Backend).recognize(roi, time.monotonic() - 1)
    assert result.status == 'timeout'


def test_http_transport_upload_poll_download_and_failures():
    roi = ROI('photo', 'sha', (10, 10), (0, 0, 10, 10), 'full', 'image/png', b'png')
    for api_status, expected in [(200, 'ok'), (401, 'provider_error'), (429, 'provider_error')]:
        calls = []

        def handler(request):
            calls.append(request)
            if request.url.host == 'mineru.net':
                assert request.headers['authorization'] == 'Bearer test-token'
                if api_status != 200:
                    return httpx.Response(api_status, text='secret upstream body')
                if request.method == 'POST':
                    assert request.url.path == '/api/v4/file-urls/batch'
                    return httpx.Response(200, json={'code': 0, 'data': {'batch_id': 'b1', 'file_urls': [
                        'https://mineru.oss-cn-shanghai.aliyuncs.com/u?signature=secret']}})
                return httpx.Response(200, json={'code': 0, 'data': {'batch_id': 'b1', 'extract_result': [
                    {'data_id': __import__('hashlib').sha256(roi.crop).hexdigest(),
                     'state': 'done', 'full_zip_url': 'https://cdn-mineru.openxlab.org.cn/r.zip?signature=secret'}]}})
            assert 'authorization' not in request.headers
            if request.method == 'PUT':
                assert request.read() == roi.crop
                return httpx.Response(200)
            archive = io.BytesIO()
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('full.md', 'Каберне 2022')
            return httpx.Response(200, content=archive.getvalue())

        def factory():
            return MinerUBackend('test-token', client=httpx.Client(transport=httpx.MockTransport(handler)),
                                 min_interval=0, retries=0, max_zip_bytes=1024)

        result = MinerUImageOCR('test-token', backend_factory=factory).recognize(roi, time.monotonic() + 5)
        assert result.status == expected
        assert result.error_code == {200: None, 401: 'invalid_api_response', 429: 'upstream_busy'}[api_status]
        assert len(calls) == (4 if api_status == 200 else 1)
        assert 'secret' not in str(result)
