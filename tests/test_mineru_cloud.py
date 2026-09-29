"""No network, no credentials, no OCR weights. Opt-in Precision document track."""
import hashlib
import io
import json
import os
import zipfile

import httpx
import pytest
from fastapi.testclient import TestClient

from wineid.mineru_cloud import (MinerUBackend, MinerUConfig, MinerUError,
    digest_file, prepare, safe_url, unpack_result, normalize_ru, verify_callback)
from wineid.mineru_jobs import JobStore, JobWorker, create_callback_app
from wineid.mineru_eval import evaluate as evaluate_cer

SIGNED = 'https://mineru.oss-cn-shanghai.aliyuncs.com/api-upload/abc?signature=secret'
ZIP_URL = 'https://cdn-mineru.openxlab.org.cn/pdf/abc.zip?signature=secret'
SEED = 's' * 32


def archive(entries=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, value in (entries or {'nested/full.md': 'Привет, Ёж! 2024',
                                        'nested/doc_content_list.json': '[{"type":"text","page_idx":0}]',
                                        'nested/layout.json': '{}'}).items():
            z.writestr(name, value)
    return buffer.getvalue()


def test_confirmed_language_and_config_controls():
    cfg = MinerUConfig()
    assert cfg.request('part.pdf', 'a'*64)['files'] == [
        {'name': 'part.pdf', 'data_id': 'a'*64, 'is_ocr': True}]
    assert cfg.request('part.pdf', 'a'*64)['language'] == 'cyrillic'
    assert cfg.sha256 == MinerUConfig(callback='https://example.org/callback', seed=SEED).sha256
    for kw in ({'language': 'ch'}, {'model_version': 'MinerU-HTML'},
               {'callback': 'https://127.0.0.1/callback', 'seed': SEED},
               {'callback': 'http://example.org/callback', 'seed': SEED},
               {'callback': 'https://example.org/callback'}, {'is_ocr': 1}):
        with pytest.raises((ValueError, MinerUError)):
            MinerUConfig(**kw)
    for url in ('http://mineru.oss-cn-shanghai.aliyuncs.com/path',
                'https://127.0.0.1/a', 'https://mineru.oss-cn-shanghai.aliyuncs.com.evil.tld/a',
                'https://mineru.oss-cn-shanghai.aliyuncs.com:8080/x',
                'https://user:pass@mineru.oss-cn-shanghai.aliyuncs.com/x'):
        with pytest.raises(MinerUError, match='unsafe_remote_url'):
            safe_url(url, {'mineru.oss-cn-shanghai.aliyuncs.com'})


def test_precision_http_contract_and_no_token_on_signed_urls(tmp_path):
    data = tmp_path / 'part.pdf'
    data.write_bytes(b'%PDF-1.4 dummy')
    calls = []
    def handler(req):
        calls.append(req)
        if req.method == 'POST':
            body = json.loads(req.content)
            assert body['model_version'] == 'pipeline' and body['language'] == 'cyrillic'
            assert body['files'][0]['data_id'] == digest_file(data)
            assert req.headers['Authorization'] == 'Bearer token'
            return httpx.Response(200, json={'code': 0, 'data': {'batch_id': 'batch-1',
                                                                  'file_urls': [SIGNED]}})
        if req.method == 'PUT':
            assert req.url == SIGNED and 'Authorization' not in req.headers
            assert 'content-type' not in req.headers
            assert req.read() == data.read_bytes()
            return httpx.Response(200)
        if req.url.path.startswith('/api/v4/extract-results/'):
            assert req.headers['Authorization'] == 'Bearer token'
            return httpx.Response(200, json={'code': 0, 'data': {'batch_id': 'batch-1',
                'extract_result': [{'data_id': digest_file(data), 'state': 'done',
                                    'full_zip_url': ZIP_URL}]}})
        assert req.method == 'GET' and str(req.url) == ZIP_URL
        assert 'Authorization' not in req.headers
        return httpx.Response(200, content=archive())
    client = httpx.Client(transport=httpx.MockTransport(handler))
    backend = MinerUBackend('token', client=client, min_interval=0)
    batch, url = backend.reserve(data.name, digest_file(data))
    backend.upload(url, data)
    assert backend.poll(batch, digest_file(data))['state'] == 'done'
    output = tmp_path / 'out.zip'
    backend.download(ZIP_URL, output)
    qa = unpack_result(output, tmp_path / 'result', data_id=digest_file(data), config_sha=backend.config.sha256)
    assert unpack_result(output, tmp_path / 'result', data_id=digest_file(data),
                         config_sha=backend.config.sha256) == qa
    assert (tmp_path / 'result' / 'normalized.txt').read_text(encoding='utf-8') == 'привет, еж! 2024'
    assert qa['empty_text'] is False and qa['outputs'] == ['content_list.json', 'full.md', 'layout.json']
    assert len(calls) == 4


def test_fail_closed_post_no_retry_quotas_and_poll_retry():
    n = 0
    def handler(req):
        nonlocal n
        n += 1
        if req.method == 'POST':
            return httpx.Response(200, json={'code': -60018, 'msg': 'secret or document text'})
        if n == 2:
            return httpx.Response(503)
        return httpx.Response(200, json={'code': 0, 'data': {'batch_id': 'id',
            'extract_result': [{'data_id': 'a'*64, 'state': 'pending'}]}})
    backend = MinerUBackend('token', client=httpx.Client(transport=httpx.MockTransport(handler)),
                            min_interval=0, retries=1)
    with pytest.raises(MinerUError, match='daily_quota') as error:
        backend.reserve('part.pdf', 'a'*64)
    assert 'secret' not in str(error.value) and n == 1
    assert backend.poll('id', 'a'*64)['state'] == 'pending' and n == 3
    with pytest.raises(MinerUError, match='result_data_mismatch'):
        backend.poll('id', 'b'*64)
    def disconnect(_): raise httpx.ConnectTimeout('no')
    fail = MinerUBackend('token', client=httpx.Client(transport=httpx.MockTransport(disconnect)), retries=3,
                         min_interval=0)
    with pytest.raises(MinerUError, match='network_failure'):
        fail.reserve('part.pdf', 'a'*64)  # POST not repeated even if response was lost


def test_safe_zip_and_pdf_splitting(tmp_path):
    names = ['../bad/full.md', '/abs/full.md', 'ok/../full.md']
    if os.name != 'nt':
        # Python's zipfile rewrites '\' to '/' on Windows before the extractor
        # sees it, so the raw-backslash entry can only be built on POSIX.
        names.append('ok\\full.md')
    for name in names:
        location = tmp_path / 'bad.zip'
        location.write_bytes(archive({name: 'hello', 'x_content_list.json': '[]'}))
        with pytest.raises(MinerUError, match='unsafe_zip'):
            unpack_result(location, tmp_path / 'out', data_id='x', config_sha='y')
    location.write_bytes(archive({'full.md': 'hi'}))
    with pytest.raises(MinerUError, match='missing_zip_outputs'):
        unpack_result(location, tmp_path / 'out', data_id='x', config_sha='y')
    from pypdf import PdfWriter, PdfReader
    pdf = tmp_path / 'book.pdf'
    writer = PdfWriter()
    for _ in range(205): writer.add_blank_page(width=100, height=100)
    writer.write(pdf)
    parts = prepare(pdf, tmp_path / 'spool')
    assert [pages for _, pages in parts] == [(1, 200), (201, 205)]
    assert sum(len(PdfReader(str(path)).pages) for path, _ in parts) == 205
    assert prepare(pdf, tmp_path / 'spool') == parts
    assert normalize_ru('  Ёж   ПрИвЕт \n ') == 'еж привет'


def test_poll_worker_end_to_end_with_mock_transport(tmp_path):
    doc = tmp_path / 'scan.png'
    from PIL import Image
    Image.new('RGB', (20, 20), 'white').save(doc)
    digest = digest_file(doc)
    def handler(request):
        if request.method == 'POST':
            assert request.headers['authorization'] == 'Bearer test-token'
            return httpx.Response(200, json={'code': 0, 'data': {'batch_id': 'poll-batch',
                                                                  'file_urls': [SIGNED]}})
        if request.method == 'PUT':
            assert request.read() == doc.read_bytes()
            return httpx.Response(200)
        if request.url.host == 'mineru.net':
            assert request.method == 'GET'
            return httpx.Response(200, json={'code': 0, 'data': {'batch_id': 'poll-batch',
                'extract_result': [{'state': 'done', 'data_id': digest, 'full_zip_url': ZIP_URL}]}})
        return httpx.Response(200, content=archive())
    store = JobStore(tmp_path / 'private')
    [key] = store.enqueue([doc], MinerUConfig())
    backend = MinerUBackend('test-token', client=httpx.Client(transport=httpx.MockTransport(handler)),
                            min_interval=0)
    worker = JobWorker(store, backend, poll_interval=1)
    assert worker.once() == 'submitted'
    store.update(key, next_at=0)
    assert worker.once() == 'ready'
    assert worker.once() == 'done'
    assert (store.root / 'results' / key / 'content_list.json').exists()
    assert len(store.summary()) == 1
    store.close()


def test_callback_signature_replay_and_queue(tmp_path):
    work_dir = tmp_path / 'private'
    store = JobStore(work_dir)
    doc = tmp_path / 'doc.png'
    from PIL import Image
    Image.new('RGB', (30, 30), 'white').save(doc)
    cfg = MinerUConfig(callback='https://example.org/callbacks/mineru', seed=SEED)
    [key] = store.enqueue([doc], cfg)
    assert store.enqueue([doc], cfg) == [key] and len(store.summary()) == 1
    assert store.db.execute('SELECT count(*) FROM sources WHERE job_key=?', (key,)).fetchone()[0] == 1
    class Backend:
        config = cfg
        download_hosts = frozenset({'cdn-mineru.openxlab.org.cn'})
        def reserve(self, name, digest):
            assert name == 'doc.png' and digest == digest_file(doc)
            return 'b1', SIGNED
        def upload(self, url, path): assert url == SIGNED and path == doc
        def poll(self, batch, digest): raise AssertionError('callback mode must not poll immediately')
        def download(self, url, path):
            assert url == ZIP_URL
            path.write_bytes(archive())
    worker = JobWorker(store, Backend())
    assert worker.once() == 'submitted'
    assert worker.once() is None
    content = json.dumps({'batch_id': 'b1', 'extract_result': [
        {'state': 'done', 'data_id': digest_file(doc), 'full_zip_url': ZIP_URL}]})
    checksum = hashlib.sha256(('uid' + SEED + content).encode()).hexdigest()
    app = create_callback_app(store, uid='uid', seed=SEED)
    client = TestClient(app)
    assert client.post('/callbacks/mineru', json={'content': content, 'checksum': '0'*64}).status_code == 400
    assert client.post('/callbacks/mineru', json={'content': content, 'checksum': checksum}).json() == {'status': 'ok'}
    assert client.post('/callbacks/mineru', json={'content': content, 'checksum': checksum}).status_code == 200
    assert store.get(key)['state'] == 'ready'
    assert worker.once() == 'done'
    assert store.summary()[0]['result_dir']
    assert worker.once() is None
    store.close()


def test_queue_capacity_is_atomic_and_claim_is_exclusive(tmp_path):
    store = JobStore(tmp_path / 'private', max_active=1)
    a, b = tmp_path / 'one.png', tmp_path / 'two.png'
    from PIL import Image
    Image.new('RGB', (20, 20), 'white').save(a)
    Image.new('RGB', (20, 20), 'red').save(b)
    with pytest.raises(MinerUError, match='queue_full'):
        store.enqueue([a, b], MinerUConfig())
    assert store.summary() == []
    [key] = store.enqueue([a], MinerUConfig())
    assert store.claim(callback_mode=False)['key'] == key
    assert store.claim(callback_mode=False) is None
    assert store.get(key)['state'] == 'reserving'  # no duplicate POST, even after restart
    store.close()


def test_interrupted_reservation_never_reposts_and_input_mutation(tmp_path):
    store = JobStore(tmp_path / 'private')
    doc = tmp_path / 'doc.png'
    from PIL import Image
    Image.new('RGB', (30, 30), 'white').save(doc)
    cfg = MinerUConfig()
    [key] = store.enqueue([doc], cfg)
    doc.write_bytes(b'modified')
    class Backend:
        config = cfg
        def reserve(self, *args): raise AssertionError('no changed input may be sent')
    assert JobWorker(store, Backend()).once() == 'needs_review'
    assert store.get(key)['error_code'] == 'input_or_config_changed'
    store.close()


def test_independent_cer_by_font_class(tmp_path):
    truth = tmp_path / 'gt.txt'
    truth.write_text('Привет, Ёж! 2024', encoding='utf-8')
    key = 'a'*64
    result = tmp_path / 'results' / key
    result.mkdir(parents=True)
    (result / 'normalized.txt').write_text('привет, еж! 2024', encoding='utf-8')
    manifest = tmp_path / 'gt.jsonl'
    row = {'key': key, 'gt_path': 'gt.txt', 'gt_sha256': digest_file(truth),
           'font_class': 'print', 'split': 'validation', 'gt_source': 'manual'}
    manifest.write_text(json.dumps(row) + '\n')
    metrics = evaluate_cer(manifest, tmp_path / 'results')
    assert metrics['by_font_class']['print']['cer'] == 0
    (result / 'normalized.txt').unlink()
    metrics = evaluate_cer(manifest, tmp_path / 'results')
    assert metrics['by_font_class']['print']['missing'] == 1
    assert metrics['by_font_class']['print']['cer'] == 1
    with pytest.raises(ValueError, match='empty split'):
        evaluate_cer(manifest, tmp_path / 'results', split='test')


def test_callback_rejects_tamper_and_orphan(tmp_path):
    store = JobStore(tmp_path / 'private')
    app = TestClient(create_callback_app(store, uid='uid', seed=SEED))
    content = json.dumps({'batch_id': 'other', 'extract_result': [
        {'state': 'done', 'data_id': 'x'*64, 'full_zip_url': ZIP_URL}]})
    payload = {'content': content, 'checksum': hashlib.sha256(('uid' + SEED + content).encode()).hexdigest()}
    assert verify_callback(payload, uid='uid', seed=SEED)['batch_id'] == 'other'
    assert app.post('/callbacks/mineru', json=payload).status_code == 400
    assert app.post('/callbacks/mineru', content=b'a'*71000, headers={'content-type': 'application/json'}).status_code == 413
    store.close()
