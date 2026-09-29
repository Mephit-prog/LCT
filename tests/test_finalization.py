"""PR-03/05 contracts: one observation path; admission before upload parsing.

All fixtures are synthetic; no OCR/CLIP weights, MinerU API or network required.
"""
import asyncio
import hashlib
import io
import json
import socket
import threading

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from wineid.api import create_app
from wineid.blend import Blender, CandidateEvidence
from wineid.catalog import Wine, normalize
from wineid.fusion_eval import evaluate, verify_runtime_evidence
from wineid.fusion_policy import build_fusion_policy
from wineid.fusion_run import observe_queries
from wineid.ocr import OCRResult
from wineid.pipeline import Pipeline
from wineid.release_bundle import build, verify
from wineid.runtime import open_runtime
from wineid.search import Index
from wineid.server import BoundedHTTPServer, create_handler


def photo():
    stream = io.BytesIO()
    Image.new('RGB', (120, 140), 'white').save(stream, 'PNG')
    return stream.getvalue()


def catalog():
    return [Wine(slug, slug, name, 'Дом', normalize(name), normalize('Дом'),
                 (1,), {'Название вина': name, 'Винодельня': 'Дом'}, 'fixture')
            for slug, name in [('first', 'Красное'), ('second', 'Белое')]]


class Vision:
    classes = gallery = None
    class_fingerprint = 'fixture'
    query_views = 1

    class Encoder:
        config = {'fixture': True}
    encoder = Encoder()

    def __init__(self):
        self.failure = None
        self.sizes = []

    def observe(self, image, deadline):
        self.sizes.append(image.size)
        if self.failure:
            raise self.failure
        return [CandidateEvidence('second', 'class', .9, 1, 'cosine', 'fixture')]


class OCR:
    model = 'fixture'
    ready = True

    def __init__(self):
        self.status = 'ok'

    def recognize(self, roi, deadline):
        return OCRResult(roi.roi_id, self.status,
                         'Красное Дом' if self.status == 'ok' else None,
                         'красное дом' if self.status == 'ok' else None,
                         self.model, 0, 'bad_engine' if self.status == 'provider_error' else None)


def manifest(tmp_path, image):
    (tmp_path / 'q.png').write_bytes(image)
    row = {'query_id': 'q', 'image_path': 'q.png',
           'image_sha256': hashlib.sha256(image).hexdigest(), 'split': 'validation',
           'gt_status': 'known', 'gt_slug': 'second', 'source_group': 'human',
           'gt_source': 'annotator'}
    (tmp_path / 'queries.jsonl').write_text(json.dumps(row) + '\n')
    return row


@pytest.mark.parametrize('ocr_status', ['ok', 'no_text', 'provider_error'])
def test_offline_http_evidence_parity(tmp_path, ocr_status):
    data = photo()
    row = manifest(tmp_path, data)
    wines = catalog()
    index = Index(wines)
    vision, ocr = Vision(), OCR()
    ocr.status = ocr_status
    blender = Blender(wines)
    pipe = Pipeline(index, ocr, csv_sha256='fixture', vision=vision, blender=blender,
                    roi_mode='center_80_crop')
    offline = observe_queries([row], root=tmp_path, index=index, blender=blender,
        vision=vision, ocr=ocr, ocr_mode='live', roi_mode='center_80_crop',
        seconds=pipe.request_timeout, runtime_profile=pipe.fusion_config, pipeline=pipe)[0]
    online = pipe.predict(data, _include_evidence=True)
    response = TestClient(create_app(pipe)).post('/api/recognize',
        files={'image': ('q.png', data)})
    assert response.status_code == 200
    http = response.json()
    assert offline['evidence'] == online['_evidence']
    assert offline['ocr_status'] == ocr_status
    assert offline['online_status'] == online['status'] == http['status']
    assert offline['reason_codes'] == online['reason_codes'] == http['reason_codes']
    assert offline['roi'] == online['roi']
    assert json.loads(json.dumps(online['roi'])) == http['roi']
    assert [r['slug'] for r in offline['top_candidates']] == [r['slug'] for r in http['candidates']]
    assert offline['fusion_config_sha256'] is not None
    # A manual bbox is a different request profile; both implementations crop
    # vision too, and must not present its digest as calibrated evidence.
    boxed = {**row, 'bbox': [10, 20, 80, 100]}
    cropped = observe_queries([boxed], root=tmp_path, index=index, blender=blender,
        vision=vision, ocr=ocr, ocr_mode='live', roi_mode='center_80_crop',
        seconds=pipe.request_timeout, runtime_profile=pipe.fusion_config, pipeline=pipe)[0]
    assert cropped['fusion_config_sha256'] is None
    assert vision.sizes[-1] != (120, 140)
    assert cropped['roi']['bbox'] == pipe.predict(data, bbox=boxed['bbox'])['roi']['bbox']


def test_timeout_and_capacity_not_accepted(tmp_path):
    data = photo()
    row = manifest(tmp_path, data)
    wines, vision = catalog(), Vision()
    index = Index(wines)
    pipe = Pipeline(index, csv_sha256='fixture', vision=vision)
    blender = pipe.blender
    vision.failure = TimeoutError('deadline')
    offline = observe_queries([row], root=tmp_path, index=index, blender=blender,
        vision=vision, seconds=8.5, pipeline=pipe)[0]
    assert offline['online_status'] == 'error'
    assert offline['reason_codes'] == pipe.predict(data)['reason_codes'] == ['deadline']
    assert offline['top_candidates'] == []
    vision.failure = RuntimeError('vision_capacity_exceeded')
    assert TestClient(create_app(pipe)).post('/api/recognize',
        files={'image': ('q.png', data)}).status_code == 503
    assert 'vision_capacity_exceeded' in pipe.predict(data)['reason_codes']


def test_policy_reinference_rejects_forged_scores(tmp_path):
    data = photo()
    row = manifest(tmp_path, data)
    wines, vision = catalog(), Vision()
    index = Index(wines)
    base = Pipeline(index, csv_sha256='fixture', vision=vision, roi_mode='refuse')
    policy = build_fusion_policy(tmp_path / 'queries.jsonl', wines=wines,
        config=base.fusion_config, min_cosine=0, min_text_score=0, min_margin=0)
    pipe = Pipeline(index, policy=policy, csv_sha256='fixture', vision=vision,
                    blender=base.blender, roi_mode='refuse')
    saved = observe_queries([row], root=tmp_path, index=index, blender=pipe.blender,
        vision=vision, pipeline=pipe, seconds=pipe.request_timeout,
        runtime_profile=pipe.fusion_config)[0]
    verified = verify_runtime_evidence([row], [saved], pipe, root=tmp_path,
                                       seconds=pipe.request_timeout)
    stats = evaluate(wines, [row], verified, policy=policy, pipeline=pipe)
    assert stats['predictions'][0]['status'] == pipe.predict(data)['status']
    forged = {**saved, 'evidence': [{**saved['evidence'][0], 'score': .99}]}
    # Even with the real image and config SHA, an edited observation is rejected.
    for modified in (forged, {**saved, 'image_sha256': '0' * 64},
                     {**saved, 'online_status': 'accepted'}):
        with pytest.raises(ValueError, match='differs'):
            verify_runtime_evidence([row], [modified], pipe, root=tmp_path,
                                    seconds=pipe.request_timeout)
    with pytest.raises(ValueError, match='manual bbox'):
        verify_runtime_evidence([{**row, 'bbox': [1, 1, 50, 50]}], [saved], pipe,
                                root=tmp_path, seconds=pipe.request_timeout)


def test_offline_policy_preserves_capacity_refusal(tmp_path):
    data = photo()
    row = manifest(tmp_path, data)
    wines, vision, ocr = catalog(), Vision(), OCR()
    index = Index(wines)
    base = Pipeline(index, ocr, csv_sha256='fixture', vision=vision,
                    roi_mode='center_80_crop')
    policy = build_fusion_policy(tmp_path / 'queries.jsonl', wines=wines,
        config=base.fusion_config, min_cosine=0, min_text_score=0, min_margin=0)
    pipe = Pipeline(index, ocr, policy, csv_sha256='fixture', vision=vision,
                    blender=base.blender, roi_mode='center_80_crop')
    vision.failure = RuntimeError('vision_capacity_exceeded')
    saved = observe_queries([row], root=tmp_path, index=index, blender=pipe.blender,
        vision=vision, ocr=ocr, ocr_mode='live', roi_mode='center_80_crop',
        seconds=pipe.request_timeout, runtime_profile=pipe.fusion_config, pipeline=pipe)
    assert saved[0]['evidence']  # text search still returned candidates
    assert saved[0]['online_status'] == 'error'
    verified = verify_runtime_evidence([row], saved, pipe, root=tmp_path,
                                       seconds=pipe.request_timeout)
    decision = evaluate(wines, [row], verified, policy=policy, pipeline=pipe)['predictions'][0]
    assert decision['status'] == 'error' and decision['accepted_slug'] is None
    assert TestClient(create_app(pipe)).post('/api/recognize',
        files={'image': ('q.png', data)}).status_code == 503


def test_admission_auth_and_capacity_precede_body_read():
    pipe = Pipeline(Index(catalog()), csv_sha256='fixture')
    app = create_app(pipe, token='secret', max_workers=1, request_timeout=2.)

    async def scenario():
        scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
                 'method': 'POST', 'scheme': 'http', 'path': '/api/search',
                 'query_string': b'', 'client': ('127.0.0.1', 1), 'server': ('test', 80),
                 'headers': [(b'content-type', b'application/json'),
                             (b'authorization', b'Bearer secret')]}
        async def send(message):
            if message['type'] == 'http.response.start':
                statuses.append(message['status'])
        async def never_read():
            raise AssertionError('body read before auth/capacity decision')
        statuses = []
        await app({**scope, 'headers': scope['headers'][:1]}, never_read, send)
        assert statuses == [401]
        entered, release = asyncio.Event(), asyncio.Event()
        async def slow_receive():
            entered.set()
            await release.wait()
            return {'type': 'http.request', 'body': b'{"text":""}', 'more_body': False}
        first = asyncio.create_task(app(scope, slow_receive, send))
        await asyncio.wait_for(entered.wait(), 1)
        await app(scope, never_read, send)
        assert statuses == [401, 503]
        release.set()
        await asyncio.wait_for(first, 2)
        assert statuses[-1] == 200
        async def normal_receive():
            return {'type': 'http.request', 'body': b'{"text":""}', 'more_body': False}
        await app(scope, normal_receive, send)
        assert statuses[-1] == 200
    asyncio.run(scenario())


def test_cors_on_prebody_auth_failure():
    client = TestClient(create_app(Pipeline(Index(catalog()), csv_sha256='fixture'),
                                   token='secret', cors_origins=('http://localhost:5173',)))
    response = client.post('/api/search', json={'text': 'Вино'},
                           headers={'Origin': 'http://localhost:5173'})
    assert response.status_code == 401
    assert response.headers['access-control-allow-origin'] == 'http://localhost:5173'


def test_invalid_multipart_and_search_limit():
    client = TestClient(create_app(Pipeline(Index(catalog()), csv_sha256='fixture')))
    too_many = client.post('/api/recognize', files={'image': ('q.png', photo())},
                           data={'extra': 'ignored?'})
    assert too_many.status_code == 400
    assert too_many.json()['reason_codes'] == ['invalid_multipart']
    large_search = client.post('/api/search', json={'text': 'a' * (16 * 1024)})
    assert large_search.status_code == 413
    assert large_search.json()['reason_codes'] == ['invalid_body_size']


def test_stdlib_unauthorized_upload_is_not_read():
    pipe = Pipeline(Index(catalog()), csv_sha256='fixture')
    server = BoundedHTTPServer(('127.0.0.1', 0), create_handler(pipe, token='secret'),
                               max_workers=1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with socket.create_connection(server.server_address, timeout=2) as client:
            client.settimeout(.5)
            client.sendall(b'POST /v1/recognize HTTP/1.1\r\nHost: test\r\n'
                           b'Content-Length: 12000000\r\nConnection: close\r\n\r\n')
            # Deliberately send NO body. Old behavior held this worker until the
            # upload deadline, potentially starving all authorized traffic.
            assert b' 401 ' in client.recv(4096)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_release_file_inventory_and_preflight(tmp_path):
    (tmp_path / 'catalog.csv').write_bytes(b'catalog-fixture')
    (tmp_path / 'aliases.txt').write_bytes(b'aliases-fixture')
    revision = 'a' * 40
    files = {'csv': 'catalog.csv', 'aliases': 'aliases.txt'}
    inventory = build(tmp_path, files, revision)
    assert verify(tmp_path, inventory) == inventory
    path = tmp_path / 'bundle.json'
    path.write_text(json.dumps(inventory))
    env = {'WINE_CSV': str(tmp_path / 'catalog.csv'),
           'WINE_ALIASES': str(tmp_path / 'aliases.txt'),
           'WINE_RELEASE_MANIFEST': str(path)}
    # The preflight must fail before catalog/model loading, including when the
    # self-described bundle has not been signed by any external channel.
    (tmp_path / 'catalog.csv').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='checksum'):
        with open_runtime(env):
            pass
    (tmp_path / 'catalog.csv').write_bytes(b'catalog-fixture')
    (tmp_path / 'elsewhere.txt').write_bytes(b'aliases-fixture')
    with pytest.raises(ValueError, match='path mismatch'):
        with open_runtime({**env, 'WINE_ALIASES': str(tmp_path / 'elsewhere.txt')}):
            pass
    with pytest.raises(ValueError, match='unsafe bundle path'):
        build(tmp_path, {**files, 'classes': '../outside.npz'}, revision)
    (tmp_path / 'link').symlink_to(tmp_path / 'catalog.csv')
    with pytest.raises(ValueError, match='symlink'):
        build(tmp_path, {**files, 'classes': 'link'}, revision)


def test_separate_catalog_and_recognition_readiness():
    pipe = Pipeline(Index(catalog()), csv_sha256='fixture')
    client = TestClient(create_app(pipe))
    assert client.get('/ready').status_code == 200
    assert client.get('/api/health/recognition').status_code == 503
    assert not client.get('/api/health').json()['acceptance_ready']
