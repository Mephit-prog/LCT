"""Killable persistent vision process contracts; no torch, weights or network."""
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from wineid.api import create_app
from wineid.catalog import Wine
from wineid.clip_zero_shot import EmbeddingIndex, SCHEMA
from wineid.fusion_vision import VisionSources, load_vision
from wineid.fusion_policy import build_fusion_policy
from wineid.ocr import MockOCR
from wineid.pipeline import Pipeline
from wineid.jina_clip import MODEL, REVISION, CODE_REVISION, PREPROCESS
from wineid.search import Index
from wineid.text_policy import load_policy
from wineid.vision_ipc import IsolatedVisionEncoder, VisionWorkerError

CONFIG = {'model': 'fixture', 'revision': 'fixture', 'preprocess': 'fixture',
          'dimensions': 2}


def picture():
    buf = io.BytesIO()
    Image.new('RGB', (60, 60), 'white').save(buf, 'PNG')
    return buf.getvalue()


@pytest.fixture
def fake(tmp_path):
    script = tmp_path / 'vision.py'
    script.write_text('''import json, os, subprocess, sys, time
mode, marker, child = sys.argv[1:]
if mode == 'startup_hang':
    time.sleep(30)
config = {'model': 'fixture', 'revision': 'fixture', 'preprocess': 'fixture', 'dimensions': 2}
if mode == 'bad_config':
    config['dimensions'] = 99
print(json.dumps({'ready': True, 'config': config, 'device': 'cpu'}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    assert set(request) == {'images'}
    if mode == 'crash':
        sys.exit(1)
    if mode in ('hang', 'once') and (mode == 'hang' or not os.path.exists(marker)):
        open(marker, 'w').close()
        p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        open(child, 'w').write(str(p.pid))
        time.sleep(30)
    if mode == 'bad_vectors':
        print(json.dumps({'vectors': [[float('nan'), 0]]}), flush=True)
    else:
        print(json.dumps({'vectors': [[1., 0.]] * len(request['images'])}), flush=True)
''')
    def make(mode):
        return IsolatedVisionEncoder(command=[sys.executable, str(script), mode,
                        str(tmp_path / 'marker'), str(tmp_path / 'child')],
                        expected_config=CONFIG, startup_timeout=2)
    return make, tmp_path


def build_pipe(encoder):
    wines = [Wine(s, s, s, 'Дом', s, 'дом', (1,), {}, 'sha') for s in ('a', 'b')]
    classes = EmbeddingIndex(np.array([[1., 0.], [0., 1.]], dtype='float32'),
        {'schema': SCHEMA, 'kind': 'classes', 'encoder': CONFIG,
         'prompt_variant': 'title', 'ensemble': False, 'source_sha256': 'sha',
         'items': [{'id': w.slug, 'prompts': [w.name]} for w in wines],
         'prompt_collisions': {}})
    vision = VisionSources(encoder, wines, classes=classes)
    return Pipeline(Index(wines), vision=vision, csv_sha256='sha')


def test_real_worker_protocol_with_fake_encoder(tmp_path):
    # Exercise actual decode/canary/frame loop without importing torch/remote code.
    script = tmp_path / 'entry.py'
    script.write_text('''import numpy as np
from wineid.vision_worker import serve
class Encoder:
    config = {'model': 'fixture', 'revision': 'fixture', 'preprocess': 'fixture', 'dimensions': 2}
    device = 'cpu'
    def encode_images(self, images):
        assert all(image.mode == 'RGB' for image in images)
        return np.tile([[1., 0.]], (len(images), 1))
    def close(self):
        pass
serve(Encoder())
''')
    worker = IsolatedVisionEncoder(command=[sys.executable, '-c', script.read_text()],
                                   expected_config=CONFIG, startup_timeout=2)
    try:
        worker.start()
        assert worker.encode_images_deadline([Image.new('RGB', (9, 9))],
                                             time.monotonic() + 2).tolist() == [[1., 0.]]
    finally:
        worker.close()


def test_old_fusion_policy_cannot_be_relabelled(tmp_path):
    artifact = tmp_path / 'old-policy.json'
    artifact.write_text(json.dumps({'version': 'fusion-policy-v3'}))
    with pytest.raises(ValueError, match='obsolete policy schema'):
        load_policy(artifact, allow_legacy=False)


def test_load_vision_uses_pinned_isolated_worker_not_parent_weights(tmp_path):
    config = {'model': MODEL, 'revision': REVISION, 'code_revision': CODE_REVISION,
              'preprocess': PREPROCESS, 'text_task': None, 'normalize': 'l2',
              'dimensions': 1024}
    wines = [Wine('a', 'a', 'a', 'Дом', 'a', 'дом', (1,), {}, 'sha')]
    classes = EmbeddingIndex(np.eye(1, 1024, dtype='float32'),
        {'schema': SCHEMA, 'kind': 'classes', 'encoder': config,
         'prompt_variant': 'title', 'ensemble': False, 'source_sha256': 'sha',
         'items': [{'id': 'a', 'prompts': ['a']}], 'prompt_collisions': {}})
    path = tmp_path / 'classes.npz'
    classes.save(path)
    made = []
    class FakeEncoder:
        model = True
        device = 'cpu'
        def __init__(self, **kwargs):
            self.config = kwargs['expected_config']
            made.append(self)
        def start(self):
            self.started = True
        def close(self):
            self.closed = True
        def encode_images_deadline(self, images, deadline):
            return np.eye(len(images), 1024, dtype='float32')
    with patch('wineid.vision_ipc.IsolatedVisionEncoder', FakeEncoder):
        vision = load_vision(wines, classes_path=path, allow_remote_code=True,
                             isolated=True)
    try:
        assert made[0].started and vision.ready
        assert vision.observe(Image.new('RGB', (20, 20)), time.monotonic() + 2)[0].slug == 'a'
    finally:
        vision.close()
    assert made[0].closed


def test_isolated_encoder_http_and_lifecycle(fake):
    make, _ = fake
    worker = make('ok')
    try:
        worker.start()
        pid = worker._process.pid
        pipe = build_pipe(worker)
        client = TestClient(create_app(pipe, request_timeout=2))
        assert client.get('/api/health/recognition').status_code == 200
        result = client.post('/api/recognize', files={'image': ('photo.png', picture())})
        assert result.status_code == 200
        assert result.json()['candidates'][0]['slug'] == 'a'
        assert result.json()['slug'] is None  # no validation policy
        assert client.post('/v1/eval/predict', files={'image': ('photo.png', picture())}).json() == {'slug': 'a'}
        assert worker._process.pid == pid  # weights loaded once, not per request
    finally:
        worker.close()
    assert not worker.ready


def test_hang_kills_process_group_and_recovers_off_request(fake):
    make, root = fake
    worker = make('once')
    try:
        worker.start()
        pipe = build_pipe(worker)
        client = TestClient(create_app(pipe, request_timeout=.16))
        old_pid = worker._process.pid
        start = time.monotonic()
        response = client.post('/api/recognize', files={'image': ('p.png', picture())})
        assert time.monotonic() - start < (6 if os.name == 'nt' else .8)
        assert response.json()['slug'] is None
        assert response.json()['reason_codes'] == ['deadline']
        assert not worker.ready
        assert client.get('/api/health/recognition').status_code == 503
        assert client.post('/v1/eval/predict', files={'image': ('p.png', picture())}).status_code == 503
        end = time.monotonic() + 4
        while not worker.ready and time.monotonic() < end:
            time.sleep(.05)
        assert worker.ready and worker._process.pid != old_pid
        child = int((root / 'child').read_text())
        # Gone or a not-yet-reaped zombie on Linux; never a runnable child.
        if os.name == 'nt':
            state = subprocess.run(['tasklist', '/FI', f'PID eq {child}'],
                                   capture_output=True, text=True).stdout
            assert str(child) not in state
        else:
            state = subprocess.run(['ps', '-o', 'stat=', '-p', str(child)],
                                   capture_output=True, text=True).stdout.strip()
            assert not state or state.startswith('Z')
        while time.monotonic() < end:
            if client.post('/v1/eval/predict', files={'image': ('p.png', picture())}).json() == {'slug': 'a'}:
                break
            time.sleep(.05)
        else:
            pytest.fail('recovered worker could not serve requests')
    finally:
        worker.close()


@pytest.mark.parametrize('mode', ['crash', 'bad_vectors'])
def test_failed_worker_returns_503_without_forced_fallback(fake, mode):
    worker = fake[0](mode)
    try:
        worker.start()
        client = TestClient(create_app(build_pipe(worker), request_timeout=2))
        response = client.post('/api/recognize', files={'image': ('p.png', picture())})
        assert response.status_code == 503
        assert response.json()['reason_codes'] == ['vision_unavailable']
        assert client.post('/v1/eval/predict', files={'image': ('p.png', picture())}).status_code == 503
    finally:
        worker.close()


def test_failed_vision_does_not_accept_text_only_under_fusion_policy(fake):
    make, root = fake
    worker = make('crash')
    try:
        worker.start()
        source = build_pipe(worker)
        base = Pipeline(source.index, MockOCR('a'), csv_sha256='sha',
                        roi_mode='center_80_crop', vision=source.vision)
        manifest = root / 'validation.jsonl'
        data = picture()
        (root / 'photo.png').write_bytes(data)
        manifest.write_text(json.dumps({'query_id': 'q', 'image_path': 'photo.png',
            'image_sha256': hashlib.sha256(data).hexdigest(), 'source_group': 'shoot-1',
            'split': 'validation', 'gt_status': 'known', 'gt_slug': 'a',
            'gt_source': 'annotator'}) + '\n')
        policy = build_fusion_policy(manifest, wines=base.index.wines,
            config=base.fusion_config, min_cosine=0, min_text_score=0, min_margin=0)
        assert policy.config['vision_executor'] == 'isolated-v1'
        with pytest.raises(ValueError, match='configuration mismatch'):
            policy.check_config({**base.fusion_config, 'vision_executor': 'inprocess-v1'})
        pipe = Pipeline(base.index, MockOCR('a'), policy, 'sha', 'center_80_crop',
                        vision=base.vision)
        result = pipe.predict(data)
        assert result['candidates'] and result['slug'] is None
        assert result['status'] == 'ambiguous'
        assert 'partial_evidence' in result['reason_codes']
        assert result['reason_codes'][0] == 'vision_worker_failed'
    finally:
        worker.close()


def test_shutdown_interrupts_hung_startup(fake):
    worker = fake[0]('startup_hang')
    errors = []
    thread = threading.Thread(target=lambda: _start_catching(worker, errors))
    thread.start()
    end = time.monotonic() + 2
    while worker._process is None and time.monotonic() < end:
        time.sleep(.01)
    assert worker._process is not None
    start = time.monotonic()
    worker.close()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert time.monotonic() - start < .8
    assert errors and not worker.ready


def _start_catching(worker, errors):
    try:
        worker.start()
    except VisionWorkerError as exc:
        errors.append(str(exc))


def test_startup_mismatch_and_fork_refusal(fake):
    worker = fake[0]('bad_config')
    try:
        with pytest.raises(VisionWorkerError, match='incompatible'):
            worker.start()
        assert not worker.ready
    finally:
        worker.close()
    worker = fake[0]('ok')
    try:
        worker.start()
        worker._owner = -1  # simulate a fork; must not operate on the parent's pipes
        with pytest.raises(VisionWorkerError, match='unavailable'):
            worker.encode_images_deadline([Image.new('RGB', (5, 5))], time.monotonic() + 1)
    finally:
        worker._owner = os.getpid()
        worker.close()


def test_large_frame_is_downscaled_for_vision_not_refused(fake):
    # Phone photos exceed 6 MP; Jina resizes to 512 px anyway, so the worker
    # must get a downscaled copy instead of answering vision_input_too_large.
    make, _ = fake
    worker = make('ok')
    try:
        worker.start()
        big = Image.new('RGB', (4000, 2000), 'white')
        vectors = worker.encode_images_deadline([big], time.monotonic() + 5)
        assert vectors.shape == (1, 2)
        assert big.size == (4000, 2000)  # caller's image untouched
        buf = io.BytesIO()
        big.save(buf, 'JPEG')
        client = TestClient(create_app(build_pipe(worker)))
        response = client.post('/v1/eval/predict', files={'image': ('big.jpg', buf.getvalue())})
        assert response.status_code == 200, response.json()
        assert response.json()['slug'] in ('a', 'b')
    finally:
        worker.close()
