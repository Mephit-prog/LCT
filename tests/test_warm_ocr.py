"""Offline IPC/HTTP integration contracts. No Paddle weights or network required."""
import asyncio
import hashlib
import io
import json
import os
import subprocess
import sys
import threading
import time

import pytest
from PIL import Image
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from wineid.api import MAX_BODY, create_app
from wineid.catalog import Wine
from wineid.local_ocr import LocalOCR, LocalFailure, _execute
from wineid.ocr import MockOCR, configured_ocr
from wineid.pipeline import Pipeline
from wineid.roi import full_frame_roi
from wineid.search import Index
from wineid.warm_ocr import WarmPaddleOCR, verify_manifest


def wait_until(predicate, seconds=4):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(.02)
    return predicate()


def crop():
    buf = io.BytesIO()
    Image.new('RGB', (160, 80), 'white').save(buf, 'PNG')
    return full_frame_roi(buf.getvalue(), time.monotonic() + 5)


@pytest.fixture
def fake_worker(tmp_path):
    script = tmp_path / 'worker.py'
    script.write_text('''import json, sys, time
mode = sys.argv[1]
print(json.dumps({'ready': True}), flush=True)
for line in sys.stdin:
    if mode in ('slow', 'once_slow'):
        time.sleep(1)
        mode = 'ok'
    if mode == 'crash':
        sys.exit(1)
    if mode == 'invalid':
        print('not json', flush=True)
    elif mode == 'huge':
        print('x' * (2 * 1024 * 1024 + 2), flush=True)
    else:
        request = json.loads(line)
        assert request['op'] == 'recognize'
        print(json.dumps({'text': 'Алиготе 2024'}), flush=True)
''', encoding='utf-8')
    return lambda mode: WarmPaddleOCR(command=[sys.executable, str(script), mode], startup_timeout=2)


def test_warm_pipeline_and_restart(fake_worker):
    worker = fake_worker('ok')
    try:
        assert not worker.ready
        worker.start()
        assert worker.ready
        wine = Wine('a', 'a', 'Алиготе 2024', 'Завод', 'алиготе 2024', 'завод',
                    (2,), {}, 'sha')
        pipe = Pipeline(Index([wine]), worker, csv_sha256='sha',
                        roi_mode='center_80_crop')
        assert pipe.recognition_ready
        buf = io.BytesIO()
        Image.new('RGB', (160, 80), 'white').save(buf, 'PNG')
        result = pipe.predict(buf.getvalue())
        assert result['ocr']['raw_text'] == 'Алиготе 2024'
        assert result['ocr']['normalized_text'] == 'алиготе 2024'
        assert result['candidates'][0]['slug'] == 'a'
        assert result['slug'] is None  # no calibrated acceptance policy
        worker._process.kill()
        worker._process.wait()
        assert not pipe.recognition_ready
        http = TestClient(create_app(pipe, request_timeout=3))
        failed = http.post('/api/recognize', files={'image': ('a.png', buf.getvalue())})
        assert failed.status_code == 503
        assert failed.json()['reason_codes'] == ['ocr_unavailable']
        forced = http.post('/v1/eval/predict', files={'image': ('a.png', buf.getvalue())})
        assert forced.status_code == 503 and forced.json()['slug'] is None
        before = time.monotonic()
        unavailable = worker.recognize(crop(), before + 3)
        assert (unavailable.status, unavailable.error_code) == ('provider_error', 'worker_unavailable')
        assert time.monotonic() - before < .5  # never hashes/reloads weights in HTTP
        assert wait_until(lambda: worker.ready and not worker._recovery.is_alive())
        assert worker.recognize(crop(), time.monotonic() + 3).status == 'ok'
        assert pipe.recognition_ready
    finally:
        worker.close()


@pytest.mark.parametrize('mode,code', [('crash', 'worker_failed'), ('invalid', 'worker_failed'),
                                      ('huge', 'worker_failed')])
def test_bad_worker_is_killed(fake_worker, mode, code):
    worker = fake_worker(mode)
    try:
        worker.start()
        result = worker.recognize(crop(), time.monotonic() + 2)
        assert (result.status, result.error_code) == ('provider_error', code)
        assert not worker.ready
        assert result.raw_text is None
    finally:
        worker.close()


def test_timeout_kills_and_recovers_off_request(fake_worker):
    worker = fake_worker('slow')
    try:
        worker.start()
        before = time.monotonic()
        result = worker.recognize(crop(), before + .1)
        assert (result.status, result.error_code) == ('timeout', 'deadline')
        assert time.monotonic() - before < .8
        assert not worker.ready
        assert worker._process is None
        # Recovery is delayed and bounded; short requests never wait for canary.
        assert worker.recognize(crop(), time.monotonic() + .05).error_code == 'worker_unavailable'
    finally:
        worker.close()


def test_warm_rejects_parallel_jobs(fake_worker):
    worker = fake_worker('slow')
    result = []
    try:
        worker.start()
        thread = threading.Thread(target=lambda: result.append(
            worker.recognize(crop(), time.monotonic() + 3)))
        thread.start()
        time.sleep(.1)
        busy = worker.recognize(crop(), time.monotonic() + 2)
        thread.join(3)
        assert (busy.status, busy.error_code) == ('provider_error', 'busy')
        assert result[0].status == 'ok'
    finally:
        worker.close()


def test_no_implicit_remote_or_mock_and_manifest(tmp_path):
    assert configured_ocr({'MISTRAL_API_KEY': 'secret', 'WINE_MOCK_OCR': 'hi'}) is None
    with pytest.raises(ValueError):
        configured_ocr({'WINE_OCR_PROVIDER': 'mock', 'WINE_MOCK_OCR': 'hi'})
    with pytest.raises(ValueError):
        configured_ocr({'WINE_OCR_PROVIDER': 'paddle', 'WINE_OCR_MODE': 'warm'})
    roots = [tmp_path / name for name in ('det', 'rec')]
    for root in roots:
        root.mkdir()
        (root / 'model.bin').write_bytes(b'model')
    digest = hashlib.sha256(b'model').hexdigest()
    manifest = {'format': 'wineid-paddle-v1', 'ocr_version': 'PP-OCRv5', 'lang': 'ru',
                'packages': {'paddleocr': '3.7.0', 'paddlepaddle': '3.3.1'},
                'det': {'model.bin': digest}, 'rec': {'model.bin': digest}}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    assert verify_manifest(path, *roots)[0] == manifest
    from wineid import ocr_manifest
    import unittest.mock
    with unittest.mock.patch.object(ocr_manifest.importlib.metadata, 'version',
                                    side_effect=lambda name: manifest['packages'][name]):
        assert ocr_manifest.build(*roots) == manifest
    (roots[1] / 'model.bin').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='hash mismatch'):
        verify_manifest(path, *roots)


def test_manifest_cannot_change_on_worker_restart(fake_worker, tmp_path):
    det, rec = tmp_path / 'det', tmp_path / 'rec'
    det.mkdir(); rec.mkdir()
    for root in (det, rec):
        (root / 'weights').write_bytes(b'old')
    digest = hashlib.sha256(b'old').hexdigest()
    model = tmp_path / 'manifest.json'
    manifest = {'format': 'wineid-paddle-v1', 'ocr_version': 'PP-OCRv5', 'lang': 'ru',
                'packages': {'paddleocr': '3.7.0', 'paddlepaddle': '3.3.1'},
                'det': {'weights': digest}, 'rec': {'weights': digest}}
    model.write_text(json.dumps(manifest))
    worker = fake_worker('ok')
    worker.manifest = str(model)
    worker.env['WINE_PADDLE_DET_DIR'] = str(det)
    worker.env['WINE_PADDLE_REC_DIR'] = str(rec)
    try:
        worker.start()
        worker._process.kill()
        worker._process.wait()
        model.write_text(json.dumps(manifest, indent=2))  # valid but new manifest SHA
        result = worker.recognize(crop(), time.monotonic() + 2)
        assert (result.status, result.error_code) == ('provider_error', 'worker_unavailable')
        assert wait_until(lambda: worker._incompatible and not worker._recovery.is_alive())
        assert not worker.ready
        assert worker._process is None
        # No repeated hashing/respawning on every following request.
        assert worker.recognize(crop(), time.monotonic() + 1).error_code == 'worker_unavailable'
        assert worker._recovery is not None and not worker._recovery.is_alive()
    finally:
        worker.close()


def test_slow_manifest_verification_and_canary_never_run_in_request(fake_worker, monkeypatch):
    import wineid.warm_ocr as ipc
    monkeypatch.setattr(ipc, 'RECOVERY_DELAY', .02)
    worker = fake_worker('ok')
    entered, release = threading.Event(), threading.Event()
    try:
        worker.start()
        original = worker._verify
        def slow_verify():
            entered.set()
            assert release.wait(2)
            original()
        monkeypatch.setattr(worker, '_verify', slow_verify)
        worker._process.kill()
        worker._process.wait()
        start = time.monotonic()
        result = worker.recognize(crop(), start + .15)
        assert (result.status, result.error_code) == ('provider_error', 'worker_unavailable')
        assert time.monotonic() - start < .15
        assert entered.wait(2)
        start = time.monotonic()
        assert worker.recognize(crop(), start + .1).error_code == 'busy'
        assert time.monotonic() - start < .1
        release.set()
        assert wait_until(lambda: worker.ready and not worker._recovery.is_alive())
        assert worker.recognize(crop(), time.monotonic() + 2).status == 'ok'
    finally:
        release.set()
        worker.close()


def test_background_retries_are_bounded(fake_worker, monkeypatch):
    import wineid.warm_ocr as ipc
    monkeypatch.setattr(ipc, 'RECOVERY_DELAY', .01)
    worker = fake_worker('ok')
    try:
        worker.start()
        worker.startup_timeout = .04
        worker.command = [sys.executable, '-c', 'import time; time.sleep(30)']
        worker._process.kill()
        worker._process.wait()
        assert worker.recognize(crop(), time.monotonic() + 1).error_code == 'worker_unavailable'
        assert wait_until(lambda: worker._retry_after > time.monotonic()
                          and not worker._recovery.is_alive())
        thread = worker._recovery
        assert worker._process is None and not worker.ready
        assert worker.recognize(crop(), time.monotonic() + 1).error_code == 'worker_unavailable'
        assert worker._recovery is thread  # failed canary cannot trigger a restart storm
    finally:
        worker.close()


def test_timeout_kills_warm_worker_descendants(tmp_path):
    marker = tmp_path / 'survivor'
    script = tmp_path / 'descendant.py'
    script.write_text('''import json, subprocess, sys, time
print(json.dumps({'ready': True}), flush=True)
for line in sys.stdin:
    subprocess.Popen([sys.executable, '-c',
        'import pathlib, time; time.sleep(.5); pathlib.Path(%r).touch()' % sys.argv[1]])
    time.sleep(30)
''')
    worker = WarmPaddleOCR(command=[sys.executable, str(script), str(marker)], startup_timeout=2)
    try:
        worker.start()
        result = worker.recognize(crop(), time.monotonic() + .2)
        assert (result.status, result.error_code) == ('timeout', 'deadline')
        time.sleep(.6)
        assert not marker.exists()  # process group was killed, not just its parent
    finally:
        worker.close()


def test_shutdown_interrupts_hung_inference(fake_worker):
    worker = fake_worker('slow')
    result = []
    worker.start()
    try:
        thread = threading.Thread(target=lambda: result.append(
            worker.recognize(crop(), time.monotonic() + 20)))
        thread.start()
        time.sleep(.05)
        before = time.monotonic()
        worker.close()
        thread.join(timeout=1)
        assert not thread.is_alive() and time.monotonic() - before < .8
        assert not worker.ready and result[0].status == 'provider_error'
        assert worker.recognize(crop(), time.monotonic() + 1).error_code == 'worker_unavailable'
    finally:
        worker.close()


def test_real_worker_protocol_with_fake_paddle_package(tmp_path, monkeypatch):
    # Exercises the actual worker module/canary/protocol without weights/network.
    import pathlib
    fake = tmp_path / 'packages'
    fake.mkdir()
    (fake / 'paddleocr.py').write_text('''class PaddleOCR:
    def __init__(self, **kwargs):
        assert kwargs['lang'] == 'ru'
        assert kwargs['ocr_version'] == 'PP-OCRv5'
    def predict(self, path):
        return [type('Page', (), {'json': {'res': {'rec_texts': ['Саперави 2024']}}})()]
''', encoding='utf-8')
    for package, version in [('paddleocr', '3.7.0'), ('paddlepaddle', '3.3.1')]:
        dist = fake / f'{package}-{version}.dist-info'
        dist.mkdir()
        (dist / 'METADATA').write_text(f'Metadata-Version: 2.1\nName: {package}\nVersion: {version}\n')
    roots = [tmp_path / name for name in ('det', 'rec')]
    for root in roots:
        root.mkdir()
        (root / 'weights.bin').write_bytes(b'fixture weights')
    digest = hashlib.sha256(b'fixture weights').hexdigest()
    manifest = tmp_path / 'model.json'
    manifest.write_text(json.dumps({'format': 'wineid-paddle-v1', 'ocr_version': 'PP-OCRv5',
        'lang': 'ru', 'packages': {'paddleocr': '3.7.0', 'paddlepaddle': '3.3.1'},
        'det': {'weights.bin': digest}, 'rec': {'weights.bin': digest}}))
    monkeypatch.setenv('PYTHONPATH', str(fake) + os.pathsep + str(pathlib.Path(__file__).resolve().parents[1]))
    worker = WarmPaddleOCR(python=sys.executable, det_dir=str(roots[0]),
                           rec_dir=str(roots[1]), manifest=str(manifest), startup_timeout=3)
    try:
        worker.start()
        result = worker.recognize(crop(), time.monotonic() + 3)
        assert (result.status, result.raw_text) == ('ok', 'Саперави 2024')
        assert result.model.endswith(hashlib.sha256(manifest.read_bytes()).hexdigest())
    finally:
        worker.close()


def test_cold_missing_cli_and_bounded_output(tmp_path):
    ocr = LocalOCR('tesseract', tesseract=str(tmp_path / 'no-tesseract'))
    wine = Wine('a', 'a', 'Алиготе', 'Завод', 'алиготе', 'завод', (2,), {}, 'sha')
    pipe = Pipeline(Index([wine]), ocr, csv_sha256='sha', roi_mode='center_80_crop')
    assert not ocr.ready
    health = TestClient(create_app(pipe)).get('/api/health').json()
    assert health['catalog_ready'] and health['ocr_configured']
    assert not health['ocr_ready'] and not health['recognition_ready']
    result = ocr.recognize(crop(), time.monotonic() + 2)
    assert (result.status, result.error_code) == ('provider_error', 'engine_missing')
    with pytest.raises(RuntimeError, match='canary failed'):
        ocr.start()
    script = tmp_path / 'big.py'
    script.write_text('import sys; sys.stdout.write("x" * (3 * 1024 * 1024))')
    with pytest.raises(LocalFailure, match='output_too_large'):
        _execute([sys.executable, str(script)], None, time.monotonic() + 3)


def test_ocr_capacity_maps_to_http_busy():
    from wineid.ocr import OCRResult
    from wineid.blend import Blender
    class BusyOCR:
        ready = True
        model = 'fixture'
        def recognize(self, roi, deadline):
            return OCRResult(roi.roi_id, 'provider_error', None, None,
                             self.model, 0, 'busy')
    wine = Wine('a', 'a', 'Алиготе', 'Завод', 'алиготе', 'завод', (2,), {}, 'sha')
    index = Index([wine])
    image = io.BytesIO()
    Image.new('RGB', (160, 80), 'white').save(image, 'PNG')
    for blender in (None, Blender(index.wines)):
        pipe = Pipeline(index, BusyOCR(), csv_sha256='sha', roi_mode='center_80_crop',
                        blender=blender)
        response = TestClient(create_app(pipe)).post('/api/recognize',
                       files={'image': ('sample.png', image.getvalue())})
        assert response.status_code == 503
        assert response.json()['reason_codes'] == ['busy']


def test_fastapi_chunked_limit_and_upload_deadline():
    wine = Wine('a', 'a', 'Алиготе', 'Завод', 'алиготе', 'завод', (2,), {}, 'sha')
    app = create_app(Pipeline(Index([wine]), MockOCR('Алиготе'), csv_sha256='sha'),
                     request_timeout=.06)
    async def run():
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
            async def too_big():
                yield b'x' * (MAX_BODY // 2)
                yield b'x' * (MAX_BODY // 2 + 1)
            big = await client.post('/api/recognize', content=too_big(),
                                    headers={'content-type': 'application/octet-stream'})
            assert big.status_code == 413
            assert big.json()['reason_codes'] == ['invalid_body_size']
            async def slow():
                yield b'a'
                await asyncio.sleep(.15)
                yield b'b'
            late = await client.post('/api/recognize', content=slow(),
                                     headers={'content-type': 'application/octet-stream'})
            assert late.status_code == 408
            assert late.json()['reason_codes'] == ['deadline']
    asyncio.run(run())


def test_blended_vision_cannot_accept_after_deadline():
    class SlowVision:
        classes = gallery = None
        class Encoder:
            config = {'fixture': True}
        encoder = Encoder()
        def observe(self, image, deadline):
            time.sleep(.08)
            return []
    wine = Wine('a', 'a', 'Алиготе', 'Завод', 'алиготе', 'завод', (2,), {}, 'sha')
    pipe = Pipeline(Index([wine]), MockOCR('Алиготе'), csv_sha256='sha',
                    vision=SlowVision(), roi_mode='center_80_crop')
    buf = io.BytesIO()
    Image.new('RGB', (160, 80), 'white').save(buf, 'PNG')
    result = pipe.predict(buf.getvalue(), seconds=.04, force_top1=True)
    assert result['slug'] is None
    assert result['status'] == 'error'
    assert 'deadline' in result['reason_codes']


def test_cold_timeout_kills_descendants(tmp_path):
    marker = tmp_path / 'survivor'
    script = tmp_path / 'descendants.py'
    script.write_text('''import subprocess, sys, time
subprocess.Popen([sys.executable, '-c',
    'import time, pathlib; time.sleep(0.5); pathlib.Path(%r).touch()' % sys.argv[1]])
time.sleep(2)
''')
    with pytest.raises(subprocess.TimeoutExpired):
        _execute([sys.executable, str(script), str(marker)], None, time.monotonic() + .15)
    time.sleep(.55)
    assert not marker.exists()
