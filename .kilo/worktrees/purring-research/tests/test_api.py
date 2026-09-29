"""FastAPI backend contract tests (frontend integration layer).

These tests use a fake index and MockOCR; they never download model weights and
never call a paid OCR provider. They assert the HTTP contract, auth, error
shaping, CORS and media serving — not recognition quality.
"""
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path

import asyncio

from PIL import Image
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from wineid.api import create_app, load_media_links
from wineid.catalog import Wine, normalize
from wineid.pipeline import Pipeline, Policy
from wineid.ocr import MockOCR, OCRResult
from wineid.search import Index

CSV_SHA = 'csv-sha-test'


def wine(slug, name, producer, **raw):
    row = {'Название вина': name, 'Винодельня': producer}
    row.update(raw)
    return Wine(slug, slug, name, producer, normalize(name), normalize(producer),
                (2,), row, CSV_SHA)


def picture(fmt='WEBP'):
    buffer = io.BytesIO()
    Image.new('RGB', (120, 90), 'white').save(buffer, format=fmt)
    return buffer.getvalue()


WINES = [
    wine('aligote-2021', 'Алиготе', 'Завод А', Категория='Белое',
         **{'Сорт винограда': 'Алиготе'}),
    wine('aligote-2022', 'Алиготе', 'Завод А', Категория='Белое',
         **{'Сорт винограда': 'Алиготе'}),
    wine('kaberne-2020', 'Каберне', 'Завод Б', Категория='Красное',
         **{'Сорт винограда': 'Каберне Совиньон'}),
]


def build_client(**kwargs):
    index = Index(list(WINES))
    pipeline = Pipeline(index, MockOCR('Алиготе'), csv_sha256=CSV_SHA,
                        roi_mode='full_frame_experimental')
    app = create_app(pipeline, **kwargs)
    return TestClient(app)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = build_client()

    def test_health_ready(self):
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body['ready'])
        self.assertTrue(body['catalog_ready'])
        self.assertTrue(body['recognition_ready'])
        self.assertEqual(body['wine_count'], 3)
        self.assertTrue(body['ocr_configured'])
        self.assertEqual(body['roi_mode'], 'full_frame_experimental')
        self.assertFalse(body['policy_configured'])

    def test_recognition_ready_requires_ocr_and_non_refusing_roi(self):
        index = Index(list(WINES))
        pipe = Pipeline(index, MockOCR('Алиготе'), csv_sha256=CSV_SHA,
                        roi_mode='refuse')
        body = TestClient(create_app(pipe)).get('/api/health').json()
        self.assertTrue(body['catalog_ready'])
        self.assertFalse(body['recognition_ready'])
        no_ocr = Pipeline(index, None, csv_sha256=CSV_SHA,
                          roi_mode='full_frame_experimental')
        body = TestClient(create_app(no_ocr)).get('/api/health').json()
        self.assertFalse(body['recognition_ready'])

    def test_legacy_ready(self):
        self.assertEqual(self.client.get('/ready').status_code, 200)

    def test_ready_503_without_matching_catalog(self):
        index = Index([wine('a-1', 'Вино', 'Завод', )])
        pipeline = Pipeline(index, None, csv_sha256='other-sha')
        client = TestClient(create_app(pipeline))
        self.assertEqual(client.get('/ready').status_code, 503)
        self.assertFalse(client.get('/api/health').json()['ready'])

    def test_list_wines_pagination(self):
        response = self.client.get('/api/wines', params={'limit': 2, 'offset': 0})
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body['total'], 3)
        self.assertEqual(body['count'], 2)
        self.assertEqual(body['items'][0]['slug'], 'aligote-2021')

    def test_list_wines_query_and_detail(self):
        response = self.client.get('/api/wines', params={'q': 'каберне'})
        slugs = [item['slug'] for item in response.json()['items']]
        self.assertIn('kaberne-2020', slugs)
        detail = self.client.get('/api/wines/kaberne-2020').json()
        self.assertEqual(detail['name'], 'Каберне')
        self.assertEqual(detail['color'], 'Красное')
        self.assertEqual(detail['grape'], 'каберне совиньон')

    def test_wine_detail_404(self):
        response = self.client.get('/api/wines/does-not-exist')
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()['reason_codes'], ['not_found'])

    def test_search_returns_cards_and_score_not_probability(self):
        response = self.client.post('/api/search', json={'text': 'Алиготе', 'k': 5})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['backend'], Index(list(WINES)).backend)
        self.assertTrue(body['candidates'])
        top = body['candidates'][0]
        self.assertIn('score', top)
        self.assertIn('wine', top)
        self.assertIn(top['slug'], {'aligote-2021', 'aligote-2022'})

    def test_search_empty_text(self):
        response = self.client.post('/api/search', json={'text': '   '})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['candidates'], [])

    def test_search_invalid_vintage_mode(self):
        response = self.client.post('/api/search',
                                    json={'text': 'Алиготе', 'vintage_mode': 'nope'})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()['reason_codes'], ['invalid_vintage_mode'])

    def test_recognize_full_frame_returns_candidates(self):
        response = self.client.post('/api/recognize',
                                    files={'image': ('a.webp', picture())})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn(body['status'], {'ambiguous', 'accepted', 'unknown'})
        self.assertIn('policy_not_calibrated', body['reason_codes'])
        self.assertIn('full_frame_experimental', body['reason_codes'])
        self.assertTrue(body['candidates'])
        self.assertIn('wine', body['candidates'][0])
        self.assertIsNotNone(body['roi'])
        self.assertEqual(body['roi']['status'], 'fallback_bbox')

    def test_recognize_manual_bbox_and_invalid_bbox(self):
        good = self.client.post('/api/recognize',
                                files={'image': ('a.webp', picture())},
                                data={'bbox': '0,0,100,80'})
        self.assertEqual(good.status_code, 200)
        self.assertEqual(good.json()['roi']['status'], 'manual')
        bad = self.client.post('/api/recognize',
                               files={'image': ('a.webp', picture())},
                               data={'bbox': '0,0,100'})
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(bad.json()['reason_codes'], ['invalid_bbox'])

    def test_recognize_center_80_crop_mode(self):
        # Override roi_mode per-request to ТЗ §4.2 central 80% crop
        response = self.client.post('/api/recognize',
                                    files={'image': ('a.webp', picture())},
                                    data={'roi_mode': 'center_80_crop'})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body['roi']['status'], 'fallback_bbox')
        self.assertEqual(body['roi']['crop_version'], 'oriented-rgb-center-80-v1')
        self.assertIn('center_80_crop', body['reason_codes'])

    def test_recognize_invalid_roi_mode_400(self):
        response = self.client.post('/api/recognize',
                                    files={'image': ('a.webp', picture())},
                                    data={'roi_mode': 'magical_detector'})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()['reason_codes'], ['invalid_roi_mode'])

    def test_aliases_endpoint(self):
        response = self.client.get('/api/aliases')
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn('count', body)
        self.assertIn('producers', body)

    def test_recognize_empty_image_413(self):
        response = self.client.post('/api/recognize',
                                    files={'image': ('a.webp', b'')})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()['reason_codes'], ['invalid_size'])

    def test_legacy_endpoints_shapes(self):
        legacy = self.client.post('/v1/recognize',
                                  files={'image': ('a.webp', picture())})
        self.assertEqual(legacy.status_code, 200)
        self.assertIn('status', legacy.json())
        eval_response = self.client.post('/v1/eval/predict',
                                         files={'image': ('a.webp', picture())})
        self.assertEqual(eval_response.status_code, 200)
        self.assertEqual(set(eval_response.json().keys()), {'slug'})

    def test_unhandled_error_is_internal(self):
        index = Index(list(WINES))
        pipeline = Pipeline(index, MockOCR('Алиготе'), csv_sha256=CSV_SHA)
        app = create_app(pipeline)

        @app.get('/api/boom')
        async def boom():
            raise RuntimeError('do not leak me')

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get('/api/boom')
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()['reason_codes'], ['internal'])
        self.assertNotIn('do not leak me', response.text)


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.client = build_client(token='sekret')

    def test_health_is_open(self):
        self.assertEqual(self.client.get('/api/health').status_code, 200)

    def test_post_requires_bearer(self):
        missing = self.client.post('/api/search', json={'text': 'Алиготе'})
        self.assertEqual(missing.status_code, 401)
        self.assertEqual(missing.json()['reason_codes'], ['unauthorized'])
        wrong = self.client.post('/api/search', json={'text': 'Алиготе'},
                                 headers={'Authorization': 'Bearer wrong'})
        self.assertEqual(wrong.status_code, 401)
        good = self.client.post('/api/search', json={'text': 'Алиготе'},
                                headers={'Authorization': 'Bearer sekret'})
        self.assertEqual(good.status_code, 200)

    def test_eval_requires_bearer(self):
        response = self.client.post('/v1/eval/predict',
                                    files={'image': ('a.webp', picture())})
        self.assertEqual(response.status_code, 401)
        self.assertIsNone(response.json()['slug'])


class CorsTests(unittest.TestCase):
    def test_cors_header_for_configured_origin(self):
        client = build_client(cors_origins=('http://localhost:5173',))
        response = client.get('/api/health',
                              headers={'Origin': 'http://localhost:5173'})
        self.assertEqual(response.headers.get('access-control-allow-origin'),
                         'http://localhost:5173')


class MediaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'asset.webp').write_bytes(picture())
        self.links = {'aligote-2021': {'image_path': 'asset.webp',
                                       'mapping_status': 'candidate'}}

    def tearDown(self):
        self.temp.cleanup()

    def test_media_only_allowed_candidates(self):
        client = build_client(media_links=self.links, uploads_root=self.root)
        served = client.get('/api/media/asset.webp')
        self.assertEqual(served.status_code, 200)
        self.assertGreater(len(served.content), 0)
        self.assertEqual(client.get('/api/media/other.webp').status_code, 404)

    def test_media_url_on_card_and_status(self):
        client = build_client(media_links=self.links, uploads_root=self.root)
        detail = client.get('/api/wines/aligote-2021').json()
        self.assertEqual(detail['media_url'], '/api/media/asset.webp')
        self.assertEqual(detail['media_mapping_status'], 'candidate')

    def test_media_404_without_uploads_root(self):
        client = build_client(media_links=self.links)
        self.assertEqual(client.get('/api/media/asset.webp').status_code, 404)

    def test_media_base_url_override(self):
        client = build_client(media_links=self.links,
                              media_base_url='https://cdn.example.com/wine')
        detail = client.get('/api/wines/aligote-2021').json()
        self.assertEqual(detail['media_url'], 'https://cdn.example.com/wine/asset.webp')

    def test_load_media_links_missing_file(self):
        self.assertEqual(load_media_links(self.root / 'missing.jsonl'), {})

    def test_load_media_links_reads_candidates(self):
        path = self.root / 'links.jsonl'
        path.write_text(json.dumps({'slug': 'a', 'image_path': 'x.webp',
                                    'mapping_status': 'candidate'}) + '\n',
                        encoding='utf-8')
        links = load_media_links(path)
        self.assertEqual(links['a']['image_path'], 'x.webp')


class BlockingOCR:
    """OCR stub that blocks inside recognize() until released."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def recognize(self, roi, deadline):
        self.entered.set()
        self.release.wait(5)
        return OCRResult(roi.roi_id, 'ok', 'Алиготе', 'алиготе', 'blocking', 0)


class ConcurrencyTests(unittest.TestCase):
    def test_saturation_returns_503_busy(self):
        ocr = BlockingOCR()
        pipeline = Pipeline(Index(list(WINES)), ocr, csv_sha256=CSV_SHA,
                            roi_mode='full_frame_experimental')
        app = create_app(pipeline, max_workers=1)
        transport = ASGITransport(app=app)

        async def scenario():
            async with AsyncClient(transport=transport, base_url='http://test') as client:
                first = asyncio.ensure_future(client.post(
                    '/api/recognize', files={'image': ('a.webp', picture())}))
                loop = asyncio.get_running_loop()
                # Wait off the event loop until the only worker is inside OCR.
                awaiting = await loop.run_in_executor(None, ocr.entered.wait, 3)
                self.assertTrue(awaiting)
                busy = await client.post('/api/recognize',
                                         files={'image': ('a.webp', picture())})
                ocr.release.set()
                return await first, busy

        first, busy = asyncio.run(scenario())
        self.assertEqual(first.status_code, 200)
        self.assertEqual(busy.status_code, 503)
        self.assertEqual(busy.json()['reason_codes'], ['busy'])


class PolicyTests(unittest.TestCase):
    def test_calibrated_policy_can_accept(self):
        index = Index(list(WINES))
        policy = Policy(min_score=0, min_margin=0, max_distance=1,
                        version='test-policy')
        pipeline = Pipeline(index, MockOCR('Алиготе'), policy=policy,
                            csv_sha256=CSV_SHA, roi_mode='full_frame_experimental')
        client = TestClient(create_app(pipeline))
        body = client.post('/api/recognize', files={'image': ('a.webp', picture())}).json()
        self.assertEqual(body['status'], 'accepted')
        self.assertIn(body['slug'], {'aligote-2021', 'aligote-2022'})
        self.assertEqual(body['versions']['policy'], 'test-policy')


if __name__ == '__main__':
    unittest.main()
