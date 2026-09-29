"""Opt-in offline smoke test (STATUS.md B07).

Skipped unless ``WINE_SMOKE_OFFLINE=1``. Runs the real catalog/search/pipeline
and FastAPI app against local artifacts only — no paid OCR calls, no model
weights. A separate, heavier ``WINE_SMOKE_CLIP=1`` gate would additionally load
the pinned Jina weights; it is intentionally not enabled by default here.

    WINE_SMOKE_OFFLINE=1 python3 -m pytest tests/test_smoke_offline.py -q
"""
import io
import os
import unittest
from pathlib import Path

from PIL import Image

from wineid.api import create_app
from wineid.catalog import load_catalog
from wineid.pipeline import Pipeline
from wineid.search import Index

ROOT = Path(__file__).resolve().parent.parent
SMOKE = os.environ.get('WINE_SMOKE_OFFLINE') == '1'


def picture():
    buffer = io.BytesIO()
    Image.new('RGB', (120, 90), 'white').save(buffer, format='WEBP')
    return buffer.getvalue()


@unittest.skipUnless(SMOKE, 'set WINE_SMOKE_OFFLINE=1 to run the offline smoke test')
class OfflineSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.csv = ROOT / os.environ.get('WINE_CSV', 'strapi_output0709.csv')
        if not cls.csv.is_file():
            raise unittest.SkipTest(f'catalog CSV not found: {cls.csv}')
        cls.wines, cls.report = load_catalog(cls.csv)
        cls.index = Index(cls.wines)

    def test_catalog_and_index_are_loaded(self):
        self.assertGreater(len(self.wines), 1000)
        self.assertEqual(len({w.slug for w in self.wines}), len(self.wines))
        self.assertTrue(self.index.backend)

    def test_search_returns_catalog_candidates(self):
        candidates = self.index.search('Арпачино Иноходец Сибирьковый', 5)
        self.assertTrue(candidates)
        slugs = {w.slug for w in self.wines}
        self.assertTrue(all(c.slug in slugs for c in candidates))
        self.assertTrue(all(0 <= c.score <= 100 for c in candidates))

    def test_pipeline_refuses_without_ocr(self):
        pipeline = Pipeline(self.index, None, csv_sha256=self.report['csv_sha256'],
                            roi_mode='refuse')
        result = pipeline.predict(picture(), seconds=5)
        self.assertEqual(result['status'], 'unreadable')
        self.assertIn('automatic_roi_unavailable', result['reason_codes'])

    def test_fastapi_health_wines_and_search(self):
        from fastapi.testclient import TestClient
        pipeline = Pipeline(self.index, None, csv_sha256=self.report['csv_sha256'],
                            roi_mode='refuse')
        client = TestClient(create_app(pipeline))
        health = client.get('/api/health').json()
        self.assertTrue(health['catalog_ready'])
        self.assertFalse(health['recognition_ready'])
        wines = client.get('/api/wines', params={'limit': 1}).json()
        self.assertEqual(wines['count'], 1)
        search = client.post('/api/search', json={'text': 'каберне', 'k': 3}).json()
        self.assertTrue(search['candidates'])


if __name__ == '__main__':
    unittest.main()
