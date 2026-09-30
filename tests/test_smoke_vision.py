"""Opt-in real, local cached-weights integration test (no OCR/network calls).

Run with WINE_SMOKE_JINA=1 WINE_ALLOW_REMOTE_CODE=1 after license/code review.
Requires artifacts/classes-lora-canonical.npz built with the adapter below.
"""
import os
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('WINE_SMOKE_JINA') == '1' and
                     os.environ.get('WINE_ALLOW_REMOTE_CODE') == '1',
                     'opt-in local Jina smoke only; review license and remote code')
class VisionSmokeTests(unittest.TestCase):
    def test_real_bytes_weights_class_index_http(self):
        from wineid.api import create_app
        from wineid.catalog import load_catalog
        from wineid.fusion_vision import load_vision
        from wineid.pipeline import Pipeline
        from wineid.search import Index

        classes = ROOT / 'artifacts/classes-lora-canonical.npz'
        adapter = ROOT / 'artifacts/lora/jina-clip-v2-lora-synthetic.pt'
        if not classes.is_file():
            self.skipTest('build pinned class index first; see JINA_CLIP.md')
        wines, meta = load_catalog(ROOT / 'strapi_output0709.csv')
        vision = load_vision(wines, classes_path=classes, adapter_path=adapter,
                             allow_remote_code=True, device='cpu')
        app = create_app(Pipeline(Index(wines), csv_sha256=meta['csv_sha256'], vision=vision))
        photo = (ROOT / 'eval/queries/02eef911.webp').read_bytes()
        client = TestClient(app)
        normal = client.post('/api/recognize', files={'image': ('query.webp', photo)})
        self.assertEqual(normal.status_code, 200)
        self.assertEqual(normal.json()['status'], 'ambiguous')
        self.assertIsNone(normal.json()['slug'])
        self.assertIn('class', normal.json()['versions']['sources'])
        forced = client.post('/v1/eval/predict', files={'image': ('query.webp', photo)})
        self.assertEqual(forced.status_code, 200)
        self.assertIn(forced.json()['slug'], {w.slug for w in wines})
        bad = client.post('/v1/eval/predict', files={'image': ('bad.jpg', b'bad')})
        self.assertEqual(bad.status_code, 422)
        self.assertIsNone(bad.json()['slug'])
