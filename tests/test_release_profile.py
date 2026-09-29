"""PR-02: policy compatibility and deterministic HTTP bootstrap (offline fixtures)."""
import io
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from wineid.api import create_app
from wineid.catalog import Wine, normalize
from wineid.ocr import MockOCR
from wineid.pipeline import Pipeline
from wineid.profile import DECISION_FILES
from wineid.runtime import Settings, open_runtime
from wineid.search import Index
from wineid.server import create_handler
from wineid.text_policy import TextPolicy, fingerprint

SHA = 'a' * 64


def image():
    buf = io.BytesIO()
    Image.new('RGB', (100, 100), 'white').save(buf, 'PNG')
    return buf.getvalue()


def wines():
    return [Wine(slug, slug, slug, 'Завод', normalize(slug), 'завод', (1,), {}, SHA)
            for slug in ('Алиготе', 'Каберне')]


def calibrated_pipeline():
    index, ocr = Index(wines()), MockOCR('Алиготе')
    base = Pipeline(index, ocr, csv_sha256=SHA, roi_mode='full_frame_experimental')
    payload = dict(version='text-policy-v3', min_score=0, min_margin=0,
                   max_distance=1, config=base.text_config,
                   calibration_manifest_sha256='b' * 64,
                   calibration_source_groups=['independent-shoot'])
    policy = TextPolicy(**payload, artifact_sha256=fingerprint(payload))
    return Pipeline(index, ocr, policy, SHA, roi_mode='full_frame_experimental')


class ReleaseProfileTests(unittest.TestCase):
    def test_mutated_dependency_and_request_budget_refuse_acceptance(self):
        pipe = calibrated_pipeline()
        self.assertEqual(pipe.predict(image())['status'], 'accepted')
        from wineid.profile import version
        with patch('wineid.profile.version', side_effect=lambda name: (
                'changed' if name == 'Pillow' else version(name))):
            self.assertIn('policy_configuration_changed', pipe.predict(image())['reason_codes'])
        for kwargs in ({'request_timeout': 9.}, {'max_concurrent_ocr': 3}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, 'configuration mismatch'):
                Pipeline(pipe.index, pipe.ocr, pipe.policy, SHA, roi_mode='full_frame_experimental',
                         **kwargs)
        self.assertEqual(pipe.predict(image(), seconds=9.)['slug'], None)
        self.assertIn('uncalibrated_request_profile', pipe.predict(image(), seconds=9.)['reason_codes'])
        with self.assertRaisesRegex(ValueError, 'request timeout'):
            create_app(pipe, request_timeout=9.)
        with self.assertRaisesRegex(ValueError, 'request timeout'):
            create_handler(pipe, timeout=9.)
        client = TestClient(create_app(pipe))
        self.assertEqual(client.post('/api/recognize', files={
            'image': ('photo.png', image())}).json()['status'], 'accepted')

    def test_code_profile_covers_workers_and_bootstrap(self):
        for name in ('runtime.py', 'local_ocr_worker.py', 'blend_decision.py',
                     'warm_ocr_worker.py', 'clip_roi.py'):
            self.assertIn(name, DECISION_FILES)

    def test_preflight_rejects_ignored_or_invalid_profiles(self):
        for env in ({'WINE_ALIASES': 'missing-aliases.txt'},
                    {'WINE_MAX_CONCURRENT_OCR': '0'},
                    {'WINE_MAX_CONCURRENT_OCR': 'nan'},
                    {'WINE_OCR_PROVIDER': 'paddle', 'WINE_OCR_MODE': 'warm',
                     'WINE_OCR_MANIFEST': 'missing.json'},
                    {'WINE_REQUEST_TIMEOUT': 'inf'},
                    {'WINE_CONDITIONAL_OCR': '1'},
                    {'WINE_QUERY_VIEWS': '3'}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                Settings.from_env(env)

    def test_partial_vision_load_releases_encoder(self):
        from wineid.fusion_vision import load_vision
        from wineid.jina_clip import MODEL, PREPROCESS, REVISION
        class IndexFixture:
            kind = 'classes'
            meta = {'source_sha256': SHA, 'encoder': {
                'adapter_sha256': None, 'model': MODEL,
                'revision': REVISION, 'preprocess': PREPROCESS}}
            def check_encoder(self, encoder):
                raise ValueError('bad index dimensions')
        class Encoder:
            def __init__(self):
                self.closed = 0
            def close(self):
                self.closed += 1
        encoder = Encoder()
        with (patch('wineid.fusion_vision.EmbeddingIndex.load', return_value=IndexFixture()),
              patch('wineid.fusion_vision.AdaptedEncoder', return_value=encoder)):
            with self.assertRaisesRegex(ValueError, 'bad index dimensions'):
                load_vision(wines(), classes_path='fixture.npz', allow_remote_code=True)
        self.assertEqual(encoder.closed, 1)

    def test_canary_failure_closes_ocr_before_vision_load(self):
        class FailedOCR:
            def __init__(self):
                self.closed = 0
            def start(self):
                raise RuntimeError('canary failed')
            def close(self):
                self.closed += 1
        ocr = FailedOCR()
        with (patch('wineid.runtime.load_catalog', return_value=(wines(), {'csv_sha256': SHA})),
              patch('wineid.runtime.configured_ocr', return_value=ocr),
              patch('wineid.runtime.load_vision', side_effect=AssertionError('vision loaded'))):
            with self.assertRaisesRegex(RuntimeError, 'canary failed'):
                with open_runtime():
                    pass
        self.assertEqual(ocr.closed, 1)

    def test_failed_pipeline_closes_started_ocr_and_vision(self):
        class Resource:
            def __init__(self):
                self.started = self.closed = 0
            def start(self):
                self.started += 1
            def close(self):
                self.closed += 1
        ocr, vision = Resource(), Resource()
        with (patch('wineid.runtime.load_catalog', return_value=(wines(), {'csv_sha256': SHA})),
              patch('wineid.runtime.configured_ocr', return_value=ocr),
              patch('wineid.runtime.load_vision', return_value=vision),
              patch('wineid.runtime.Pipeline', side_effect=ValueError('bad policy'))):
            with self.assertRaisesRegex(ValueError, 'bad policy'):
                with open_runtime():
                    pass
        self.assertEqual((ocr.started, ocr.closed, vision.closed), (1, 1, 1))


if __name__ == '__main__':
    unittest.main()
