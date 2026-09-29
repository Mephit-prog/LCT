"""Offline release-preflight and policy provenance regressions; no model downloads."""
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from wineid.api import create_app
from wineid.blend import Blender, BlendConfig, CandidateEvidence
from wineid.canonical import ProducerAliases
from wineid.catalog import Wine, normalize
from wineid.fusion_policy import build_fusion_policy
from wineid.fusion_eval import evaluate
from wineid.ocr import MockOCR
from wineid.pipeline import Pipeline
from wineid.runtime import Settings, open_runtime
from wineid.search import Index
from wineid.text_policy import build_text_policy, fingerprint

SHA = 'a' * 64


def image():
    buffer = io.BytesIO()
    Image.new('RGB', (100, 120), 'white').save(buffer, 'PNG')
    return buffer.getvalue()


def wines():
    return [Wine(s, s, s, 'Завод', normalize(s), 'завод', (1,), {}, SHA)
            for s in ('Алиготе', 'Каберне')]


class RuntimeProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.photo = image()
        (self.root / 'photo.png').write_bytes(self.photo)
        self.manifest = self.root / 'validation.jsonl'
        self.manifest.write_text(json.dumps({
            'query_id': 'q', 'image_path': 'photo.png',
            'image_sha256': hashlib.sha256(self.photo).hexdigest(),
            'source_group': 'shoot-1', 'split': 'validation',
            'gt_status': 'known', 'gt_slug': 'Алиготе', 'gt_source': 'annotator'}) + '\n')

    def test_preflight_validation_before_loading_models(self):
        with patch('wineid.runtime.load_vision', side_effect=AssertionError('loaded')):
            for env in ({'WINE_REQUEST_TIMEOUT': 'nan'}, {'WINE_REQUEST_TIMEOUT': '0'},
                        {'WINE_PORT': '65536'}, {'WINE_MAX_WORKERS': '-1'},
                        {'WINE_QUERY_VIEWS': '2'}, {'WINE_ROI_MODE': 'typo'},
                        {'WINE_OCR_MODE': 'warm', 'WINE_OCR_PROVIDER': 'tesseract'},
                        {'WINE_CLASS_INDEX': 'missing.npz', 'WINE_ALLOW_REMOTE_CODE': '1'},
                        {'WINE_POLICY': 'missing.json'},
                        {'WINE_OCR_PROVIDER': 'mock', 'WINE_ENABLE_MOCK_OCR': '1',
                         'WINE_POLICY': 'missing.json'}):
                with self.subTest(env=env), self.assertRaises(ValueError):
                    Settings.from_env(env)
            with self.assertRaises(ValueError):
                with open_runtime({'WINE_REQUEST_TIMEOUT': 'nan'}):
                    pass

    def test_runtime_closes_worker_on_failure_and_shutdown(self):
        class OCR:
            model = 'fixture'
            ready = True
            def __init__(self):
                self.started = self.closed = 0
            def start(self):
                self.started += 1
            def close(self):
                self.closed += 1
        ocr = OCR()
        with (patch('wineid.runtime.load_catalog', return_value=(wines(), {'csv_sha256': SHA})),
              patch('wineid.runtime.configured_ocr', return_value=ocr),
              patch('wineid.runtime.load_vision', side_effect=ValueError('bad index'))):
            with self.assertRaisesRegex(ValueError, 'bad index'):
                with open_runtime({'WINE_CSV': str(self.root / 'photo.png')}):
                    pass
        self.assertEqual(ocr.closed, 1)
        self.assertEqual(ocr.started, 1)  # canary precedes vision loading
        with (patch('wineid.runtime.load_catalog', return_value=(wines(), {'csv_sha256': SHA})),
              patch('wineid.runtime.configured_ocr', return_value=ocr),
              patch('wineid.runtime.load_vision', return_value=None)):
            with open_runtime({'WINE_CSV': str(self.root / 'photo.png')}) as runtime:
                self.assertEqual(runtime.pipeline.roi_mode, 'refuse')
                self.assertEqual(runtime.settings.timeout, 8.5)
                self.assertEqual(ocr.started, 2)
                self.assertEqual(ocr.closed, 1)
            self.assertEqual(ocr.closed, 2)

    def text_pipe(self):
        aliases = ProducerAliases({'винодельня': 'завод'})
        return Pipeline(Index(wines(), aliases=aliases), MockOCR('Алиготе'),
                        csv_sha256=SHA, roi_mode='center_80_crop')

    def test_text_policy_mutations_and_request_roi_fail_closed(self):
        base = self.text_pipe()
        policy = build_text_policy(self.manifest, min_score=0, min_margin=0,
                                   max_distance=1, config=base.text_config)
        pipe = Pipeline(base.index, base.ocr, policy, SHA, roi_mode='center_80_crop')
        self.assertEqual(pipe.predict(self.photo)['status'], 'accepted')
        for key, val in {'roi_mode': 'refuse', 'ocr': None,
                         'aliases_sha256': 'b'*64, 'search_backend': 'new',
                         'decision_code_sha256': 'b'*64}.items():
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'configuration mismatch'):
                build = self.text_pipe()
                changed = {**policy.payload(), 'config': {**policy.config, key: val}}
                # Rehash so the artifact is internally valid, but mismatched.
                from wineid.text_policy import TextPolicy, fingerprint
                candidate = TextPolicy(**changed, artifact_sha256=fingerprint(changed))
                Pipeline(build.index, build.ocr, candidate, SHA, roi_mode='center_80_crop')
        for kwargs in ({'bbox': (0, 0, 80, 100)},
                       {'roi_mode': 'full_frame_experimental'},
                       {'manual_text': 'Алиготе'}):
            result = pipe.predict(self.photo, **kwargs)
            self.assertEqual(result['status'], 'ambiguous')
            self.assertIsNone(result['slug'])
            self.assertIn('uncalibrated_request_profile', result['reason_codes'])
        self.assertEqual(TestClient(create_app(pipe)).post('/api/recognize',
                         files={'image': ('photo.png', self.photo)},
                         data={'roi_mode': 'full_frame_experimental'}).json()['slug'], None)
        pipe.ocr.model = 'changed'
        self.assertIn('policy_configuration_changed', pipe.predict(self.photo)['reason_codes'])
        pipe.ocr.model = 'mock'
        pipe.index.aliases.variants['new'] = 'new'
        self.assertIn('policy_configuration_changed', pipe.predict(self.photo)['reason_codes'])
        pipe.index.aliases.variants.pop('new')
        pipe.policy.config['roi_mode'] = 'refuse'  # dataclass frozen, nested dict mutable
        self.assertIn('policy_configuration_changed', pipe.predict(self.photo)['reason_codes'])

    def test_evaluator_uses_online_collision_gate_and_evidence_binding(self):
        class Vision:
            class Classes:
                archive_sha256 = 'index-sha'
                meta = {'prompt_collisions': {'Алиготе': ['Алиготе', 'Каберне']}}
            classes = Classes()
            gallery = None
            class_fingerprint = 'index-sha'
            class Encoder:
                config = {'fixture': True}
            encoder = Encoder()
            query_views = 1
            def observe(self, image, deadline):
                return [CandidateEvidence(s, 'class', score, rank, 'cosine', 'index-sha')
                        for rank, (s, score) in enumerate([('Алиготе', .9), ('Каберне', .7)], 1)]
        vision = Vision()
        base = Pipeline(Index(wines()), vision=vision, csv_sha256=SHA,
                        roi_mode='center_80_crop')
        policy = build_fusion_policy(self.manifest, wines=wines(), config=base.fusion_config,
                                     min_cosine=.2, min_text_score=0, min_margin=0)
        pipe = Pipeline(base.index, vision=vision, csv_sha256=SHA,
                        roi_mode='center_80_crop', policy=policy)
        online = pipe.predict(self.photo)
        self.assertEqual(online['status'], 'ambiguous')
        row = json.loads(self.manifest.read_text())
        observations = [CandidateEvidence(**{
            'slug': s, 'source': 'class', 'score': score, 'rank': rank,
            'score_type': 'cosine', 'provenance': 'index-sha'})
            for rank, (s, score) in enumerate([('Алиготе', .9), ('Каберне', .7)], 1)]
        from dataclasses import asdict
        evidence = {'query_id': 'q', 'image_sha256': row['image_sha256'],
                    'catalog_sha256': SHA, 'fusion_config_sha256': fingerprint(pipe.fusion_config),
                    'evidence': [asdict(e) for e in observations], 'ocr_status': 'not_run'}
        offline = evaluate(wines(), [row], [evidence], policy=policy, pipeline=pipe)
        self.assertEqual(offline['predictions'][0]['status'], online['status'])
        self.assertEqual(offline['predictions'][0]['accepted_slug'], online['slug'])
        self.assertEqual(offline['predictions'][0]['top5'],
                         [e['slug'] for e in online['candidates'][:5]])
        for changed in ({'image_sha256': 'b'*64}, {'fusion_config_sha256': '0'*64},
                        {'catalog_sha256': 'wrong'}, {'ocr_status': 'manual_oracle'},
                        {'evidence': [{**evidence['evidence'][0], 'provenance': 'wrong'},
                                      evidence['evidence'][1]]}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                evaluate(wines(), [row], [{**evidence, **changed}], policy=policy, pipeline=pipe)
        with self.assertRaisesRegex(ValueError, 'requires a pipeline'):
            evaluate(wines(), [row], [evidence], policy=policy)

    def test_fusion_mutations_fail_closed(self):
        class Vision:
            classes = gallery = None
            class Encoder:
                config = {'fixture': 'v1'}
            encoder = Encoder()
            query_views = 1
            def observe(self, image, deadline):
                return [CandidateEvidence(s, 'class', score, rank, 'cosine', 'fixture')
                        for rank, (s, score) in enumerate([('Алиготе', .9), ('Каберне', .7)], 1)]
        vision = Vision()
        base = Pipeline(Index(wines()), vision=vision, csv_sha256=SHA,
                        blender=Blender(wines()), roi_mode='center_80_crop')
        policy = build_fusion_policy(self.manifest, wines=wines(), config=base.fusion_config,
                                     min_cosine=.2, min_text_score=0, min_margin=0)
        pipe = Pipeline(base.index, policy=policy, vision=vision, blender=base.blender,
                        csv_sha256=SHA, roi_mode='center_80_crop')
        self.assertEqual(pipe.predict(self.photo)['status'], 'accepted')
        for change, undo in ((lambda: setattr(pipe, 'conditional_ocr', True),
                              lambda: setattr(pipe, 'conditional_ocr', False)),
                             (lambda: setattr(vision, 'query_views', 3),
                              lambda: setattr(vision, 'query_views', 1)),
                             (lambda: setattr(pipe.blender, 'config', BlendConfig(top_k=2)),
                              lambda: setattr(pipe.blender, 'config', BlendConfig()))):
            change()
            self.assertIn('policy_configuration_changed', pipe.predict(self.photo)['reason_codes'])
            undo()
            self.assertEqual(pipe.predict(self.photo)['status'], 'accepted')
        result = pipe.predict(self.photo, bbox=(0, 0, 70, 80))
        self.assertIn('uncalibrated_request_profile', result['reason_codes'])
        self.assertIsNone(result['slug'])


if __name__ == '__main__':
    unittest.main()
