"""CPU-only contract/regression tests; fixture ranks are not accuracy estimates."""
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from wineid.api import create_app
from wineid.blend import (Blender, BlendConfig, CandidateEvidence, clean_ocr,
                          ocr_attributes)
from wineid.catalog import Wine, normalize
from wineid.clip_zero_shot import EmbeddingIndex, SCHEMA
from wineid.fusion_vision import VisionSources
from wineid.fusion_policy import FusionPolicy, build_fusion_policy
from wineid.fusion_eval import checked_queries, evaluate, tune_validation, calibrate_temperature
from wineid.ocr import MockOCR, OCRResult
from wineid.pipeline import Pipeline
from wineid.text_policy import load_policy
from wineid.search import Index
from wineid.visual import Gallery, build_gallery, MODEL, PRETRAINED


def wine(slug, name, producer, *, grape='', color='Красное', sha='test'):
    raw = {'Название вина': name, 'Винодельня': producer, 'Категория': color,
           'Сорт винограда': grape, 'Slug': slug}
    return Wine(slug, slug, name, producer, normalize(name), normalize(producer),
                (2,), raw, sha)


def evidence(slug, source, score, rank):
    return CandidateEvidence(slug, source, score, rank, 'text_score_0_100' if source == 'text'
                             else 'cosine', 'fixture-sha')


def picture():
    buf = io.BytesIO()
    Image.new('RGB', (100, 120), 'white').save(buf, 'PNG')
    return buf.getvalue()


WINES = [wine('series-saperavi-135', '100 оттенков Саперави', 'Фанагория', grape='Саперави'),
         wine('series-kaberne-130', '100 оттенков Каберне', 'Фанагория', grape='Каберне'),
         wine('other', 'Другое', 'Сосед', grape='Рислинг', color='Белое')]
# Names above differ => distinct families; for family expansion use same name with
# structured colour/style/grape distinguishing labels (as in real catalog).
FAMILY = [wine('dry-135', 'Линия', 'Дом', grape='Саперави'),
          wine('sweet-125', 'Линия', 'Дом', grape='Каберне'),
          wine('outsider', 'Вино иное', 'Другой', grape='Рислинг')]


class FusionTests(unittest.TestCase):
    def test_clean_ocr_and_three_states(self):
        cleaned = clean_ocr('ФАНАГ0РИЯ Сапeрави сyхое 1OO Blanc Merlot 13,5%')
        self.assertIn('фанагория саперави сухое 100 blanc merlot 13,5%', cleaned)
        attrs = ocr_attributes('Каберне Совиньон 2019 2021 100% 13,5% белое',
                               {'каберне', 'каберне совиньон', 'саперави'})
        self.assertIsNone(attrs['vintage'])  # two years -> ambiguous, NOT a conflict
        self.assertEqual(attrs['alcohol'], 13.5)
        self.assertEqual(attrs['grapes'], {'каберне совиньон'})
        self.assertEqual(attrs['color'], 'Белое')
        self.assertIsNone(ocr_attributes('Саперави')['alcohol'])
        self.assertEqual(clean_ocr('Blanc Merlot 2019'), 'blanc merlot 2019')

    def test_family_expansion_and_member_disambiguation(self):
        b = Blender(FAMILY, config=BlendConfig(top_k=1))
        cues = [evidence('dry-135', 'photo', .8, 1),
                evidence('sweet-125', 'photo', .78, 2),
                evidence('outsider', 'photo', .75, 3)]
        # top-K only contained dry; retrieve sweet from same family and use OCR.
        result = b.rank(cues, 'Линия Каберне 12,5%')
        self.assertEqual(result[0]['slug'], 'sweet-125')
        self.assertEqual(result[0]['family_size'], 2)
        self.assertGreater(result[0]['attribute_score'], result[1]['attribute_score'])
        self.assertIn('photo', result[0]['signals'])
        self.assertTrue(result[0]['blend_score'] > result[1]['blend_score'])
        self.assertEqual(b.rank(cues)[0]['slug'], 'dry-135')
        self.assertEqual(b.rank([], 'Линия'), [])

    def test_union_missing_photo_is_not_negative(self):
        b = Blender(WINES, config=BlendConfig(top_k=1))
        cues = [evidence('series-kaberne-130', 'photo', .8, 1),
                evidence('series-saperavi-135', 'text', 96, 1)]
        out = b.rank(cues, 'Саперави 13,5%')
        self.assertEqual(out[0]['slug'], 'series-saperavi-135')
        self.assertNotIn('photo', out[0]['signals'])
        self.assertEqual(out[0]['score'], 100)
        self.assertGreater(out[0]['blend_score'], out[1]['blend_score'])
        self.assertEqual(Blender(WINES).rank([evidence('other', 'text', 90, 1)])[0]['slug'], 'other')

    def test_rrf_and_gaps_and_invalid_evidence(self):
        a, c = evidence('other', 'class', .4, 1), evidence('series-kaberne-130', 'class', .39, 2)
        b = Blender(WINES, config=BlendConfig(mode='rrf', top_k=2))
        self.assertEqual(b.rank([a, c])[0]['slug'], 'other')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            b.rank([a, a])
        with self.assertRaisesRegex(ValueError, 'unknown slug'):
            b.rank([evidence('absent', 'photo', .4, 1)])
        for options in ({'tau': {'photo': 0, 'text': 10, 'class': .05}},
                        {'mode': 'zscore'}, {'top_k': 0}):
            with self.assertRaises(ValueError):
                BlendConfig(**options)
        with self.assertRaises(ValueError):
            evidence('other', 'photo', float('nan'), 1)

    def test_no_photo_vision_runtime_and_mismatch(self):
        class FakeEncoder:
            config = {'model': 'fixture', 'revision': 'revision', 'preprocess': 'fixture',
                      'dimensions': 2}
            def encode_images(self, images):
                return np.array([[1, 0]], dtype='float32')
        items = [{'id': w.slug, 'prompts': [w.name]} for w in WINES]
        classes = EmbeddingIndex(np.array([[.8, .6], [1, 0], [.6, .8]]), {
            'schema': SCHEMA, 'kind': 'classes', 'encoder': FakeEncoder.config,
            'prompt_variant': 'title', 'ensemble': False, 'source_sha256': 'test',
            'items': items, 'prompt_collisions': {}})
        source = VisionSources(FakeEncoder(), WINES, classes=classes)
        obs = source.observe(Image.new('RGB', (10, 10)), __import__('time').monotonic()+5)
        self.assertEqual(obs[0].slug, 'series-kaberne-130')
        self.assertEqual(obs[0].score, 1)
        class FakeTTAEncoder(FakeEncoder):
            def encode_images(self, images):
                return np.array([[1, 0], [0, 1], [1, 0]], dtype='float32')
        tta = VisionSources(FakeTTAEncoder(), WINES, classes=classes, query_views=3)
        crop_obs = tta.observe(Image.new('RGB', (10, 10)), __import__('time').monotonic()+5)
        self.assertEqual(next(r for r in crop_obs if r.slug == 'other').query_view, 'center_80')
        bad = [wine('new', 'New', 'X', sha='different')]
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            VisionSources(FakeEncoder(), bad, classes=classes)

    def test_verified_multiview_gallery_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = picture()
            (root/'reference.png').write_bytes(img)
            row = {'asset_id': 'verified-1', 'wine_id': FAMILY[0].wine_id,
                   'role': 'front', 'source_group': 'gallery-shoot',
                   'mapping_status': 'verified', 'verified_by': 'human',
                   'verification_method': 'read label', 'image_path': 'reference.png',
                   'image_sha256': hashlib.sha256(img).hexdigest(),
                   'bbox': [0, 0, 100, 120]}
            (root/'gallery.jsonl').write_text(json.dumps(row)+'\n')
            def fake_embed(pngs, **kwargs):
                return np.tile([1., 0.], (len(pngs), 1))
            build_gallery(root/'gallery.jsonl', FAMILY, root/'gallery.npz',
                          embed=fake_embed, synthetic_views=2, batch_size=1)
            g = Gallery(root/'gallery.npz', model=MODEL,
                        pretrained=PRETRAINED, csv_sha256='test')
            self.assertEqual(len(g.sources), 3)
            self.assertEqual(g.meta['synthetic_views'], 2)
            self.assertEqual(g.scores([1., 0.], ['dry-135'])['dry-135']['reference_count'], 3)
            self.assertEqual(len({s['image_sha256'] for s in g.sources}), 3)

    def test_end_to_end_visual_and_ocr_one_image_encoder(self):
        class FakeVision:
            classes = gallery = None
            class Encoder:
                config = {'fixture': True}
            encoder = Encoder()
            def __init__(self):
                self.calls = 0
            def observe(self, image, deadline):
                self.calls += 1
                return [evidence('dry-135', 'photo', .8, 1),
                        evidence('sweet-125', 'photo', .78, 2)]
        visual = FakeVision()
        pipe = Pipeline(Index(FAMILY), MockOCR('Линия Каберне 12,5%'),
                        csv_sha256='test', roi_mode='center_80_crop', vision=visual)
        response = pipe.predict(picture())
        self.assertEqual(response['status'], 'ambiguous')
        self.assertIsNone(response['slug'])
        self.assertEqual(response['candidates'][0]['slug'], 'sweet-125')
        self.assertEqual(visual.calls, 1)
        self.assertEqual(response['ocr']['cleaned_text'], 'линия каберне 12,5%')
        forced = pipe.predict(picture(), force_top1=True)
        self.assertEqual(forced['slug'], 'sweet-125')
        client = TestClient(create_app(pipe))
        r = client.post('/v1/eval/predict', files={'image': ('photo.png', picture())})
        self.assertEqual(r.json(), {'slug': 'sweet-125'})
        self.assertIsNone(client.post('/api/recognize', files={
            'image': ('photo.png', picture())}).json()['slug'])

    def test_visual_survives_ocr_failure_and_conditional_skip(self):
        class Vision:
            classes = gallery = None
            class Encoder:
                config = {'fixture': True}
            encoder = Encoder()
            def observe(self, image, deadline):
                return [evidence('outsider', 'photo', .9, 1),
                        evidence('dry-135', 'photo', .7, 2)]
        class FailingOCR:
            model = 'offline-failure'
            def __init__(self):
                self.calls = 0
            def recognize(self, roi, deadline):
                self.calls += 1
                return OCRResult(roi.roi_id, 'timeout', None, None, self.model, 1, 'deadline')
        ocr = FailingOCR()
        pipe = Pipeline(Index(FAMILY), ocr, csv_sha256='test',
                        vision=Vision(), roi_mode='center_80_crop')
        result = pipe.predict(picture())
        self.assertEqual(result['candidates'][0]['slug'], 'outsider')
        self.assertIn('ocr_timeout', result['reason_codes'])
        self.assertIsNone(result['slug'])
        pipe.conditional_ocr = True
        skipped = pipe.predict(picture())
        self.assertEqual(skipped['candidates'][0]['slug'], 'outsider')
        self.assertIn('conditional_ocr_skipped_experimental', skipped['reason_codes'])
        self.assertEqual(ocr.calls, 1)

    def test_eval_fallback_only_on_valid_image(self):
        pipe = Pipeline(Index(FAMILY), csv_sha256='test')
        client = TestClient(create_app(pipe))
        r = client.post('/v1/eval/predict', files={'image': ('p.png', picture())})
        self.assertEqual(r.json(), {'slug': 'dry-135'})  # logged catalog fallback, NOT evidence
        invalid = client.post('/v1/eval/predict', files={'image': ('p.png', b'nonsense')})
        self.assertEqual(invalid.status_code, 422)
        self.assertIsNone(invalid.json()['slug'])

    def test_offline_validation_no_leakage_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            img = picture()
            (root/'query.png').write_bytes(img)
            queries = [{'query_id': 'q', 'image_path': 'query.png',
                        'image_sha256': hashlib.sha256(img).hexdigest(),
                        'source_group': 'independent', 'split': 'validation',
                        'gt_status': 'known', 'gt_slug': 'dry-135', 'gt_source': 'human'}]
            path = root / 'queries.jsonl'
            path.write_text(json.dumps(queries[0])+'\n')
            checked = checked_queries(path, FAMILY)
            rows = [{'query_id': 'q', 'evidence': [evidence('dry-135', 'photo', .8, 1).__dict__,
                     evidence('sweet-125', 'photo', .7, 2).__dict__], 'ocr_text': 'Линия Саперави'}]
            metrics = evaluate(FAMILY, checked, rows)
            self.assertEqual(metrics['accuracy_at_1']['numerator'], 1)
            self.assertEqual(metrics['family_accuracy_at_1']['numerator'], 1)
            self.assertEqual(metrics['member_accuracy_given_correct_multi_sku_family']['numerator'], 1)
            self.assertEqual(tune_validation(FAMILY, checked, rows)[0]['accuracy_at_1']['value'], 1)
            self.assertEqual(calibrate_temperature(FAMILY, checked, rows)['known_in_pool'], 1)
            with self.assertRaisesRegex(ValueError, 'validation-only'):
                tune_validation(FAMILY, [{**queries[0], 'split': 'test'}], rows)
            path.write_text(json.dumps({**queries[0], 'image_sha256': '0'*64})+'\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                checked_queries(path, FAMILY)

    def test_versioned_policy_and_acceptance(self):
        pipe = Pipeline(Index(FAMILY), MockOCR('Линия'), csv_sha256='test',
                        roi_mode='center_80_crop', blender=Blender(FAMILY))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'valid.jsonl'
            (Path(tmp) / 'q.png').write_bytes(picture())
            path.write_text(json.dumps({'query_id': 'q', 'source_group': 'new-shoot',
                                        'image_path': 'q.png',
                                        'split': 'validation', 'gt_status': 'known',
                                        'gt_slug': 'dry-135', 'gt_source': 'annotator',
                                        'image_sha256': hashlib.sha256(picture()).hexdigest()})+'\n')
            policy = build_fusion_policy(path, wines=FAMILY, config=pipe.fusion_config,
                                         min_cosine=.6, min_text_score=40, min_margin=.2)
            pipe_with_policy = Pipeline(pipe.index, pipe.ocr, policy, 'test',
                                        roi_mode='center_80_crop', blender=pipe.blender)
            self.assertEqual(pipe_with_policy.policy.version, 'fusion-policy-v4')
            (Path(tmp)/'policy.json').write_text(json.dumps({**policy.payload(),
                                         'artifact_sha256': policy.artifact_sha256}))
            loaded = load_policy(Path(tmp)/'policy.json')
            self.assertEqual(loaded, policy)
            self.assertAlmostEqual(loaded.confidence([{'blend_score': 1.},
                {'blend_score': 0.}])['p_top5_given_pool'], 1.)
            with self.assertRaisesRegex(ValueError, 'configuration mismatch'):
                Pipeline(pipe.index, pipe.ocr, policy, 'another',
                         roi_mode='center_80_crop', blender=pipe.blender)
            with self.assertRaises(ValueError):
                FusionPolicy(**{**policy.__dict__, 'min_margin': 1.})
            path.write_text(path.read_text().replace('validation', 'test'))
            with self.assertRaisesRegex(ValueError, 'validation-only'):
                build_fusion_policy(path, wines=FAMILY, config=pipe.fusion_config,
                                    min_cosine=.6, min_text_score=40, min_margin=.2)


if __name__ == '__main__':
    unittest.main()
