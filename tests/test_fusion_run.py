"""Offline runner tests use no weights and never call an external OCR provider."""
import csv
import hashlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path

from PIL import Image

from wineid.blend import Blender, BlendConfig, CandidateEvidence, ocr_attributes
from wineid.catalog import Wine, normalize
from wineid.clip_zero_shot import class_prompts
from wineid.fusion_eval import evaluate
from wineid.fusion_compare import compare_candidate_reports
from wineid.fusion_policy import build_fusion_policy
from wineid.fusion_run import candidate_queries, observe_queries, unlabelled_queries
from wineid.search import Index
from wineid.pipeline import Pipeline
from wineid.api import create_app
from fastapi.testclient import TestClient


def fixture():
    buf = io.BytesIO()
    Image.new('RGB', (120, 140), 'white').save(buf, 'PNG')
    return buf.getvalue()


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.wines = [Wine('first', 'first', 'Красное', 'Дом', normalize('Красное'),
                           normalize('Дом'), (1,), {'Название вина': 'Красное',
                           'Винодельня': 'Дом', 'Категория': 'Красное'}, 'fixture'),
                      Wine('second', 'second', 'Белое', 'Дом', normalize('Белое'),
                           normalize('Дом'), (2,), {'Название вина': 'Белое',
                           'Винодельня': 'Дом', 'Категория': 'Белое'}, 'fixture')]
        self.index = Index(self.wines)
        self.blender = Blender(self.wines)

    def test_foundation_year_unknown_and_bad_source_ranks(self):
        self.assertEqual(ocr_attributes('Основана в 1998 2020')['vintage'], '2020')
        self.assertIsNone(ocr_attributes('Основана в 1998')['vintage'])
        with self.assertRaisesRegex(ValueError, 'duplicate ranks'):
            self.blender.rank([CandidateEvidence('first', 'class', .8, 1, 'cosine', 'fixture'),
                               CandidateEvidence('second', 'class', .7, 1, 'cosine', 'fixture')])
        with self.assertRaisesRegex(ValueError, 'inconsistent ranks'):
            self.blender.rank([CandidateEvidence('first', 'class', .7, 1, 'cosine', 'fixture'),
                               CandidateEvidence('second', 'class', .8, 2, 'cosine', 'fixture')])
        self.assertFalse(BlendConfig().use_slug_alcohol)

    def test_canonical_class_duplicates_rejected_before_model_load(self):
        row = {'class_id': 'first', 'name_ru': 'Вино', 'producer': 'Дом'}
        with self.assertRaisesRegex(ValueError, 'duplicate class_id'):
            class_prompts([row, row], variant='canonical')

    def test_unlabelled_real_bytes_ranked_without_gt(self):
        class Vision:
            def observe(self, image, deadline):
                assert time.monotonic() < deadline
                return [CandidateEvidence('second', 'class', .8, 1, 'cosine', 'fixture'),
                        CandidateEvidence('first', 'class', .6, 2, 'cosine', 'fixture')]
        with tempfile.TemporaryDirectory() as tmp:
            photo = Path(tmp)/'photo.png'
            photo.write_bytes(fixture())
            rows = unlabelled_queries([photo])
            self.assertEqual(set(rows[0]), {'query_id', 'image_path', 'image_sha256'})
            found = observe_queries(rows, root=None, index=self.index,
                                    blender=self.blender, vision=Vision())
            self.assertEqual(found[0]['predicted_slug'], 'second')
            self.assertEqual(found[0]['status'], 'ranked_only')
            self.assertEqual(len(found[0]['evidence']), 2)
            self.assertEqual(found[0]['ocr_status'], 'not_run')
            with self.assertRaisesRegex(ValueError, 'duplicate image bytes'):
                unlabelled_queries([photo, photo])
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                observe_queries([{**rows[0], 'image_sha256': '0'*64}], root=None,
                                index=self.index, blender=self.blender, vision=Vision())

    def test_manual_transcript_is_oracle_and_not_gt_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'photo.png').write_bytes(fixture())
            row = {'query_id': 'q', 'image_path': 'photo.png',
                   'image_sha256': hashlib.sha256(fixture()).hexdigest(),
                   'manual_text': 'Белое Дом', 'gt_status': 'known',
                   'gt_slug': 'first', 'gt_source': 'annotator',
                   'source_group': 'camera', 'split': 'validation'}
            found = observe_queries([row], root=root, index=self.index,
                                    blender=self.blender, vision=None, ocr_mode='manual')
            self.assertEqual(found[0]['ocr_status'], 'manual_oracle')
            self.assertTrue(all('gt_' not in key for key in found[0]))
            self.assertTrue(any(e['source'] == 'text' for e in found[0]['evidence']))
            report = evaluate(self.wines, [row], found)
            self.assertEqual(report['known'], 1)
            with self.assertRaisesRegex(ValueError, 'manual_text required'):
                observe_queries([{**row, 'manual_text': None}], root=root,
                                index=self.index, blender=self.blender,
                                vision=None, ocr_mode='manual')

    def test_large_valid_photo_forced_top1_uses_vision_when_crop_too_large(self):
        class Vision:
            classes = gallery = None
            class Encoder:
                config = {'fixture': True}
            encoder = Encoder()
            def observe(self, image, deadline):
                return [CandidateEvidence('second', 'class', .8, 1, 'cosine', 'fixture')]
        buf = io.BytesIO()
        Image.new('RGB', (3200, 3600), 'white').save(buf, 'PNG')
        pipe = Pipeline(self.index, csv_sha256='fixture', vision=Vision())
        normal = pipe.predict(buf.getvalue(), seconds=30.)
        self.assertEqual(normal['status'], 'ambiguous')
        self.assertIsNone(normal['slug'])
        forced = pipe.predict(buf.getvalue(), force_top1=True, seconds=30.)
        self.assertEqual(forced['slug'], 'second')
        self.assertIn('crop_too_large', forced['reason_codes'])
        self.assertIn('eval_forced_top1', forced['reason_codes'])
        client = TestClient(create_app(pipe, request_timeout=30.))
        response = client.post('/v1/eval/predict', files={'image': ('q.png', buf.getvalue())})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'slug': 'second'})
        invalid = client.post('/v1/eval/predict', files={'image': ('q.png', b'invalid')})
        self.assertEqual(invalid.status_code, 422)

    def test_identical_class_prompts_cannot_be_accepted_without_other_signal(self):
        class Vision:
            class Classes:
                archive_sha256 = 'fixture-index'
                meta = {'prompt_collisions': {'first': ['first', 'second'],
                                               'second': ['first', 'second']}}
            classes = Classes()
            gallery = None
            class Encoder:
                config = {'fixture': True}
            encoder = Encoder()
            def observe(self, image, deadline):
                return [CandidateEvidence('first', 'class', .9, 1, 'cosine', 'fixture'),
                        CandidateEvidence('second', 'class', .9, 2, 'cosine', 'fixture')]
        vision = Vision()
        pipe = Pipeline(self.index, csv_sha256='fixture', vision=vision)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'q.png').write_bytes(fixture())
            (root/'validation.jsonl').write_text(json.dumps({
                'query_id': 'q', 'image_path': 'q.png',
                'image_sha256': hashlib.sha256(fixture()).hexdigest(),
                'source_group': 'human', 'split': 'validation', 'gt_status': 'known',
                'gt_slug': 'first', 'gt_source': 'annotator'})+'\n')
            policy = build_fusion_policy(root/'validation.jsonl', wines=self.wines,
                                         config=pipe.fusion_config, min_cosine=0,
                                         min_text_score=0, min_margin=0)
            guarded = Pipeline(self.index, csv_sha256='fixture', vision=vision, policy=policy)
            result = guarded.predict(fixture())
            self.assertEqual(result['status'], 'ambiguous')
            self.assertIsNone(result['slug'])
            self.assertIn('indistinguishable_class_prompt', result['reason_codes'])
            self.assertEqual(guarded.predict(fixture(), force_top1=True)['slug'], 'first')

    def test_eval_does_not_force_slug_after_vision_deadline(self):
        class TimeoutVision:
            classes = gallery = None
            class Encoder:
                config = {'fixture': True}
            encoder = Encoder()
            def observe(self, image, deadline):
                raise TimeoutError('deadline')
        pipe = Pipeline(self.index, csv_sha256='fixture', vision=TimeoutVision())
        forced = pipe.predict(fixture(), force_top1=True)
        self.assertIsNone(forced['slug'])
        self.assertEqual(forced['reason_codes'], ['deadline'])
        client = TestClient(create_app(pipe))
        r = client.post('/v1/eval/predict', files={'image': ('q.png', fixture())})
        self.assertEqual(r.status_code, 503)
        self.assertIsNone(r.json()['slug'])

    def test_candidate_packshots_are_not_promoted_to_ground_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            photo = root/'photo.png'
            photo.write_bytes(fixture())
            index = root/'images.csv'
            with index.open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['image_id', 'wine_id', 'path', 'sha256'])
                writer.writeheader()
                writer.writerow({'image_id': 'upload-1', 'wine_id': 'first',
                                 'path': 'photo.png', 'sha256': hashlib.sha256(fixture()).hexdigest()})
            rows, uploads = candidate_queries(index, root, self.wines)
            self.assertEqual(uploads, root.resolve())
            self.assertEqual(rows[0]['_candidate_slug'], 'first')
            self.assertNotIn('gt_slug', rows[0])
            photo.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                candidate_queries(index, root, self.wines)

    def test_paired_diagnostic_checks_image_hash_and_config(self):
        base = {'data_status': 'candidate_upload_links_not_ground_truth',
                'catalog_sha256': 'x', 'catalog_wines': 2, 'config': {},
                'fusion_version': 'v', 'search_version': 'v', 'ocr_mode': 'none',
                'class_prompt': {'prompt_variant': 'canonical'},
                'results': [{'query_id': 'q', 'image_sha256': 'sha',
                             'predicted_slug': 'other', 'top5': ['other']}],
                'candidate_link_diagnostic': {
                    'images_csv_sha256': 'manifest', 'seed': '0', 'sample_requested': 1,
                    'sampled': 1, 'agreement_at_1': {'numerator': 0, 'denominator': 1},
                    'agreement_at_5': {'numerator': 0, 'denominator': 1},
                    'pairs': [{'query_id': 'q', 'candidate_slug': 'first',
                               'predicted_slug': 'other'}]}}
        import copy
        after = copy.deepcopy(base)
        after['results'][0]['predicted_slug'] = 'first'
        after['candidate_link_diagnostic']['pairs'][0]['predicted_slug'] = 'first'
        after['results'][0]['top5'] = ['first']
        for k in (1, 5):
            after['candidate_link_diagnostic'][f'agreement_at_{k}']['numerator'] = 1
        paired = compare_candidate_reports(base, after)
        self.assertEqual(paired['paired_images'], 1)
        self.assertEqual(paired['helped_candidate_link'], ['q'])
        after['results'][0]['image_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'not the same'):
            compare_candidate_reports(base, after)
        after['results'][0]['image_sha256'] = 'sha'
        after['data_status'] = 'unlabelled_real_images_no_accuracy'
        with self.assertRaisesRegex(ValueError, 'candidate-link'):
            compare_candidate_reports(base, after)

    def test_missing_roi_blocks_live_ocr_before_provider(self):
        class OCR:
            def recognize(self, *_):
                raise AssertionError('unexpected paid OCR request')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'photo.png').write_bytes(fixture())
            row = {'query_id': 'q', 'image_path': 'photo.png',
                   'image_sha256': hashlib.sha256(fixture()).hexdigest()}
            with self.assertRaisesRegex(ValueError, 'manual bbox required'):
                observe_queries([row], root=root, index=self.index, blender=self.blender,
                                vision=None, ocr=OCR(), ocr_mode='live')
            with self.assertRaisesRegex(ValueError, 'outside manifest'):
                observe_queries([{**row, 'image_path': '../other.png'}], root=root,
                                index=self.index, blender=self.blender, vision=None,
                                ocr_mode='manual')


if __name__ == '__main__':
    unittest.main()
