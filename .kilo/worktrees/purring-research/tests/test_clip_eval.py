import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from wineid.catalog import Wine, normalize
from wineid.search import Index
from wineid.pipeline import Pipeline
from wineid.clip_eval import VisualPolicy, rank, visual_decision, evaluate, comparison, fingerprint
from wineid.visual import Gallery, IncompatibleGallery, build_gallery, MODEL, PRETRAINED


def photo(color):
    buffer = io.BytesIO()
    Image.new('RGB', (100, 100), color).save(buffer, 'PNG')
    return buffer.getvalue()


def wine(slug):
    return Wine(slug, slug, slug, 'Производитель', normalize(slug),
                normalize('Производитель'), (2,), {}, 'test')


class ClipExperiment(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.wines = [wine('one-2020'), wine('one-2021')]
        self.samples = [photo('red'), photo('blue'), photo('green')]
        for i, data in enumerate(self.samples):
            (self.root/f'{i}.png').write_bytes(data)
        gallery_rows = [dict(asset_id=f'a{i}', wine_id=w.wine_id, role='front',
                             source_group=f'gallery-{i}', mapping_status='verified',
                             verified_by='human', verification_method='checked label',
                             image_path=f'{i}.png', image_sha256=hashlib.sha256(self.samples[i]).hexdigest(),
                             bbox=[0, 0, 100, 100]) for i, w in enumerate(self.wines)]
        (self.root/'gallery.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in gallery_rows))
        self.calls = []
        def embed(pngs, **kwargs):
            self.calls.append(len(pngs))
            return np.array([[1., 0.] if len(self.calls) == 1 else [0., 1.]
                             for _ in pngs], dtype='float32')
        self.embed = embed
        build_gallery(self.root/'gallery.jsonl', self.wines, self.root/'gallery.npz',
                      embed=embed, batch_size=1)
        self.gallery = Gallery(self.root/'gallery.npz', model=MODEL,
                               pretrained=PRETRAINED, csv_sha256='test')
        self.query = dict(query_id='q', image_path='2.png',
                          image_sha256=hashlib.sha256(self.samples[2]).hexdigest(),
                          bbox=[0, 0, 100, 100], source_group='query-shoot', split='validation',
                          gt_status='known', gt_slug='one-2021', gt_source='independent annotation')
        self.save_query()
        calibration = photo('yellow')
        (self.root/'calibration.png').write_bytes(calibration)
        self.calibration = dict(self.query, query_id='cal', image_path='calibration.png',
                                image_sha256=hashlib.sha256(calibration).hexdigest(),
                                source_group='calibration-shoot')
        (self.root/'calibration.jsonl').write_text(json.dumps(self.calibration)+'\n')
        self.baseline = Pipeline(Index(self.wines), csv_sha256='test').predict(
            self.samples[2], bbox=(0, 0, 100, 100), manual_text='one-2020')
        self.baseline.update(query_id='q', image_sha256=self.query['image_sha256'],
                             source_group='query-shoot', split='validation',
                             gt_status='known', gt_slug='one-2021')
        self.save_baseline()

    def save_query(self):
        (self.root/'query.jsonl').write_text(json.dumps(self.query)+'\n')

    def save_baseline(self):
        (self.root/'baseline.jsonl').write_text(json.dumps(self.baseline)+'\n')

    def evaluate(self, **kwargs):
        return evaluate(self.root/'query.jsonl', self.root/'baseline.jsonl',
                        self.gallery, lambda pngs, **kw: np.array([[0., 1.]], dtype='float32'),
                        calibration_manifest=self.root/'calibration.jsonl', **kwargs)

    def policy(self, *, mode='clip', weight=0.5, role='front', top_k=20):
        config = dict(mode=mode, weight=weight, role=role, top_k=top_k,
                      model=self.gallery.meta['model'], pretrained=self.gallery.meta['pretrained'],
                      preprocess=self.gallery.meta['preprocess'],
                      crop_version=self.gallery.meta['crop_version'],
                      gallery_manifest_sha256=self.gallery.meta['manifest_sha256'],
                      gallery_archive_sha256=self.gallery.archive_sha256, csv_sha256='test')
        payload = dict(version='visual-policy-v1', min_text_score=0, min_cosine=0.5,
                       min_margin=0.2, config=config,
                       calibration_manifest_sha256=hashlib.sha256(
                           (self.root/'calibration.jsonl').read_bytes()).hexdigest(),
                       calibration_source_groups=['calibration-shoot'])
        return VisualPolicy(**payload, artifact_sha256=fingerprint(payload))

    def test_build_batches_and_role_missing(self):
        self.assertEqual(self.calls, [1, 1])
        scores = self.gallery.scores([0, 1], ['one-2020', 'one-2021', 'absent'])
        self.assertEqual(scores['one-2021']['reference_id'], 'a1')
        self.assertEqual(scores['absent']['reference_count'], 0)
        self.assertIsNone(self.gallery.scores([1, 0], ['one-2020'], role='back')['one-2020']['cosine'])
        with self.assertRaises(IncompatibleGallery):
            self.gallery.scores([0, 0], ['one-2021'])
        with self.assertRaises(IncompatibleGallery):
            Gallery(self.root/'gallery.npz', model='different',
                    pretrained='laion2b_s34b_b79k', csv_sha256='test')

    def test_paired_rerank_and_policy(self):
        rows = self.evaluate()
        self.assertEqual(rows[0]['clip_reranked'][0], 'one-2021')
        self.assertEqual(rows[0]['status'], 'ambiguous')
        self.assertIsNone(rows[0]['confidence_calibrated'])
        summary = comparison([self.baseline], rows)
        self.assertEqual(summary['helped_query_ids'], ['q'])
        self.assertEqual(summary['unconditional_retrieval_top1']['visual']['numerator'], 1)
        self.assertEqual(summary['conditional_top1_at_gt_in_k']['visual']['numerator'], 1)
        self.assertTrue(summary['oracle_text_included'])
        harmed = [dict(rows[0], gt_slug='one-2020')]
        old = [dict(self.baseline, gt_slug='one-2020')]
        self.assertEqual(comparison(old, harmed)['harmed_query_ids'], ['q'])
        policy = self.policy()
        accepted = self.evaluate(mode='clip', policy=policy)[0]
        self.assertEqual((accepted['status'], accepted['slug']), ('accepted', 'one-2021'))
        # Identical designs/vintages are not resolved by arbitrary rank order.
        candidates = self.baseline['candidates'][:2]
        identical = {c['wine_id']: {'cosine': 0.9} for c in candidates}
        ranked, _ = rank(candidates, identical)
        self.assertEqual(visual_decision(candidates, identical, ranked, policy)[0], 'ambiguous')
        identical[candidates[0]['wine_id']]['cosine'] = None
        self.assertEqual(rank(candidates, identical), (None, None))
        self.assertEqual(visual_decision(candidates, identical, None, policy)[2], ['missing_reference'])

    def test_visual_margin_policy_and_leakage(self):
        candidates = [dict(wine_id=x, slug=x, score=s, rank=i) for i, (x, s) in
                      enumerate([('A', 100), ('B', 90), ('C', 0)], 1)]
        obs = {x: {'cosine': cos} for x, cos in [('A', .8), ('B', .5), ('C', .95)]}
        clip, fusion = rank(candidates, obs)
        self.assertEqual(fusion[:2], ['A', 'B'])
        self.assertEqual(visual_decision(candidates, obs, fusion, self.policy(mode='fusion'))[0], 'ambiguous')
        self.assertEqual(visual_decision(candidates[:1], obs, ['A'], self.policy())[2],
                         ['no_visual_competitor'])
        with self.assertRaises(IncompatibleGallery):
            self.evaluate(mode='fusion', policy=self.policy())
        with self.assertRaises(IncompatibleGallery):
            self.evaluate(role='back', mode='clip', policy=self.policy())
        with self.assertRaises(IncompatibleGallery):
            self.evaluate(mode='clip', weight=0.4, policy=self.policy())
        with self.assertRaises(IncompatibleGallery):
            self.evaluate(mode='clip', top_k=10, policy=self.policy())
        with self.assertRaises(ValueError):
            evaluate(self.root/'query.jsonl', self.root/'baseline.jsonl', self.gallery,
                     lambda *a, **kw: None, mode='clip', policy=self.policy())
        with self.assertRaises(ValueError):
            VisualPolicy(**{**self.policy().payload(), 'artifact_sha256': '0'*64})
        rows = self.evaluate()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            comparison([self.baseline, self.baseline], rows)
        with self.assertRaisesRegex(ValueError, 'GT/source'):
            comparison([self.baseline], [dict(rows[0], source_group='another')])
        policy = self.policy()
        with self.assertRaisesRegex(ValueError, 'calibration manifest hash'):
            (self.root/'calibration.jsonl').write_text('{}\n')
            self.evaluate(mode='clip', policy=policy)
        (self.root/'calibration.jsonl').write_text(json.dumps(self.calibration)+'\n')
        other = dict(self.query, query_id='other', split='test', image_path='0.png',
                     source_group='other-shoot',
                     image_sha256=hashlib.sha256(self.samples[0]).hexdigest())
        (self.root/'query.jsonl').write_text(json.dumps(self.query)+'\n'+json.dumps(other)+'\n')
        with self.assertRaisesRegex(ValueError, 'leakage'):
            self.evaluate()
        other.update(image_path='2.png', image_sha256=self.query['image_sha256'],
                     source_group='different')
        (self.root/'query.jsonl').write_text(json.dumps(self.query)+'\n'+json.dumps(other)+'\n')
        with self.assertRaisesRegex(ValueError, 'across splits'):
            self.evaluate()

    def test_whole_manifest_gt_and_duplicate_audit(self):
        other = dict(self.query, split='test', query_id='q2', source_group='other-shoot')
        (self.root/'query.jsonl').write_text(json.dumps(self.query)+'\n'+json.dumps(other)+'\n')
        with self.assertRaisesRegex(ValueError, 'image hash across splits'):
            self.evaluate()
        other.update(image_path='calibration.png', image_sha256=self.calibration['image_sha256'],
                     gt_status='unknown', gt_slug='one-2021')
        (self.root/'query.jsonl').write_text(json.dumps(self.query)+'\n'+json.dumps(other)+'\n')
        with self.assertRaisesRegex(ValueError, 'GT must be null'):
            self.evaluate()
        other['query_id'] = 'q'
        (self.root/'query.jsonl').write_text(json.dumps(self.query)+'\n'+json.dumps(other)+'\n')
        with self.assertRaisesRegex(ValueError, 'duplicate id'):
            self.evaluate()

    def test_unknown_and_provenance_checks(self):
        self.query.update(gt_status='unknown', gt_slug=None)
        self.baseline.update(gt_status='unknown', gt_slug=None)
        self.save_query()
        self.save_baseline()
        rows = self.evaluate(mode='clip', policy=self.policy())
        self.assertEqual(comparison([self.baseline], rows)['visual']['unknown_false_accept']['numerator'], 1)
        self.query['source_group'] = 'gallery-0'
        self.baseline['source_group'] = 'gallery-0'
        self.save_query()
        self.save_baseline()
        with self.assertRaisesRegex(ValueError, 'leakage'):
            self.evaluate()
        self.query['source_group'] = 'query-shoot'
        self.baseline['source_group'] = 'query-shoot'
        self.query['image_sha256'] = hashlib.sha256(self.samples[0]).hexdigest()
        self.save_query()
        self.save_baseline()
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.evaluate()
        self.query['image_sha256'] = hashlib.sha256(self.samples[2]).hexdigest()
        self.save_query()
        self.baseline['versions']['csv_sha256'] = 'different'
        self.save_baseline()
        with self.assertRaises(IncompatibleGallery):
            self.evaluate()


if __name__ == '__main__':
    unittest.main()
