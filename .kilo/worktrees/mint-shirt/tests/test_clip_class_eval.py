import contextlib
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from wineid.clip_class_eval import audit_class_queries, class_report, compare_class_reports
from wineid.clip_zero_shot import EmbeddingIndex, build_classes, build_images, main
from wineid.jina_clip import REVISION


class FakeEncoder:
    config = {'model': 'fake', 'revision': REVISION, 'dimensions': 2}
    batch_size = 2

    def __init__(self):
        self.image_batches = []

    def encode_text(self, texts):
        return np.array([[1., 0.] if 'Красное' in text else [0., 1.] for text in texts])

    def encode_images(self, images):
        self.image_batches.append(len(images))
        return np.array([[1., 0.] if image.getpixel((0, 0))[0] > 100 else [0., 1.]
                         for image in images])


class ClassEvalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.encoder = FakeEncoder()
        classes = [dict(class_id='red', name='Красное'), dict(class_id='red-2', name='Красное'),
                   dict(class_id='blue', name='Синее')]
        self.index = build_classes(classes, self.encoder, source_sha256='a'*64)
        self.manifest = self.root / 'query.jsonl'
        self.rows = [self.row('q1', 'red.png', 'red', 'shoot-1', 'validation', 'known', 'red'),
                     self.row('q2', 'blue.png', 'blue', 'shoot-2', 'validation', 'known', 'blue'),
                     self.row('q3', 'green.png', 'green', 'shoot-3', 'validation', 'unknown', None),
                     self.row('q4', 'yellow.png', 'yellow', 'shoot-4', 'validation', 'ambiguous', None),
                     self.row('q5', 'test.png', 'blue', 'shoot-5', 'test', 'known', 'red-2')]
        self.save()

    def row(self, qid, filename, color, group, split, status, slug):
        path = self.root / filename
        Image.new('RGB', (8, 9 if filename == 'test.png' else 8), color).save(path)
        return {'query_id': qid, 'image_path': filename,
                'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'source_group': group, 'split': split, 'gt_status': status,
                'gt_slug': slug, 'gt_source': 'independent human annotation'}

    def save(self):
        self.manifest.write_text(''.join(json.dumps(row)+'\n' for row in self.rows))

    def test_ranking_metrics_unknown_and_collision_not_accepted(self):
        result = class_report(self.index, self.manifest, self.encoder)
        self.assertEqual(self.encoder.image_batches, [2, 2])  # test split not encoded
        metrics = result['metrics']
        self.assertEqual((metrics['known'], metrics['unknown'], metrics['ambiguous_gt']), (2, 1, 1))
        self.assertEqual(metrics['raw_top1_known']['numerator'], 2)
        self.assertEqual(metrics['raw_top5_known']['numerator'], 2)
        self.assertEqual(metrics['known_gt_in_prompt_collision']['numerator'], 1)
        self.assertEqual(metrics['known_gt_in_normalized_prompt_collision']['numerator'], 1)
        self.assertEqual(result['prompt_collisions']['exact']['classes'], 2)
        self.assertEqual(result['prompt_collisions']['normalized']['groups'], 1)
        self.assertEqual(metrics['known_source_groups'], 2)
        self.assertEqual(metrics['phash_band_8_16_pairs'], 0)
        self.assertIsNone(metrics['raw_top1_known_phash_band_8_16']['value'])
        self.assertIsInstance(result['queries'][0]['phash'], int)
        self.assertIsNotNone(metrics['raw_top1_known_group_interval'])
        red = result['queries'][0]
        self.assertTrue(red['top1_correct'])
        self.assertIsNone(red['predicted_class_id'])
        self.assertEqual(red['indistinguishable_class_ids'], ['red', 'red-2'])
        self.assertEqual(red['status'], 'ambiguous_prompt')
        self.assertIsNone(result['queries'][2]['gt_slug'])
        self.assertNotIn('unknown_false_accept', metrics)
        self.assertEqual(result['config']['query_manifest_sha256'],
                         hashlib.sha256(self.manifest.read_bytes()).hexdigest())
        test = class_report(self.index, self.manifest, self.encoder, split='test')
        self.assertEqual(test['metrics']['raw_top1_known']['numerator'], 0)
        self.assertEqual(test['metrics']['wrong_top1_outside_gt_prompt_group']['numerator'], 1)
        self.assertIsNone(test['metrics']['raw_top1_known_group_interval'])
        self.assertEqual(test['queries'][0]['gt_prompt_collision'], ['red', 'red-2'])

    def test_audit_entire_manifest_even_when_evaluating_validation(self):
        def reject(edit, expected):
            original = copy.deepcopy(self.rows)
            try:
                edit()
                self.save()
                with self.assertRaisesRegex(ValueError, expected):
                    class_report(self.index, self.manifest, self.encoder)
            finally:
                self.rows = original
                self.save()
        reject(lambda: self.rows[4].update(source_group='shoot-1'), 'source_group across splits')
        reject(lambda: self.rows[4].update(image_sha256=self.rows[0]['image_sha256']),
               'image hash across splits')
        reject(lambda: self.rows[4].update(query_id='q1'), 'duplicate id')
        reject(lambda: self.rows[4].update(query_id=[], source_group=[], image_sha256=[]),
               'missing/duplicate id')
        reject(lambda: self.rows[4].update(gt_slug='missing'), 'independent GT missing')
        reject(lambda: self.rows[4].update(gt_status='unknown'), 'GT must be null')
        reject(lambda: self.rows[2].update(gt_source=''), 'independent GT source')
        reject(lambda: self.rows[4].update(image_sha256='0'*64), 'hash mismatch')
        reject(lambda: self.rows[4].update(image_path='../red.png'), 'unsafe/missing image path')
        reject(lambda: self.rows[4].update(image_path='missing.png'), 'missing/unsafe image')
        reject(lambda: self.rows[4].update(image_path='red.png', image_sha256=self.rows[0]['image_sha256'],
                                           source_group='shoot-5', split='validation'),
               'duplicate image')
        self.assertEqual(len(audit_class_queries(self.index, self.manifest)[0]), 5)
        with self.assertRaisesRegex(ValueError, 'class index'):
            audit_class_queries(build_images(self.root, self.encoder), self.manifest)

    def test_normalized_prompt_audit(self):
        rows = [dict(class_id='first', name='Красное'), dict(class_id='second', name='красное  ')]
        index = build_classes(rows, self.encoder)
        # Both prompts differ byte-for-byte; normalized collision is counted separately.
        self.rows = [self.row('q1', 'only.png', 'red', 'one', 'validation', 'known', 'first')]
        self.save()
        result = class_report(index, self.manifest, self.encoder)
        self.assertEqual(result['prompt_collisions']['exact']['classes'], 0)
        self.assertEqual(result['prompt_collisions']['normalized']['classes'], 2)
        self.assertEqual(result['metrics']['known_gt_in_normalized_prompt_collision']['numerator'], 1)

    def test_pairing_and_rejected_mismatch(self):
        old = class_report(self.index, self.manifest, self.encoder)
        swapped = EmbeddingIndex(self.index.vectors[[2, 1, 0]], self.index.meta)
        new = class_report(swapped, self.manifest, self.encoder)
        paired = compare_class_reports(old, new)
        self.assertEqual(paired['helped_query_ids'], [])
        self.assertEqual(paired['harmed_query_ids'], ['q1', 'q2'])
        self.assertEqual(paired['harmed_among_known']['denominator'], 2)
        self.assertEqual(compare_class_reports(new, old)['helped_query_ids'], ['q1', 'q2'])
        self.assertEqual(compare_class_reports(old, old)['harmed_query_ids'], [])
        modified = copy.deepcopy(new)
        modified['queries'][0]['image_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'image/GT mismatch'):
            compare_class_reports(old, modified)
        modified = copy.deepcopy(new)
        modified['queries'].append(modified['queries'][0])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            compare_class_reports(old, modified)
        modified = copy.deepcopy(new)
        modified['config']['encoder'] = {'model': 'changed'}
        with self.assertRaisesRegex(ValueError, 'encoder'):
            compare_class_reports(old, modified)
        modified = copy.deepcopy(new)
        modified['config']['class_ids_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'class_ids_sha256'):
            compare_class_reports(old, modified)
        modified = copy.deepcopy(old)
        modified['config']['source_sha256'] = None
        with self.assertRaisesRegex(ValueError, 'source snapshot'):
            compare_class_reports(modified, new)

    def test_cli_preflight_before_loading_weights_and_model_free_comparison(self):
        path = self.root/'classes.npz'
        self.index.save(path)
        self.rows[4]['image_sha256'] = '0'*64
        self.save()
        with patch('wineid.clip_zero_shot.JinaCLIPEmbedder') as model, \
                patch.object(sys, 'argv', ['clip_zero_shot', 'evaluate-classes', str(path),
                                          str(self.manifest), '--allow-remote-code']):
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                main()
            model.assert_not_called()
        self.rows[4]['image_sha256'] = hashlib.sha256((self.root/'test.png').read_bytes()).hexdigest()
        self.save()
        with patch('wineid.clip_zero_shot.JinaCLIPEmbedder', return_value=self.encoder), \
                patch.object(sys, 'argv', ['clip_zero_shot', 'evaluate-classes', str(path),
                                          str(self.manifest), '--allow-remote-code']), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            main()
        result = json.loads(output.getvalue())
        self.assertEqual(result['metrics']['known'], 2)
        self.assertEqual(len(result['config']['index_archive_sha256']), 64)
        report = self.root/'report.json'
        report.write_text(json.dumps(result))
        with patch.object(sys, 'argv', ['clip_zero_shot', 'compare-classes', str(report), str(report)]), \
                contextlib.redirect_stdout(io.StringIO()) as compared:
            main()
        self.assertEqual(json.loads(compared.getvalue())['known'], 2)


if __name__ == '__main__':
    unittest.main()
