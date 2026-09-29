"""Near-duplicate manual-review report (clip_review_improvement.md P2)."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from wineid.duplicates import near_duplicate_report


def noise_image(size=400, seed=0, shift=0):
    array = np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype='uint8')
    image = Image.fromarray(array)
    return image.point(lambda v: min(255, v + shift)) if shift else image


class NearDuplicateReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        noise_image(seed=1).save(self.root / 'a.png')
        noise_image(seed=1, shift=1).save(self.root / 'b.png')
        noise_image(seed=2).save(self.root / 'c.png')

    def tearDown(self):
        self.temp.cleanup()

    def test_reports_near_duplicate_pair_for_review(self):
        rows = [{'query_id': 'a', 'image_path': 'a.png', 'source_group': 's1', 'split': 'validation'},
                {'query_id': 'b', 'image_path': 'b.png', 'source_group': 's2', 'split': 'test'},
                {'query_id': 'c', 'image_path': 'c.png', 'source_group': 's3', 'split': 'test'}]
        report = near_duplicate_report(rows, self.root, max_distance=6)
        self.assertEqual(report['rows'], 3)
        self.assertEqual(report['hashed'], 3)
        pairs = {(p['left'], p['right']) for p in report['near_duplicates']}
        self.assertIn(('a', 'b'), pairs)
        pair = next(p for p in report['near_duplicates'] if p['left'] == 'a')
        self.assertTrue(pair['cross_split'])
        self.assertFalse(pair['same_group'])
        self.assertIn('Manual review only', report['note'])

    def test_same_group_not_flagged_as_cross(self):
        rows = [{'query_id': 'a', 'image_path': 'a.png', 'source_group': 's1', 'split': 'validation'},
                {'query_id': 'b', 'image_path': 'b.png', 'source_group': 's1', 'split': 'validation'}]
        report = near_duplicate_report(rows, self.root, max_distance=6)
        self.assertTrue(all(p['same_group'] for p in report['near_duplicates']))
        self.assertTrue(all(not p['cross_split'] for p in report['near_duplicates']))

    def test_missing_and_unsafe_paths_are_reported_not_raised(self):
        rows = [{'query_id': 'missing', 'image_path': 'nope.png'},
                {'query_id': 'unsafe', 'image_path': '../escape.png'}]
        report = near_duplicate_report(rows, self.root)
        self.assertEqual(report['hashed'], 0)
        reasons = {item['id'] for item in report['unreadable']}
        self.assertEqual(reasons, {'missing', 'unsafe'})

    def test_cli_input_is_jsonl(self):
        # The report consumes the documented JSONL schema without side effects.
        rows = [{'query_id': 'a', 'image_path': 'a.png'}]
        path = self.root / 'manifest.jsonl'
        path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n', encoding='utf-8')
        loaded = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        self.assertEqual(near_duplicate_report(loaded, path.parent)['rows'], 1)


if __name__ == '__main__':
    unittest.main()
