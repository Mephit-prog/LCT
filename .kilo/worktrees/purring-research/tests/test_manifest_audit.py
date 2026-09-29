"""Shared manifest-audit contracts (STATUS.md B03): retrieval manifest and
calibration/test isolation, reused by the visual and text evaluators."""
import unittest

from wineid.data import (calibration_test_leakage_errors, query_manifest_errors,
                         retrieval_manifest_errors)
from wineid.clip_zero_shot import retrieval_report

A, B, C = 'a' * 64, 'b' * 64, 'c' * 64


def manifest_row(qid, group, split, digest, status='known', slug='wine-a',
                 source='independent annotation'):
    return {'query_id': qid, 'source_group': group, 'split': split,
            'image_sha256': digest, 'gt_status': status, 'gt_slug': slug,
            'gt_source': source}


class RetrievalManifestTests(unittest.TestCase):
    def _retrieval(self, **overrides):
        row = {'query_id': 'q1', 'text': 'Вино',
               'relevant_image_ids': ['a.jpg'], 'hard_negative': False}
        row.update(overrides)
        return [row]

    def test_valid_manifest(self):
        self.assertEqual(retrieval_manifest_errors(self._retrieval(), {'a.jpg'}), [])

    def test_missing_and_duplicate_query_id(self):
        rows = self._retrieval() + self._retrieval()
        self.assertTrue(any('duplicate query_id' in e for e in
                            retrieval_manifest_errors(rows, {'a.jpg'})))

    def test_empty_text_and_bad_relevant_ids(self):
        rows = self._retrieval(text='  ', relevant_image_ids=['missing.jpg'])
        errors = retrieval_manifest_errors(rows, {'a.jpg'})
        self.assertTrue(any('empty text' in e for e in errors))
        self.assertTrue(any('relevant_image_ids' in e for e in errors))

    def test_hard_negative_type(self):
        rows = self._retrieval(hard_negative='yes')
        self.assertTrue(any('hard_negative' in e for e in
                            retrieval_manifest_errors(rows, {'a.jpg'})))

    def test_optional_split_and_group_leakage(self):
        rows = [
            {'query_id': 'q1', 'text': 'Вино', 'relevant_image_ids': ['a.jpg'],
             'split': 'validation', 'source_group': 's1', 'image_sha256': A},
            {'query_id': 'q2', 'text': 'Вино', 'relevant_image_ids': ['a.jpg'],
             'split': 'test', 'source_group': 's1', 'image_sha256': A},
        ]
        errors = retrieval_manifest_errors(rows, {'a.jpg'})
        self.assertTrue(any('source_group across splits' in e for e in errors))
        self.assertTrue(any('image hash across splits' in e for e in errors))


class FakeIndex:
    kind = 'images'
    items = [{'id': 'a.jpg'}]

    def check_encoder(self, encoder):
        pass

    def rank(self, vectors, top_k=5):
        return [[{'id': 'a.jpg', 'rank': 1, 'cosine': 0.9}] for _ in vectors]


class FakeEncoder:
    batch_size = 8
    config = {'dimensions': 2}

    def encode_text(self, texts):
        return [[1.0, 0.0] for _ in texts]


class RetrievalReportIntegrationTests(unittest.TestCase):
    def test_report_rejects_invalid_manifest(self):
        bad = [{'query_id': 'q1', 'text': 'Вино',
                'relevant_image_ids': ['nope.jpg']}]
        with self.assertRaises(ValueError):
            retrieval_report(FakeIndex(), bad, FakeEncoder())

    def test_report_accepts_valid_manifest(self):
        good = [{'query_id': 'q1', 'text': 'Вино', 'relevant_image_ids': ['a.jpg']}]
        report = retrieval_report(FakeIndex(), good, FakeEncoder())
        self.assertEqual(report['all']['recall_at_1']['value'], 1.0)


class CalibrationLeakageTests(unittest.TestCase):
    def test_shared_source_group(self):
        errors = calibration_test_leakage_errors(
            [manifest_row('c1', 'shoot-1', 'validation', A)],
            [manifest_row('t1', 'shoot-1', 'test', B)])
        self.assertIn('calibration/test shared source_group', errors)

    def test_shared_image_hash(self):
        errors = calibration_test_leakage_errors(
            [manifest_row('c1', 'shoot-1', 'validation', A)],
            [manifest_row('t1', 'shoot-2', 'test', A)])
        self.assertIn('calibration/test shared image hash', errors)

    def test_shared_query_id(self):
        errors = calibration_test_leakage_errors(
            [manifest_row('same', 'shoot-1', 'validation', A)],
            [manifest_row('same', 'shoot-2', 'test', B)])
        self.assertIn('calibration/test shared query_id', errors)

    def test_same_manifest_path(self):
        errors = calibration_test_leakage_errors([], [], calibration_path='x.jsonl',
                                                 test_path='./x.jsonl')
        self.assertIn('calibration and test are the same manifest', errors)

    def test_disjoint_manifests(self):
        errors = calibration_test_leakage_errors(
            [manifest_row('c1', 'shoot-1', 'validation', A)],
            [manifest_row('t1', 'shoot-2', 'test', B)])
        self.assertEqual(errors, [])


class QueryManifestTests(unittest.TestCase):
    def test_known_requires_source_and_slug(self):
        errors, _ = query_manifest_errors([manifest_row('q1', 'g1', 'validation', A,
                                                        source=None)])
        self.assertTrue(any('independent GT missing' in e for e in errors))

    def test_unknown_must_have_null_slug(self):
        errors, _ = query_manifest_errors([manifest_row('q1', 'g1', 'validation', A,
                                                        status='unknown', slug='x')])
        self.assertTrue(any('GT must be null' in e for e in errors))

    def test_valid_row_and_groups(self):
        errors, groups = query_manifest_errors([manifest_row('q1', 'g1', 'test', A)])
        self.assertEqual(errors, [])
        self.assertEqual(groups, {'g1': 'test'})


if __name__ == '__main__':
    unittest.main()
