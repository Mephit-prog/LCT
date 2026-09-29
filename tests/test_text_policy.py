"""Versioned text accept policy bound to configuration and calibration (B05)."""
import json
import tempfile
import unittest
from pathlib import Path

from wineid.catalog import Wine, normalize
from wineid.ocr import MockOCR
from wineid.pipeline import Pipeline, Policy
from wineid.search import Index
from wineid.text_policy import (TEXT_POLICY_VERSION, TextPolicy, build_text_policy,
                                check_calibration_test_leakage, fingerprint, load_policy)

CSV_SHA = 'a' * 64
A, B, C, D = '1' * 64, '2' * 64, '3' * 64, '4' * 64


def wine(slug, name, producer):
    return Wine(slug, slug, name, producer, normalize(name), normalize(producer),
                (2,), {}, CSV_SHA)


def row(qid, group, split, digest, slug='wine-a'):
    return {'query_id': qid, 'image_path': f'{qid}.webp', 'image_sha256': digest,
            'source_group': group, 'split': split, 'gt_status': 'known',
            'gt_slug': slug, 'gt_source': 'independent annotation'}


class TextPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.index = Index([wine('wine-a', 'Алиготе', 'Завод А'),
                            wine('wine-b', 'Каберне', 'Завод Б')])
        self.config = Pipeline(self.index, MockOCR('Алиготе'),
                               csv_sha256=CSV_SHA,
                               roi_mode='full_frame_experimental').text_config

    def tearDown(self):
        self.temp.cleanup()

    def _manifest(self, name, rows):
        path = self.root / name
        path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n',
                        encoding='utf-8')
        return path

    def test_build_and_checksum(self):
        path = self._manifest('cal.jsonl', [row('q1', 's1', 'validation', A),
                                            row('q2', 's2', 'validation', B)])
        policy = build_text_policy(path, min_score=90, min_margin=5,
                                   max_distance=0.1, config=self.config)
        self.assertEqual(policy.version, TEXT_POLICY_VERSION)
        self.assertEqual(policy.version, 'text-policy-v3')
        self.assertEqual(policy.artifact_sha256, fingerprint(policy.payload()))
        self.assertEqual(policy.calibration_source_groups, ['s1', 's2'])

    def test_build_rejects_non_validation_and_bad_manifest(self):
        bad_split = self._manifest('test.jsonl', [row('q1', 's1', 'test', A)])
        with self.assertRaises(ValueError):
            build_text_policy(bad_split, min_score=1, min_margin=1,
                              max_distance=0.1, config=self.config)
        with self.assertRaises(ValueError):
            build_text_policy(self.root / 'missing.jsonl', min_score=1, min_margin=1,
                              max_distance=0.1, config=self.config)

    def test_invalid_artifact_checksum_and_config(self):
        path = self._manifest('cal.jsonl', [row('q1', 's1', 'validation', A)])
        policy = build_text_policy(path, min_score=90, min_margin=5,
                                   max_distance=0.1, config=self.config)
        with self.assertRaises(ValueError):
            TextPolicy(version=policy.version, min_score=policy.min_score,
                       min_margin=policy.min_margin, max_distance=policy.max_distance,
                       config=policy.config,
                       calibration_manifest_sha256=policy.calibration_manifest_sha256,
                       calibration_source_groups=policy.calibration_source_groups,
                       artifact_sha256='0' * 64)
        with self.assertRaises(ValueError):
            TextPolicy(version=policy.version, min_score=policy.min_score,
                       min_margin=policy.min_margin, max_distance=policy.max_distance,
                       config={'csv_sha256': CSV_SHA},
                       calibration_manifest_sha256=policy.calibration_manifest_sha256,
                       calibration_source_groups=policy.calibration_source_groups,
                       artifact_sha256=policy.artifact_sha256)

    def test_check_config_rejects_mismatch(self):
        path = self._manifest('cal.jsonl', [row('q1', 's1', 'validation', A)])
        policy = build_text_policy(path, min_score=90, min_margin=5,
                                   max_distance=0.1, config=self.config)
        with self.assertRaises(ValueError):
            policy.check_config({**self.config, 'csv_sha256': 'b' * 64})

    def test_pipeline_binds_policy_to_configuration(self):
        path = self._manifest('cal.jsonl', [row('q1', 's1', 'validation', A)])
        policy = build_text_policy(path, min_score=0, min_margin=0,
                                   max_distance=1, config=self.config)
        # Matching catalog/backend/blend -> constructed, and accepts.
        pipeline = Pipeline(self.index, MockOCR('Алиготе'), policy=policy,
                            csv_sha256=CSV_SHA, roi_mode='full_frame_experimental')
        self.assertEqual(pipeline.text_config, self.config)
        from unittest.mock import patch
        # Health may check a cached startup snapshot; it must not rehash OCR
        # binaries, traineddata or the decision-source tree on every poll.
        with patch.object(Pipeline, 'current_text_config', property(lambda _: (_ for _ in ()).throw(
                AssertionError('health re-read heavyweight profile')))):
            self.assertTrue(pipeline.acceptance_ready)
        # Mismatched catalog checksum -> refused at construction.
        with self.assertRaises(ValueError):
            Pipeline(self.index, MockOCR('Алиготе'), policy=policy,
                     csv_sha256='b' * 64, roi_mode='full_frame_experimental')

    def test_check_calibration_test_leakage(self):
        cal = self._manifest('cal.jsonl', [row('q1', 's1', 'validation', A)])
        policy = build_text_policy(cal, min_score=1, min_margin=1,
                                   max_distance=0.1, config=self.config)
        shared = self._manifest('test.jsonl', [row('t1', 's1', 'test', B)])
        with self.assertRaises(ValueError):
            check_calibration_test_leakage(policy, cal, shared)
        leak_hash = self._manifest('test2.jsonl', [row('t2', 's2', 'test', A)])
        with self.assertRaises(ValueError):
            check_calibration_test_leakage(policy, cal, leak_hash)
        clean = self._manifest('test3.jsonl', [row('t3', 's3', 'test', C)])
        check_calibration_test_leakage(policy, cal, clean)  # no raise

    def test_load_policy_text_and_legacy(self):
        path = self._manifest('cal.jsonl', [row('q1', 's1', 'validation', A)])
        policy = build_text_policy(path, min_score=90, min_margin=5,
                                   max_distance=0.1, config=self.config)
        artifact = self.root / 'policy.json'
        artifact.write_text(json.dumps({**policy.payload(),
                                        'artifact_sha256': policy.artifact_sha256},
                                       ensure_ascii=False), encoding='utf-8')
        loaded = load_policy(artifact)
        self.assertIsInstance(loaded, TextPolicy)
        self.assertEqual(loaded.artifact_sha256, policy.artifact_sha256)
        legacy_path = self.root / 'legacy.json'
        legacy_path.write_text(json.dumps({'min_score': 95, 'min_margin': 8,
                                           'max_distance': 0.05,
                                           'version': 'validation-2025-01'}))
        legacy = load_policy(legacy_path)
        self.assertIsInstance(legacy, Policy)
        with self.assertRaisesRegex(ValueError, 'legacy policy'):
            load_policy(legacy_path, allow_legacy=False)
        self.assertEqual(legacy.version, 'validation-2025-01')


if __name__ == '__main__':
    unittest.main()
