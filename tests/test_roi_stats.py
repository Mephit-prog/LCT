"""ROI retention geometry (plans/jina_clip_roi_catalog_enrichment.md §2)."""
import unittest

from wineid.roi_stats import (box_stats, field_retentions, frame_fov, intersection,
                              iou, native_center_crop_fov, retention, retention_report)


class GeometryTests(unittest.TestCase):
    def test_native_fov_matches_documented_crop_loss(self):
        # Jina shortest+center-crop on a tall product shot (documented example).
        fov = native_center_crop_fov(1950, 7371)
        self.assertEqual(fov, (0, 2710, 1950, 4660))
        self.assertAlmostEqual(retention((0, 0, 1950, 7371), fov), 1950 / 7371, places=4)
        self.assertAlmostEqual(1950 / 7371, 0.2646, places=4)
        narrow = native_center_crop_fov(260, 1000)
        self.assertEqual(narrow, (0, 370, 260, 630))
        self.assertAlmostEqual(retention((0, 0, 260, 1000), narrow), 0.26, places=4)

    def test_native_fov_square_keeps_everything(self):
        fov = native_center_crop_fov(512, 512)
        self.assertEqual(fov, (0, 0, 512, 512))
        self.assertEqual(retention((0, 0, 512, 512), fov), 1.0)

    def test_retention_and_iou(self):
        gt = (0, 0, 100, 100)
        self.assertEqual(retention(gt, (50, 0, 150, 100)), 0.5)
        self.assertEqual(retention(gt, (200, 200, 300, 300)), 0.0)
        self.assertEqual(retention(gt, (0, 0, 200, 200)), 1.0)
        self.assertAlmostEqual(iou(gt, (50, 0, 150, 100)), 50 * 100 / (100 * 100 + 100 * 100 - 50 * 100))
        self.assertEqual(intersection(gt, (200, 200, 300, 300)), 0)

    def test_box_stats(self):
        stats = box_stats((100, 200, 300, 400), (1000, 1000))
        self.assertAlmostEqual(stats['label_area_fraction'], 0.04)
        self.assertEqual(stats['label_width_px'], 200)
        self.assertEqual(stats['label_height_px'], 200)
        self.assertEqual(stats['label_aspect'], 1.0)
        self.assertEqual(stats['centroid'], [0.2, 0.3])

    def test_field_retentions(self):
        got = field_retentions({'name': (0, 0, 100, 50), 'vintage': (0, 100, 100, 150)},
                               (0, 0, 100, 75))
        self.assertEqual(got['name'], 1.0)
        self.assertEqual(got['vintage'], 0.0)

    def test_frame_fov_keeps_all(self):
        self.assertEqual(frame_fov(400, 600), (0, 0, 400, 600))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            native_center_crop_fov(0, 10)
        with self.assertRaises(ValueError):
            retention_report([{'query_id': 'q', 'image_size': [0, 10],
                               'label_bbox': [0, 0, 1, 1]}])


class RetentionReportTests(unittest.TestCase):
    def _rows(self):
        return [
            {'query_id': 'tall', 'image_size': [1950, 7371], 'source_group': 's1',
             # Label near the top: the native center crop keeps only y 2710..4660.
             'label_bbox': [0, 500, 1950, 1500],
             'field_bboxes': {'name': [200, 700, 1800, 900]}},
            {'query_id': 'square', 'image_size': [800, 800], 'source_group': 's2',
             'label_bbox': [100, 100, 700, 700],
             'field_bboxes': {'name': [200, 200, 600, 600]}},
        ]

    def test_native_report_low_retention_for_tall_shot(self):
        report = retention_report(self._rows(), mode='native')
        self.assertEqual(report['queries'], 2)
        self.assertEqual(report['source_groups'], 2)
        per_query = {r['query_id']: r for r in report['per_query']}
        self.assertEqual(per_query['square']['label_retention'], 1.0)
        self.assertLess(per_query['tall']['label_retention'], 0.5)
        self.assertEqual(report['retained_below_0_9'], {'numerator': 1, 'denominator': 2})

    def test_pad_report_keeps_all(self):
        report = retention_report(self._rows(), mode='pad')
        self.assertTrue(all(r['label_retention'] == 1.0 for r in report['per_query']))
        self.assertEqual(report['label_retention']['min'], 1.0)

    def test_bbox_mode_requires_model_bbox(self):
        with self.assertRaises(ValueError):
            retention_report(self._rows(), mode='bbox')
        rows = self._rows()
        rows[0]['model_bbox'] = [0, 500, 1950, 1500]
        rows[1]['model_bbox'] = [0, 0, 800, 800]
        report = retention_report(rows, mode='bbox')
        self.assertEqual(report['per_query'][0]['label_retention'], 1.0)

    def test_report_is_not_a_quality_claim(self):
        report = retention_report(self._rows(), mode='native')
        self.assertIn('not a recognition or OCR quality metric', report['note'])


if __name__ == '__main__':
    unittest.main()
