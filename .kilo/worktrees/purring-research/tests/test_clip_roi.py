import hashlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from wineid.clip_roi import cam_bbox, clip_cam_roi, experiment, summary, VERSION
from wineid.roi import InvalidImage, AmbiguousScene


class CLIPROITests(unittest.TestCase):
    def test_geometry_letterbox_and_color_crop(self):
        image = Image.new('RGB', (200, 100), 'blue')
        buf = io.BytesIO()
        image.save(buf, 'PNG')
        data = buf.getvalue()
        def cam(square):
            self.assertEqual(square.size, (200, 200))
            self.assertEqual(square.getpixel((0, 0)), (255, 255, 255))
            heat = np.zeros((10, 10), dtype='float32')
            heat[4:6, 3:7] = 1
            return heat
        roi = clip_cam_roi(data, time.monotonic()+3, cam)
        self.assertEqual(roi.bbox, (34, 6, 166, 94))
        self.assertEqual(roi.crop_version, VERSION)
        self.assertEqual(roi.status, 'fallback_bbox')
        self.assertEqual(Image.open(io.BytesIO(roi.crop)).getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(cam_bbox(cam(Image.new('RGB', (200, 200), 'white')), (200, 100)),
                         (40, 10, 160, 90))

    def test_abstentions(self):
        with self.assertRaisesRegex(InvalidImage, 'empty_cam'):
            cam_bbox(np.zeros((10, 10)), (200, 100))
        with self.assertRaisesRegex(InvalidImage, 'empty_cam'):
            cam_bbox(np.ones((10, 10)), (200, 100))
        heat = np.zeros((10, 10), dtype='float32')
        heat[1:3, 1:3] = heat[7:9, 7:9] = 1
        with self.assertRaises(AmbiguousScene):
            cam_bbox(heat, (200, 100))
        with self.assertRaisesRegex(InvalidImage, 'invalid_cam'):
            cam_bbox(np.full((10, 10), np.nan), (200, 100))
        with self.assertRaises(ValueError):
            cam_bbox(heat, (200, 100), threshold=2)

    def test_offline_experiment_hash_provenance_and_denominators(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            buf = io.BytesIO()
            Image.new('RGB', (100, 100), 'red').save(buf, 'PNG')
            data = buf.getvalue()
            (base/'sample.png').write_bytes(data)
            row = {'query_id': 'q1', 'image_path': 'sample.png',
                   'image_sha256': hashlib.sha256(data).hexdigest(),
                   'bbox': [20, 20, 80, 80], 'split': 'validation',
                   'source_group': 'shoot-1', 'gt_status': 'known',
                   'gt_slug': 'wine', 'gt_source': 'manual check'}
            manifest = base/'query.jsonl'
            manifest.write_text(json.dumps(row)+'\n')
            def cam(_):
                heat = np.zeros((10, 10), dtype='float32')
                heat[3:7, 3:7] = 1
                return heat
            results = experiment(manifest, cam)
            self.assertEqual(summary(results)['total'], 1)
            self.assertEqual(results[0]['status'], 'fallback_bbox')
            self.assertGreater(results[0]['gt_coverage'], .9)
            refused = experiment(manifest, lambda _: np.zeros((10, 10)))
            self.assertEqual(summary(refused)['located'], 0)
            self.assertEqual(refused[0]['reason'], 'empty_cam')
            row.update(label_present=False, bbox=None)
            manifest.write_text(json.dumps(row)+'\n')
            negative = experiment(manifest, cam)
            self.assertEqual(summary(negative)['false_localizations'], 1)
            self.assertIsNone(negative[0]['iou'])
            row['image_sha256'] = 'wrong'
            manifest.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                experiment(manifest, cam)
            row['image_sha256'] = hashlib.sha256(data).hexdigest()
            row['image_path'] = '../outside.png'
            manifest.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError, 'unsafe path'):
                experiment(manifest, cam)


if __name__ == '__main__':
    unittest.main()
