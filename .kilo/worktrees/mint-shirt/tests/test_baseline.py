import csv
import hashlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image, ImageDraw
from wineid.catalog import Wine, load_catalog, normalize
from wineid.search import Index, levenshtein
from wineid.roi import (decode, manual_roi, auto_bbox_roi,
                        center_crop_80_roi, InvalidImage, AmbiguousScene)
from wineid.pipeline import Pipeline, Policy
from wineid.ocr import MockOCR, MistralOCR
from wineid.server import multipart_image, create_handler
from wineid.data import audit
from wineid.compare import compare
from wineid.media import inventory
from wineid.visual import Gallery, IncompatibleGallery, build_gallery, MODEL, PRETRAINED
import numpy as np
from http.server import ThreadingHTTPServer
import threading
import httpx
from wineid.report import report


def wine(slug, name, producer):
    return Wine(slug, slug, name, producer, normalize(name), normalize(producer), (2,), {}, 'test')


def picture(fmt='WEBP'):
    buffer = io.BytesIO()
    Image.new('RGB', (120, 90), 'white').save(buffer, format=fmt)
    return buffer.getvalue()


class Baseline(unittest.TestCase):
    def setUp(self):
        self.index = Index([wine('a-2020', 'Вино 2020', 'Завод А'),
                            wine('a-2021', 'Вино 2021', 'Завод А'),
                            wine('b-2020', 'Вино 2020', 'Завод Б')])
        self.image = picture()

    def test_catalog_duplicates_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'data.csv'
            with path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['Slug', 'Название вина', 'Винодельня', 'Описание'])
                writer.writeheader()
                writer.writerow({'Slug':'s','Название вина':'Ёж','Винодельня':'А', 'Описание':'line\nnext'})
                writer.writerow({'Slug':'s','Название вина':'Ёж','Винодельня':'А', 'Описание':'line\nnext'})
            wines, stats = load_catalog(path)
            self.assertEqual(wines[0].source_row_ids, (2, 3))
            self.assertEqual(stats['exact_duplicates'], 1)
            with path.open('a', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow(['s', 'other', 'A', 'x'])
            with self.assertRaisesRegex(ValueError, 'conflicting'):
                load_catalog(path)

    def test_search_and_distance(self):
        self.assertEqual(levenshtein('kitten', 'sitting'), 3)
        self.assertEqual(levenshtein('', 'wine'), 4)
        self.assertEqual(self.index.search(''), [])
        matches = self.index.search('Вино 2021\nЗавод А', k=3)
        self.assertEqual(matches[0].slug, 'a-2021')
        self.assertEqual(len({m.wine_id for m in matches}), 3)
        self.assertNotEqual(matches[0].slug, matches[1].slug)

    def test_webp_mislabeled_and_bbox(self):
        image, mime = decode(self.image, time.monotonic()+5)
        self.assertEqual(mime, 'image/webp')
        crop = manual_roi(self.image, (0, 0, 110, 80), time.monotonic()+5)
        self.assertEqual(crop.mime, 'image/png')
        self.assertEqual(crop.bbox[0:2], (0, 0))
        with self.assertRaises(InvalidImage):
            manual_roi(self.image, (-1, 0, 90, 80), time.monotonic()+5)
        with self.assertRaises(InvalidImage):
            decode(b'not an image', time.monotonic()+5)

    def test_auto_bbox_experimental(self):
        def scene(n):
            image = Image.new('RGB', (400, 320), '#a6a6a6')
            pen = ImageDraw.Draw(image)
            for i in range(n):
                x = 40+200*i
                pen.rectangle((x, 80, x+120, 180), fill='white', outline='black', width=4)
            buf = io.BytesIO()
            image.save(buf, 'WEBP')
            return buf.getvalue()
        with self.assertRaisesRegex(InvalidImage, 'no_label_bbox'):
            auto_bbox_roi(scene(0), time.monotonic()+5)
        roi = auto_bbox_roi(scene(1), time.monotonic()+5)
        self.assertEqual(roi.status, 'fallback_bbox')
        self.assertTrue(roi.bbox[0] <= 40 < roi.bbox[2])
        with self.assertRaises(AmbiguousScene):
            auto_bbox_roi(scene(2), time.monotonic()+5)
        pipe = Pipeline(self.index, MockOCR('Вино 2021\\nЗавод А'),
                        roi_mode='auto_bbox_experimental')
        self.assertEqual(pipe.predict(scene(2))['status'], 'ambiguous_scene')
        result = pipe.predict(scene(1))
        self.assertEqual(result['status'], 'ambiguous')
        self.assertEqual(result['roi']['crop_version'], 'oriented-rgb-auto-bbox-v1')

    def test_center_crop_80_roi(self):
        roi = center_crop_80_roi(self.image, time.monotonic() + 5)
        self.assertEqual(roi.status, 'fallback_bbox')
        self.assertEqual(roi.crop_version, 'oriented-rgb-center-80-v1')
        w, h = roi.source_size
        self.assertEqual(roi.bbox, (round(w * 0.1), round(h * 0.1),
                                    round(w * 0.9), round(h * 0.9)))
        pipe = Pipeline(self.index, MockOCR('Вино 2021\nЗавод А'),
                        roi_mode='center_80_crop')
        result = pipe.predict(self.image)
        self.assertEqual(result['status'], 'ambiguous')
        self.assertEqual(result['roi']['crop_version'], 'oriented-rgb-center-80-v1')

    def test_pipeline_refuses_auto_or_uncalibrated(self):
        pipe = Pipeline(self.index, MockOCR('Вино 2021\nЗавод А'))
        self.assertEqual(pipe.predict(self.image)['status'], 'unreadable')
        result = pipe.predict(self.image, bbox=(0,0,100,80))
        self.assertEqual(result['status'], 'ambiguous')
        self.assertIsNone(result['slug'])
        self.assertEqual(result['candidates'][0]['slug'], 'a-2021')
        self.assertEqual(pipe.predict(self.image, bbox=(0,0,100,80), manual_text='')['status'], 'unreadable')
        self.assertEqual(pipe.predict(self.image, bbox=(0,0,100,80), seconds=0)['reason_codes'], ['deadline'])

    def test_policy_and_unknown(self):
        pipe = Pipeline(self.index, MockOCR('x'), Policy(90, 5, .1, 'validation-v1'))
        result = pipe.predict(self.image, bbox=(0,0,100,80), manual_text='Вино 2021\nЗавод А')
        self.assertEqual(result['status'], 'accepted')
        self.assertEqual(result['slug'], 'a-2021')
        self.assertNotEqual(pipe.predict(self.image, bbox=(0,0,100,80), manual_text='coca cola')['status'], 'accepted')

    def test_ocr_capacity_is_bounded(self):
        entered, release = threading.Event(), threading.Event()

        class SlowOCR:
            def recognize(self, roi, deadline):
                entered.set()
                release.wait(2)
                return MockOCR('Вино 2021').recognize(roi, deadline)

        pipe = Pipeline(self.index, SlowOCR(), max_concurrent_ocr=1)
        result = []
        worker = threading.Thread(target=lambda: result.append(
            pipe.predict(self.image, bbox=(0, 0, 100, 80), seconds=3)))
        worker.start()
        try:
            self.assertTrue(entered.wait(2))
            self.assertEqual(pipe.predict(self.image, bbox=(0, 0, 100, 80))['reason_codes'],
                             ['ocr_capacity_exceeded'])
        finally:
            release.set()
            worker.join(3)
        self.assertEqual(result[0]['status'], 'ambiguous')
        self.assertNotEqual(pipe.predict(self.image, bbox=(0, 0, 100, 80))['reason_codes'],
                            ['ocr_capacity_exceeded'])

    def test_multipart(self):
        body = b'--abc\r\nContent-Disposition: form-data; name="image"; filename="a.jpg"\r\nContent-Type: image/jpeg\r\n\r\n' + self.image + b'\r\n--abc--\r\n'
        self.assertEqual(multipart_image('multipart/form-data; boundary=abc', body), self.image)
        with self.assertRaises(ValueError):
            multipart_image('text/plain', body)

    def test_ocr_errors_no_secret(self):
        crop = manual_roi(self.image, (0,0,100,80), time.monotonic()+5)
        class Response:
            status_code = 429
        with patch('httpx.Client.post', return_value=Response()):
            output = MistralOCR('secret').recognize(crop, time.monotonic()+5)
            self.assertEqual(output.error_code, 'rate_limited')
            self.assertNotIn('secret', repr(output))
        with patch('httpx.Client.post', side_effect=__import__('httpx').TimeoutException('oops')):
            self.assertEqual(MistralOCR('secret').recognize(crop, time.monotonic()+5).status, 'timeout')
        class OK:
            status_code = 200
            def json(self): return {'pages':[{'markdown':'Étiquette 2021'}]}
        with patch('httpx.Client.post', return_value=OK()):
            result = MistralOCR('secret').recognize(crop, time.monotonic()+5)
            self.assertEqual(result.normalized_text, 'étiquette 2021')
            self.assertIsNone(result.confidence)

    def test_http_contract(self):
        pipe = Pipeline(self.index, MockOCR('Вино 2021'), csv_sha256='test')
        server = ThreadingHTTPServer(('127.0.0.1', 0), create_handler(pipe))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f'http://127.0.0.1:{server.server_address[1]}'
            with httpx.Client() as client:
                self.assertTrue(client.get(url+'/ready').json()['ready'])
                result = client.post(url+'/v1/recognize', files={'image': ('wrong.jpg', self.image)})
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json()['status'], 'unreadable')
                self.assertIsNone(client.post(url+'/v1/eval/predict', files={'image': ('wrong.jpg', self.image)}).json()['slug'])
                self.assertIsNone(client.post(url+'/v1/eval/predict', files={'image': ('bad.jpg', b'bad')}).json()['slug'])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_manifest_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'photo.webp').write_bytes(self.image)
            digest = hashlib.sha256(self.image).hexdigest()
            base = {'image_path':'photo.webp', 'image_sha256':digest, 'bbox':[0,0,100,80]}
            gallery = {**base, 'asset_id':'a', 'wine_id':'a-2020', 'role':'front',
                       'mapping_status':'verified', 'verified_by':'human',
                       'verification_method':'label checked', 'source_group':'shoot1'}
            query = {**base, 'query_id':'q', 'gt_status':'known', 'gt_slug':'a-2020',
                     'gt_source':'other record', 'split':'test', 'source_group':'shoot1'}
            for name, row in [('gallery', gallery), ('queries', query)]:
                (root/(name+'.jsonl')).write_text(json.dumps(row)+'\n')
            result = audit(self.index.wines, root/'gallery.jsonl', root/'queries.jsonl')
            self.assertTrue(any('identical image' in err for err in result['errors']))
            self.assertTrue(any('shared source_group' in err for err in result['errors']))

    def test_manifest_query_hash_leak_and_gt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'photo.webp').write_bytes(self.image)
            digest = hashlib.sha256(self.image).hexdigest()
            (root/'gallery.jsonl').write_text('')
            first = dict(query_id='q1', image_path='photo.webp', image_sha256=digest,
                         bbox=[0, 0, 100, 80], source_group='shoot1', split='validation',
                         gt_status='known', gt_slug='a-2020', gt_source='human')
            second = dict(first, query_id='q2', source_group='shoot2', split='test')
            manifest = root/'queries.jsonl'
            manifest.write_text(json.dumps(first)+'\n'+json.dumps(second)+'\n')
            result = audit(self.index.wines, root/'gallery.jsonl', manifest)
            self.assertTrue(any('image hash across splits' in e for e in result['errors']))
            second['query_id'] = 'q1'
            second['gt_status'] = 'unknown'
            second['gt_slug'] = 'a-2020'
            manifest.write_text(json.dumps(first)+'\n'+json.dumps(second)+'\n')
            errors = audit(self.index.wines, root/'gallery.jsonl', manifest)['errors']
            self.assertTrue(any('duplicate id' in e for e in errors))
            self.assertTrue(any('GT must be null' in e for e in errors))

    def test_gallery_roles_missing_and_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'photo.webp').write_bytes(self.image)
            row = {'asset_id':'a','wine_id':'a-2020','role':'front','source_group':'shoot1',
                   'mapping_status':'verified','verified_by':'human','verification_method':'checked',
                   'image_path':'photo.webp','image_sha256':hashlib.sha256(self.image).hexdigest(),
                   'bbox':[0,0,100,80]}
            manifest = root/'gallery.jsonl'
            manifest.write_text(json.dumps(row)+'\n')
            def fake_embed(pngs, **kwargs):
                return np.array([[1., 0.]], dtype='float32')
            build_gallery(manifest, self.index.wines, root/'gallery.npz', embed=fake_embed)
            g = Gallery(root/'gallery.npz', model=MODEL,
                        pretrained=PRETRAINED, csv_sha256='test')
            scores = g.scores([1., 0.], ['a-2020', 'a-2021'])
            self.assertEqual(scores['a-2020']['cosine'], 1.)
            self.assertIsNone(scores['a-2021']['cosine'])
            self.assertIsNone(g.scores([1., 0.], ['a-2020'], role='back')['a-2020']['cosine'])
            with self.assertRaises(IncompatibleGallery):
                Gallery(root/'gallery.npz', model='other', pretrained='laion2b_s34b_b79k', csv_sha256='test')
            row['mapping_status'] = 'unresolved'
            manifest.write_text(json.dumps(row)+'\n')
            with self.assertRaises(IncompatibleGallery):
                build_gallery(manifest, self.index.wines, root/'another.npz', embed=fake_embed)

    def test_media_inventory_unverified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'bottle_012345abcd.jpg').write_bytes(self.image)  # WEBP content
            (root/'thumbnail_bottle_012345abcd.webp').write_bytes(self.image)
            output = inventory(root)
            self.assertEqual(output['file_count'], 2)
            self.assertEqual(output['mime_counts']['image/webp'], 2)
            self.assertEqual(output['extension_mismatches'], ['bottle_012345abcd.jpg'])
            self.assertEqual(output['groups_with_multiple_files'], 1)
            self.assertIn('unverified', output['note'])

    def test_paired_roi_comparison(self):
        row = {'query_id':'q', 'image_sha256':'sha', 'split':'validation',
               'gt_slug':'a-2021', 'roi':{'bbox':[0,0,10,10]},
               'versions':{'csv_sha256':'csv'}, 'candidates':[{'slug':'a-2021'}],
               'ocr':{'model':'mock'}}
        other = {**row, 'roi':None, 'candidates':[], 'ocr':None}
        result = compare([row], [other])
        self.assertEqual(result['manual_found_experimental_missed'], ['q'])
        self.assertTrue(result['ocr_oracle_included'])
        with self.assertRaisesRegex(ValueError, 'different image'):
            compare([row], [{**other, 'image_sha256':'changed'}])
        with self.assertRaisesRegex(ValueError, 'different GT'):
            compare([row], [{**other, 'gt_status': 'unknown'}])
        with self.assertRaisesRegex(ValueError, 'different search_backend'):
            compare([row], [{**other, 'versions': {'csv_sha256': 'csv', 'search_backend': 'other'}}])

    def test_report_denominators(self):
        rows = [{'gt_slug':'a', 'status':'ambiguous', 'slug':None,
                 'candidates':[{'slug':'a'}], 'timings_ms':{'total':2}},
                {'gt_slug':None, 'gt_status':'unknown', 'status':'unknown', 'slug':None}]
        output = report(rows)
        self.assertEqual(output['exact_slug_accuracy']['value'], 0)
        self.assertEqual(output['text_recall']['1']['value'], 1)
        self.assertEqual(output['unknown_false_accept']['denominator'], 1)

    def test_report_accepted_precision_excludes_unknown_null(self):
        rows = [{'gt_slug': None, 'gt_status': 'unknown', 'status': 'accepted', 'slug': None},
                {'gt_slug': 'wine', 'gt_status': 'known', 'status': 'ambiguous', 'slug': 'wine'},
                {'gt_slug': 'wine', 'gt_status': 'known', 'status': 'accepted', 'slug': 'wine'}]
        output = report(rows)
        self.assertEqual(output['accepted_precision']['numerator'], 1)
        self.assertEqual(output['accepted_precision']['denominator'], 2)
        self.assertEqual(output['exact_slug_accuracy']['numerator'], 1)
        self.assertEqual(output['unknown_false_accept']['numerator'], 1)


if __name__ == '__main__':
    unittest.main()
