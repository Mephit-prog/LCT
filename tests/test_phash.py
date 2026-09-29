"""pHash near-duplicate detection (tz-wine-label-retrieval.md §4.1, ТЗ §8)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from wineid.phash import (NEAR_DUPLICATE_DISTANCE, distance_band_pairs, hamming,
                          near_duplicate_indices, phash)
from wineid.catalog_export import build_tz_datasets


def noise_image(size=400, seed=0):
    array = np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype='uint8')
    return Image.fromarray(array)


def save(image, path, fmt='PNG'):
    image.save(path, format=fmt)


class PhashTests(unittest.TestCase):
    def test_identical_images_same_hash(self):
        image = noise_image()
        self.assertEqual(phash(image), phash(image.copy()))
        self.assertEqual(hamming(phash(image), phash(image.copy())), 0)

    def test_resize_is_stable(self):
        image = noise_image()
        small = image.resize((200, 200))
        self.assertLess(hamming(phash(image), phash(small)), NEAR_DUPLICATE_DISTANCE)

    def test_hash_is_64_bit(self):
        self.assertTrue(0 <= phash(noise_image()) < 2 ** 64)

    def test_hamming_basic(self):
        self.assertEqual(hamming(0b1010, 0b0011), 2)

    def test_near_duplicate_indices_cross_owner_only(self):
        # 0/1 are different owners and near-identical -> both dropped.
        # 2/3 same owner are near-identical -> allowed (multi-frame SKU).
        hashes = [0b0000, 0b0001, 0b1111_0000, 0b1111_0001]
        owners = ['a', 'b', 'c', 'c']
        self.assertEqual(near_duplicate_indices(hashes, owners, max_distance=4), {0, 1})
        self.assertEqual(near_duplicate_indices(hashes, owners, max_distance=1), set())

    def test_none_hash_never_matches(self):
        self.assertEqual(near_duplicate_indices([None, 5], ['a', 'b']), set())

    def test_distance_band_pairs(self):
        pairs = distance_band_pairs([0, 0b11, 0b1111], ['a', 'b', 'c'], low=1, high=3)
        self.assertIn((0, 1, 2), pairs)
        self.assertNotIn((0, 2, 4), pairs)


class CatalogExportPhashTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.uploads = self.root / 'uploads'
        self.uploads.mkdir()
        self.csv = self.root / 'catalog.csv'

    def tearDown(self):
        self.temp.cleanup()

    def _catalog(self, rows):
        header = 'Название вина,Винодельня,Категория,Сорт винограда,Slug,Название фото\n'
        body = '\n'.join(rows) + '\n'
        self.csv.write_text(header + body, encoding='utf-8')

    def test_near_duplicate_pair_is_dropped(self):
        base = noise_image(seed=1)
        # Different bytes (lossless re-encode of a brightness-shifted frame) but
        # near-identical perception; sha256 differs so the sha filter misses it.
        shifted = base.point(lambda v: min(255, v + 1))
        save(base, self.uploads / 'wine-a_abcdef1234.png')
        save(shifted, self.uploads / 'wine-b_abcdef5678.png')
        self.assertNotEqual(phash(base), None)
        self.assertLess(hamming(phash(base), phash(shifted)), NEAR_DUPLICATE_DISTANCE)
        self._catalog(['Вино А,Завод А,Красное,Каберне,wine-a,wine-a.png',
                       'Вино Б,Завод Б,Красное,Мерло,wine-b,wine-b.png'])
        _, images_rows, summary = build_tz_datasets(self.csv, self.uploads)
        self.assertEqual(images_rows, [])
        self.assertEqual(summary['rejected'].get('phash_near_duplicate'), 2)

    def test_distinct_images_both_kept(self):
        save(noise_image(seed=1), self.uploads / 'wine-a_abcdef1234.png')
        save(noise_image(seed=2), self.uploads / 'wine-b_abcdef5678.png')
        self._catalog(['Вино А,Завод А,Красное,Каберне,wine-a,wine-a.png',
                       'Вино Б,Завод Б,Красное,Мерло,wine-b,wine-b.png'])
        _, images_rows, summary = build_tz_datasets(self.csv, self.uploads)
        self.assertEqual(len(images_rows), 2)
        self.assertNotIn('phash_near_duplicate', summary['rejected'])

    def test_note_mentions_phash_and_unimplemented_filters(self):
        save(noise_image(seed=1), self.uploads / 'wine-a_abcdef1234.png')
        self._catalog(['Вино А,Завод А,Красное,Каберне,wine-a,wine-a.png'])
        _, _, summary = build_tz_datasets(self.csv, self.uploads)
        self.assertIn('pHash', summary['note'])
        self.assertIn('watermark', summary['note'])


if __name__ == '__main__':
    unittest.main()
