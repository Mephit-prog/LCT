import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from wineid.catalog_media import catalog_media_report


class CatalogMediaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.uploads = self.root/'uploads'
        self.uploads.mkdir()
        self.csv = self.root/'catalog.csv'
        self.items = []

    def photo(self, filename):
        path = self.uploads/filename
        Image.new('RGB', (8, 8), 'red').save(path)
        return path

    def wine(self, slug, photo_name):
        self.items.append({'Slug': slug, 'Название вина': slug, 'Винодельня': 'maker',
                           'Название фото': photo_name})

    def save(self):
        with self.csv.open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['Slug', 'Название вина', 'Винодельня', 'Название фото'])
            writer.writeheader()
            writer.writerows(self.items)

    def test_csv_gt_kept_distinct_from_filename_candidates(self):
        original = self.photo('red_wine_aaaaaaaaaa.webp')
        self.photo('thumbnail_red_wine_aaaaaaaaaa.webp')
        self.wine('red-wine', 'red-wine.webp')
        self.photo('photo_only_bbbbbbbbbb.webp')
        self.wine('catalog-only-slug', 'photo_only.webp')
        self.photo('slug_only_cccccccccc.webp')
        self.wine('slug-only', 'unmatched_name.webp')
        self.photo('two_dddddddddd.webp')
        self.photo('alternate_eeeeeeeeee.webp')
        self.wine('two', 'alternate.webp')
        self.photo('shared_ffffffffff.webp')
        self.wine('shared-one', 'shared.webp')
        self.wine('shared-two', 'shared.webp')
        self.photo('duplicate_1111111111.webp')
        self.photo('duplicate_2222222222.webp')
        self.wine('duplicate', 'duplicate.webp')
        self.photo('thumbnail_thumbnail_only_3333333333.webp')
        self.wine('thumbnail-only', 'thumbnail-only.webp')
        self.wine('unmatched', 'missing.webp')
        self.save()
        rows, summary = catalog_media_report(self.csv, self.uploads)
        by_slug = {row['slug']: row for row in rows}
        self.assertEqual(summary['catalog_slugs'], len(self.items))
        self.assertEqual(summary['candidate_links'], 3)
        self.assertEqual(by_slug['red-wine']['link_method'], 'photo_name_and_slug')
        self.assertEqual(by_slug['red-wine']['mapping_status'], 'candidate')
        self.assertEqual(by_slug['red-wine']['image_path'], original.name)
        self.assertEqual(by_slug['red-wine']['image_sha256'], hashlib.sha256(original.read_bytes()).hexdigest())
        self.assertEqual(by_slug['red-wine']['candidate_families'][0]['derivatives'],
                         ['thumbnail_red_wine_aaaaaaaaaa.webp'])
        self.assertEqual(by_slug['catalog-only-slug']['link_method'], 'csv_photo_name')
        self.assertEqual(by_slug['slug-only']['link_method'], 'slug_filename_only')
        self.assertEqual(by_slug['two']['issue'], 'photo_slug_conflict')
        self.assertEqual(by_slug['shared-one']['issue'], 'shared_csv_photo_name')
        self.assertEqual(by_slug['shared-two']['issue'], 'shared_csv_photo_name')
        self.assertEqual(by_slug['duplicate']['issue'], 'multiple_asset_families')
        self.assertEqual(by_slug['thumbnail-only']['issue'], 'missing_or_multiple_originals')
        self.assertEqual(by_slug['unmatched']['issue'], 'unmatched')
        self.assertTrue(all(row['csv_sha256'] == summary['csv_sha256'] for row in rows))
        self.assertTrue(all(row['mapping_status'] != 'verified' for row in rows))

    def test_media_reused_for_different_slugs_is_not_exact_gt(self):
        self.photo('same_aaaaaaaaaa.webp')
        self.wine('one', 'same.webp')
        self.wine('same', 'other.webp')
        self.save()
        rows, _ = catalog_media_report(self.csv, self.uploads)
        self.assertEqual([row['issue'] for row in rows],
                         ['asset_candidate_shared_between_slugs'] * 2)
        self.assertTrue(all(row['image_path'] is None for row in rows))

    def test_cli_snapshot_and_conflicting_csv_duplicates(self):
        self.photo('a_aaaaaaaaaa.webp')
        self.wine('a', 'a.webp')
        self.wine('a', 'a.webp')
        self.save()
        output = self.root/'links.jsonl'
        command = [sys.executable, '-m', 'wineid.catalog_media', str(self.csv),
                   str(self.uploads), str(output)]
        run = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)['catalog_slugs'], 1)
        self.assertEqual(json.loads(output.read_text())['csv_row_ids'], [2, 3])
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.items[1]['Название фото'] = 'different.webp'
        self.save()
        with self.assertRaisesRegex(ValueError, 'conflicting duplicate slugs'):
            catalog_media_report(self.csv, self.uploads)


if __name__ == '__main__':
    unittest.main()
