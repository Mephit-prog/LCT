import csv
import hashlib
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import numpy as np

from wineid.catalog_export import build_tz_datasets, image_filters, producer_split
from wineid.train_clip import load_datasets, contrastive_train_items, info_nce_loss


def photo(path, size=(640, 640)):
    """Noisy RGB image so the compressed file exceeds the 15 KB filter."""
    buffer = io.BytesIO()
    Image.frombytes('RGB', size, os.urandom(size[0] * size[1] * 3)).save(buffer, 'WEBP')
    path.write_bytes(buffer.getvalue())
    return hashlib.sha256(buffer.getvalue()).hexdigest()


class CatalogExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.uploads = self.root / 'uploads'
        self.uploads.mkdir()
        self.csv = self.root / 'catalog.csv'
        self.items = []

    def wine(self, slug, name, producer, category, grape):
        self.items.append({'Slug': slug, 'Название вина': name, 'Винодельня': producer,
                           'Категория': category, 'Сорт винограда': grape,
                           'Название фото': slug + '.webp'})

    def save(self):
        with self.csv.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Slug', 'Название вина', 'Винодельня',
                                                   'Категория', 'Сорт винограда', 'Название фото'])
            writer.writeheader()
            writer.writerows(self.items)

    def test_producer_split_is_stable_and_bucketed(self):
        a = producer_split('Фанагория')
        self.assertEqual(a, producer_split('Фанагория'))
        self.assertIn(producer_split('Фанагория'), ('train', 'validation', 'test'))
        values = {producer_split(f'Хозяйство {i}') for i in range(200)}
        self.assertLessEqual(len(values), 3)

    def test_image_filters(self):
        small = self.uploads / 'small_aaaaaaaaaa.webp'
        small.write_bytes(b'x' * 1024)
        ok, reason = image_filters(small, 'digest')
        self.assertFalse(ok)
        self.assertEqual(reason, 'too_small')
        big = self.uploads / 'big_bbbbbbbbbb.webp'
        photo(big)
        self.assertTrue(image_filters(big, 'digest')[0])

    def test_tz_datasets(self):
        photo(self.uploads / 'a_aaaaaaaaaa.webp')
        photo(self.uploads / 'b_bbbbbbbbbb.webp')
        self.wine('a', 'Алиготе Баррель 2024', 'Коммуналка', 'Белое', 'Алиготе')
        self.wine('b', 'Рислинг', 'Завод Б', 'Красное', 'Рислинг')
        self.save()
        wines_rows, images_rows, summary = build_tz_datasets(self.csv, self.uploads)
        self.assertEqual(summary['catalog_slugs'], 2)
        by_id = {r['wine_id']: r for r in wines_rows}
        self.assertEqual(by_id['a']['vintage'], '2024')
        self.assertEqual(by_id['a']['color'], 'Белое')
        self.assertEqual(by_id['a']['producer'], 'коммуналка')
        self.assertEqual(by_id['a']['name_ru'], 'алиготе баррель 2024')
        self.assertIn(by_id['a']['producer_split'], ('train', 'validation', 'test'))
        self.assertEqual(summary['images_rows'], 2)
        self.assertEqual(images_rows[0]['view'], 'studio_front')
        self.assertEqual(images_rows[0]['usable_train'], 0)
        self.assertEqual(images_rows[0]['source'], 'strapi')

    def test_shared_asset_rejected(self):
        photo(self.uploads / 'shared_aaaaaaaaaa.webp')
        self.wine('one', 'Одно', 'Х', 'Белое', '')
        self.wine('two', 'Другое', 'Х', 'Белое', '')
        # both catalog rows claim the same photo file
        self.items[0]['Название фото'] = 'shared.webp'
        self.items[1]['Название фото'] = 'shared.webp'
        self.save()
        _, images_rows, summary = build_tz_datasets(self.csv, self.uploads)
        self.assertEqual(images_rows, [])
        reasons = {k: v for k, v in summary['rejected'].items() if v}
        self.assertEqual(sum(reasons.values()), 2)
        # catalog_media flags it already; no image may become a verified reference.
        self.assertGreaterEqual(
            summary['rejected'].get('unresolved_link', 0) +
            summary['rejected'].get('sha256_shared_between_slugs', 0), 2)


class TrainDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.wines = self.root / 'wines.csv'
        self.images = self.root / 'images.csv'
        with self.wines.open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['wine_id', 'name_ru', 'producer', 'color',
                                              'vintage', 'ean', 'producer_split'])
            w.writeheader()
            for i in range(5):
                w.writerow({'wine_id': f'w{i}', 'name_ru': f'Вино {i}', 'producer': 'Хоз',
                            'color': 'Белое', 'vintage': '', 'ean': '',
                            'producer_split': 'train'})

    def test_contrastive_filtering(self):
        with self.images.open('w', encoding='utf-8', newline='') as f:
            im = csv.DictWriter(f, fieldnames=['image_id', 'wine_id', 'path', 'view',
                                               'sha256', 'source', 'source_url',
                                               'ocr_hit', 'usable_train'])
            im.writeheader()
            # w0: single frame -> excluded
            im.writerow({'image_id': 'w0-1', 'wine_id': 'w0', 'path': 'w0.webp',
                         'view': 'studio_front', 'sha256': '0' * 64, 'source': 'own',
                         'source_url': '', 'ocr_hit': 'none', 'usable_train': 1})
            # w1: two studio frames -> no non-studio -> excluded
            im.writerow({'image_id': 'w1-1', 'wine_id': 'w1', 'path': 'w1a.webp',
                         'view': 'studio_front', 'sha256': '1' * 64, 'source': 'own',
                         'source_url': '', 'ocr_hit': 'none', 'usable_train': 1})
            im.writerow({'image_id': 'w1-2', 'wine_id': 'w1', 'path': 'w1b.webp',
                         'view': 'studio_front', 'sha256': '2' * 64, 'source': 'own',
                         'source_url': '', 'ocr_hit': 'both', 'usable_train': 1})
            # w2: studio + field -> included
            im.writerow({'image_id': 'w2-1', 'wine_id': 'w2', 'path': 'w2a.webp',
                         'view': 'studio_front', 'sha256': '3' * 64, 'source': 'own',
                         'source_url': '', 'ocr_hit': 'none', 'usable_train': 1})
            im.writerow({'image_id': 'w2-2', 'wine_id': 'w2', 'path': 'w2b.webp',
                         'view': 'field_front', 'sha256': '4' * 64, 'source': 'own',
                         'source_url': '', 'ocr_hit': 'both', 'usable_train': 1})
            # w3: usable_train=0 -> excluded
            im.writerow({'image_id': 'w3-1', 'wine_id': 'w3', 'path': 'w3a.webp',
                         'view': 'studio_front', 'sha256': '5' * 64, 'source': 'own',
                         'source_url': '', 'ocr_hit': 'none', 'usable_train': 0})
        splits = load_datasets(self.wines, self.images)
        train = contrastive_train_items(splits['train'])
        self.assertEqual({i.wine_id for i in train}, {'w2'})
        self.assertEqual(len(train), 2)

    def test_load_datasets_rejects_unknown_wine(self):
        with self.images.open('w', encoding='utf-8', newline='') as f:
            f.write('image_id,wine_id,path,view,sha256,source,source_url,ocr_hit,usable_train\n'
                    'x,nope,x.webp,studio_front,0000000000000000000000000000000000000000000000000000000000000000,own,,none,1\n')
        with self.assertRaises(ValueError):
            load_datasets(self.wines, self.images)


class LossAndAdapterTests(unittest.TestCase):
    def test_info_nce_same_producer_weighting(self):
        # P1-1: same-producer negatives are weighted 2x; the loss must exceed the
        # unweighted baseline on the same logits.
        logits = np.array([[3.0, 1.0], [1.0, 3.0]], dtype='float32')
        weighted, _ = info_nce_loss(logits, logits, 1.0, ['A', 'A'])
        baseline, _ = info_nce_loss(logits, logits, 1.0, ['A', 'B'])
        self.assertGreater(weighted, baseline)
        unweighted, _ = info_nce_loss(logits, logits, 1.0, ['A', ''])
        self.assertAlmostEqual(unweighted, baseline)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch not installed')
    def test_attach_and_load_adapter_round_trip(self):
        import io as _io
        import torch
        from torch import nn

        from wineid.train_clip import attach_lora, load_adapter

        class SelfAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(4, 4)
                self.k_proj = nn.Linear(4, 4)
                self.v_proj = nn.Linear(4, 4)
                self.out_proj = nn.Linear(4, 4)

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = SelfAttn()

        class Encoder(nn.Module):
            """Mimics HF CLIP: ``encoder.layers`` is a module list attribute."""

            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Layer() for _ in range(6)])

        class Vision(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder()
                self.head = nn.Linear(4, 4)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_model = Vision()

        model = Model()
        patched = attach_lora(model, lora_blocks=4, rank=2, alpha=8, dropout=0.0)
        self.assertEqual(len(patched), 16)  # 4 layers x 4 projections
        state = {k: v.detach().clone() for k, v in model.state_dict().items()
                 if 'lora' in k or 'vision_model.head' in k}
        state['temperature'] = torch.tensor(0.07)
        buffer = _io.BytesIO()
        torch.save({'meta': {'lora_blocks': 4, 'rank': 2, 'alpha': 8,
                             'dropout': 0.0, 'revision': 'a' * 40},
                    'state': state}, buffer)
        buffer.seek(0)

        fresh = Model()
        meta = load_adapter(fresh, buffer)
        self.assertEqual(meta['rank'], 2)
        self.assertAlmostEqual(float(meta['temperature']), 0.07)
        # LoRA layers re-attached and weights applied.
        probe = fresh.vision_model.encoder.layers[-1].self_attn.q_proj
        self.assertTrue(hasattr(probe, 'lora_a'))
        original_q = model.vision_model.encoder.layers[-1].self_attn.q_proj.lora_a
        self.assertTrue(torch.equal(probe.lora_a, original_q))
        # Unknown keys in the state are rejected.
        state['bogus_param'] = torch.zeros(1)
        buffer2 = _io.BytesIO()
        torch.save({'meta': {'lora_blocks': 4, 'rank': 2, 'alpha': 8,
                             'dropout': 0.0, 'revision': 'a' * 40},
                    'state': state}, buffer2)
        buffer2.seek(0)
        with self.assertRaises(ValueError):
            load_adapter(Model(), buffer2)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'torch not installed')
    def test_attach_lora_eva_blocks_direct_weight_access(self):
        # jina-clip-v2 (EVA02): blocks[i].attn.{q,k,v}_proj used via
        # F.linear(x, self.q_proj.weight, bias) and output attn.proj.
        import torch
        from torch import nn
        from torch.nn import functional as F

        from wineid.train_clip import (_trainable, attach_lora,
                                       inspect_vision_modules)

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(4, 4, bias=False)
                self.k_proj = nn.Linear(4, 4, bias=False)
                self.v_proj = nn.Linear(4, 4, bias=False)
                self.q_bias = nn.Parameter(torch.zeros(4))
                self.proj = nn.Linear(4, 4)

            def forward(self, x):
                q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
                k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
                v = F.linear(input=x, weight=self.v_proj.weight, bias=None)
                return self.proj(q * k + v)

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attention()

            def forward(self, x):
                return x + self.attn(x)

        class Vision(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([Block() for _ in range(6)])
                self.head = nn.Identity()

            def forward(self, x):
                for block in self.blocks:
                    x = block(x)
                return self.head(x)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_model = Vision()
                self.visual_projection = nn.Linear(4, 3, bias=False)

        torch.manual_seed(0)
        model = Model()
        report = inspect_vision_modules(model)
        self.assertEqual(report['blocks_path'], 'vision_model.blocks')
        self.assertEqual(report['blocks'][5]['lora_targets'],
                         ['q_proj', 'k_proj', 'v_proj', 'proj'])
        self.assertEqual(report['vision_projection'], [3, 4])
        model.requires_grad_(False)
        x = torch.randn(2, 4)
        before = model.visual_projection(model.vision_model(x)).detach()
        patched = attach_lora(model, lora_blocks=4, rank=2, alpha=4, dropout=0.0)
        self.assertEqual(len(patched), 16)
        self.assertIn('attn.proj', patched)
        # lora_b starts at zero: output is unchanged right after attaching.
        after = model.visual_projection(model.vision_model(x))
        self.assertTrue(torch.allclose(before, after))
        after.pow(2).sum().backward()
        lora_b = [p for n, p in model.named_parameters() if n.endswith('lora_b')]
        self.assertEqual(len(lora_b), 16)
        # Gradient reaches LoRA also through the direct .weight access path.
        q = model.vision_model.blocks[-1].attn.q_proj
        self.assertGreater(float(q.lora_b.grad.abs().sum()), 0.0)
        self.assertTrue(all(p.grad is not None for p in lora_b))
        self.assertIsNone(q.base.weight.grad)
        self.assertTrue(_trainable('visual_projection.weight'))


if __name__ == '__main__':
    unittest.main()