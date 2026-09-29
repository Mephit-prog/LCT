import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from wineid.clip_zero_shot import (EmbeddingIndex, build_images, build_classes,
                                  class_prompts, read_image, retrieval_report, main)
from wineid.jina_clip import (JinaCLIPEmbedder, IncompatibleEmbeddings, MODEL, REVISION,
                             CODE_REVISION, PREPROCESS, unit_vectors)
from wineid.visual import CLIPEmbedder, Gallery


class FakeEncoder:
    config = {'model': 'fake', 'revision': REVISION, 'dimensions': 2}

    def __init__(self):
        self.image_batches, self.text_batches = [], []

    def encode_images(self, images):
        self.image_batches.append(len(images))
        return np.array([[2, 0] if im.getpixel((0, 0))[0] > 100 else [0, 3] for im in images])

    def encode_text(self, texts):
        self.text_batches.append(list(texts))
        return np.array([[2, 0] if 'Анджуйское' in text else [0, 3] for text in texts])


class ZeroShotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        Image.new('RGB', (10, 20), 'red').save(self.root/'a.png')
        Image.new('RGB', (10, 20), 'blue').save(self.root/'b.png')
        self.encoder = FakeEncoder()
        self.classes = [{'class_id': 'a', 'name': 'Анджуйское', 'producer': 'Завод А'},
                        {'class_id': 'b', 'name': 'Другое', 'producer': 'Завод Б'}]

    def test_image_index_search_roundtrip_and_batching(self):
        index = build_images(self.root, self.encoder, batch_size=1)
        self.assertEqual(self.encoder.image_batches, [1, 1])
        self.assertEqual(index.search('Этикетка вина «Анджуйское»', self.encoder)[0]['id'], 'a.png')
        self.assertEqual(index.search('Другое', self.encoder, top_k=100)[0]['id'], 'b.png')
        np.testing.assert_allclose(np.linalg.norm(index.vectors, axis=1), [1, 1])
        index.save(self.root/'images.npz')
        loaded = EmbeddingIndex.load(self.root/'images.npz')
        self.assertEqual(index.meta, loaded.meta)
        self.assertEqual(loaded.search('Анджуйское', self.encoder, 1)[0]['cosine'], 1)
        with self.assertRaises(FileExistsError):
            index.save(self.root/'images.npz')
        self.assertEqual(index.rank([[1, 1]], top_k=2)[0][0]['id'], 'a.png')  # stable tie
        self.assertNotIn('a.png', str(self.encoder.text_batches))  # filenames aren't text features
        with self.assertRaises(ValueError):
            index.classify([], self.encoder)
        self.assertTrue(loaded.audit_images()['snapshot_valid'])
        Image.new('RGB', (10, 20), 'green').save(self.root/'a.png')
        self.assertEqual(loaded.audit_images()['problems'][0]['reason'], 'changed_or_oversized')

    def test_zero_shot_prompts_and_prediction_no_reference_photos(self):
        index = build_classes(self.classes, self.encoder, variant='name-producer', ensemble=True)
        self.assertEqual(len(self.encoder.text_batches[0]), 4)
        self.assertEqual(self.encoder.image_batches, [])
        for item in index.items:
            self.assertEqual(len(item['prompts']), 2)
            self.assertIn('производителя', item['prompts'][0])
            self.assertNotIn('a photo', item['prompts'][0])
        with Image.open(self.root/'b.png') as image:
            ranking = index.classify([image], self.encoder)[0]
        self.assertEqual(ranking[0]['id'], 'b')
        self.assertEqual(ranking[0]['cosine'], 1)
        with self.assertRaises(ValueError):
            index.search('Анджуйское', self.encoder)
        missing = [{**self.classes[0], 'producer': ''}, self.classes[1]]
        with self.assertRaisesRegex(ValueError, 'every class'):
            class_prompts(missing, 'name-producer')
        self.assertEqual(len(class_prompts(missing, 'name')), 2)
        with self.assertRaises(ValueError):
            class_prompts(self.classes, 'name-producer-visual')
        visual = [{**c, 'visual': 'белая этикетка с крупной эмблемой'} for c in self.classes]
        self.assertIn('крупной эмблемой', class_prompts(visual, 'name-producer-visual')[0]['prompts'][0])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            class_prompts([self.classes[0], self.classes[0]])
        with self.assertRaises(ValueError):
            class_prompts([])

    def test_identical_prompts_are_not_reported_as_unique_class(self):
        from wineid.clip_zero_shot import prompt_collisions
        rows = [dict(class_id='a', name='Одинаковое'),
                dict(class_id='b', name='Одинаковое')]
        index = build_classes(rows, self.encoder)
        self.assertEqual(prompt_collisions(index.items)['a'], ['a', 'b'])
        self.assertEqual(index.meta['prompt_collisions']['b'], ['a', 'b'])
        index.save(self.root/'ties.npz')
        self.assertEqual(EmbeddingIndex.load(self.root/'ties.npz').meta['prompt_collisions']['a'], ['a', 'b'])
        with patch('wineid.clip_zero_shot.JinaCLIPEmbedder', return_value=self.encoder), \
                patch.object(sys, 'argv', ['clip_zero_shot', 'predict', str(self.root/'ties.npz'),
                                          str(self.root/'a.png'), '--allow-remote-code']), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            main()
        result = json.loads(output.getvalue())['predictions'][0]
        self.assertIsNone(result['predicted_class_id'])
        self.assertEqual(result['status'], 'ambiguous_prompt')
        self.assertEqual(result['indistinguishable_class_ids'], ['a', 'b'])
        with self.assertRaises(ValueError):
            retrieval_report(index, [], self.encoder)

    def test_invalid_image_metadata_and_isolation(self):
        index = build_images(self.root, self.encoder)
        meta = copy.deepcopy(index.meta)
        meta['items'][0]['image_sha256'] = 'broken'
        with self.assertRaises(IncompatibleEmbeddings):
            EmbeddingIndex(index.vectors, meta)
        meta = copy.deepcopy(index.meta)
        valid = EmbeddingIndex(index.vectors, meta)
        meta['items'][0]['id'] = 'changed'
        self.assertEqual(valid.items[0]['id'], 'a.png')

    def test_normalized_prompt_ensemble(self):
        encoder = FakeEncoder()
        encoder.encode_text = lambda texts: np.array([[10., 0.], [0., 2.]])
        index = build_classes(self.classes[:1], encoder, ensemble=True)
        np.testing.assert_allclose(index.vectors, [[2**-.5, 2**-.5]], rtol=1e-6)
        encoder.encode_text = lambda texts: np.array([[1., 0.], [-1., 0.]])
        with self.assertRaisesRegex(IncompatibleEmbeddings, 'zero'):
            build_classes(self.classes[:1], encoder, ensemble=True)

    def test_index_incompatibility_and_invalid_vectors(self):
        index = build_images(self.root, self.encoder)
        other = FakeEncoder()
        other.config = {**other.config, 'revision': 'different'}
        with self.assertRaisesRegex(IncompatibleEmbeddings, 'mismatch'):
            index.search('Другое', other)
        for vectors in ([[0, 0]], [[float('nan'), 0]], [[float('inf'), 0]], [[1, 0, 1]]):
            with self.assertRaises(IncompatibleEmbeddings):
                index.rank(vectors)
        for top_k in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                index.rank([[1, 0]], top_k)
        with self.assertRaises(ValueError):
            index.search(' ', self.encoder)
        meta = copy.deepcopy(index.meta)
        meta['items'][1]['id'] = meta['items'][0]['id']
        with self.assertRaises(IncompatibleEmbeddings):
            EmbeddingIndex(index.vectors, meta)
        with self.assertRaisesRegex(IncompatibleEmbeddings, 'unnormalized'):
            EmbeddingIndex(index.vectors * 2, index.meta)
        (self.root/'invalid.npz').write_bytes(b'not an index')
        with self.assertRaises(IncompatibleEmbeddings):
            EmbeddingIndex.load(self.root/'invalid.npz')
        with self.assertRaises(IncompatibleEmbeddings):
            Gallery(self.root/'invalid.npz', model=MODEL, pretrained=REVISION, csv_sha256='x')
        index.save(self.root/'images.npz')
        with self.assertRaises(IncompatibleEmbeddings):
            Gallery(self.root/'images.npz', model=MODEL, pretrained=REVISION, csv_sha256='x')

    def test_local_decode_orientation_transparency_and_errors(self):
        exif = Image.Exif()
        exif[274] = 6
        Image.new('RGB', (20, 10), 'red').save(self.root/'oriented.jpg', exif=exif)
        image, digest = read_image(self.root/'oriented.jpg')
        self.assertEqual(image.size, (10, 20))
        self.assertEqual(len(digest), 64)
        image.close()
        Image.new('RGBA', (10, 10), (0, 0, 0, 0)).save(self.root/'alpha.png')
        image, _ = read_image(self.root/'alpha.png')
        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
        image.close()
        (self.root/'broken.jpg').write_bytes(b'not a photo')
        with self.assertRaises(ValueError):
            build_images(self.root, self.encoder)
        with self.assertRaises(ValueError):
            build_images(self.root/'missing', self.encoder)
        with self.assertRaises(ValueError):
            build_images(self.root, self.encoder, batch_size=0)

    def test_recall_and_hard_negatives(self):
        index = build_images(self.root, self.encoder)
        queries = [dict(query_id='q1', text='Анджуйское', relevant_image_ids=['a.png']),
                   dict(query_id='q2', text='Анджуйское', relevant_image_ids=['b.png'], hard_negative=True)]
        report = retrieval_report(index, queries, self.encoder)
        self.assertEqual(report['all']['recall_at_1']['value'], .5)
        self.assertEqual(report['all']['recall_at_5']['value'], 1)
        self.assertEqual(report['hard_negatives']['recall_at_1']['value'], 0)
        for bad in ([], queries + queries, [{**queries[0], 'relevant_image_ids': ['missing.png']}]):
            with self.assertRaises(ValueError):
                retrieval_report(index, bad, self.encoder)

    def test_cli_build_search_predict_end_to_end_with_fake_encoder(self):
        def run(*args):
            output = io.StringIO()
            with patch('wineid.clip_zero_shot.JinaCLIPEmbedder', return_value=self.encoder), \
                    patch.object(sys, 'argv', ['clip_zero_shot', *map(str, args), '--allow-remote-code']), \
                    contextlib.redirect_stdout(output):
                main()
            return json.loads(output.getvalue())

        image_index = self.root/'images.npz'
        self.assertEqual(run('build-images', self.root, image_index)['count'], 2)
        with patch.object(sys, 'argv', ['clip_zero_shot', 'audit-images', str(image_index)]), \
                contextlib.redirect_stdout(io.StringIO()) as audited:
            main()
        self.assertTrue(json.loads(audited.getvalue())['snapshot_valid'])
        result = run('search', image_index, 'Этикетка вина «Анджуйское»', '--top-k', 1)
        self.assertEqual(result['candidates'][0]['id'], 'a.png')
        source = self.root/'classes.jsonl'
        source.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in self.classes),
                          encoding='utf-8')
        class_index = self.root/'classes.npz'
        self.assertEqual(run('build-classes', class_index, '--classes', source)['count'], 2)
        result = run('predict', class_index, self.root/'a.png', self.root/'b.png', '--batch-size', 1)
        self.assertEqual([r['predicted_class_id'] for r in result['predictions']], ['a', 'b'])
        self.assertTrue(all(r['status'] == 'ranked_only' and r['confidence_calibrated'] is None
                            for r in result['predictions']))

    def test_cli_help_and_consent_without_optional_dependencies(self):
        cmd = [sys.executable, '-m', 'wineid.clip_zero_shot']
        result = subprocess.run(cmd + ['--help'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(cmd + ['build-images', str(self.root), str(self.root/'new.npz')],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--allow-remote-code', result.stderr)
        self.assertFalse((self.root/'new.npz').exists())


class JinaAdapterTests(unittest.TestCase):
    def test_approval_is_required_before_import_or_download(self):
        with self.assertRaisesRegex(ValueError, 'explicit approval'):
            JinaCLIPEmbedder()
        with self.assertRaises(ValueError):
            JinaCLIPEmbedder(model_name='ViT-B-32', allow_remote_code=True)
        with self.assertRaises(ValueError):
            JinaCLIPEmbedder(revision='main', allow_remote_code=True)

    def test_huggingface_api_contract_without_weights(self):
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        model.encode_text.return_value = np.ones((1, 1024))
        model.encode_image.return_value = np.ones((1, 1024))
        hf = types.ModuleType('transformers')
        hf.AutoModel = Mock()
        hf.AutoModel.from_pretrained.return_value = model
        hf.AutoTokenizer, hf.AutoImageProcessor = Mock(), Mock()
        torch = types.ModuleType('torch')
        torch.float32 = 'float32'
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.inference_mode = contextlib.nullcontext
        with patch.dict(sys.modules, {'transformers': hf, 'torch': torch}):
            encoder = JinaCLIPEmbedder(allow_remote_code=True, batch_size=2)
            kwargs = hf.AutoModel.from_pretrained.call_args.kwargs
            self.assertTrue(kwargs['trust_remote_code'])
            self.assertEqual(kwargs['revision'], REVISION)
            self.assertEqual(kwargs['code_revision'], CODE_REVISION)
            self.assertFalse(kwargs['use_text_flash_attn'])
            self.assertFalse(kwargs['use_vision_xformers'])
            self.assertEqual(encoder.config['preprocess'], PREPROCESS)
            self.assertFalse(hf.AutoImageProcessor.from_pretrained.call_args.kwargs['use_fast'])
            vector = encoder.encode_text('Этикетка вина')
            self.assertIsNone(model.encode_text.call_args.kwargs['task'])
            self.assertEqual(model.encode_text.call_args.args[0], ['Этикетка вина'])
            np.testing.assert_allclose(np.linalg.norm(vector, axis=1), [1])
            self.assertEqual(encoder.encode_images([Image.new('RGB', (10, 10))]).shape, (1, 1024))
            with self.assertRaises(ValueError):
                encoder.encode_images(['https://example.com/image.jpg'])
            with self.assertRaises(ValueError):
                encoder.encode_text([])
            adapter = CLIPEmbedder(allow_remote_code=True)
            with self.assertRaises(IncompatibleEmbeddings):
                adapter([], model_name='old', pretrained='old')
        with self.assertRaises(IncompatibleEmbeddings):
            unit_vectors([[0, 0]], 1)


if __name__ == '__main__':
    unittest.main()
