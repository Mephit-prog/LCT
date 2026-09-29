import unittest

from wineid.canonical import (VOLUME_RE, canonical_normalize, canonical_query,
                              canonical_text, extract_vintage, name_tokens,
                              token_f1, ProducerAliases, sweetness_token)


class CanonicalTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(canonical_normalize('ЁЖИК «Большой»'), 'ежик большой')
        self.assertEqual(canonical_normalize('  Вино\tКрасное  '), 'вино красное')

    def test_vintage(self):
        self.assertEqual(extract_vintage('Алиготе Баррель 2024'), '2024')
        self.assertEqual(extract_vintage('Без года'), None)
        self.assertEqual(extract_vintage('1950 1999 2000'), '1950')
        self.assertIsNone(extract_vintage('1800 3000'))

    def test_service_tokens_removed_from_name(self):
        self.assertEqual(name_tokens('Вино столовое сухое 0.75'), ())
        self.assertEqual(name_tokens('Алиготе Баррель'), ('алиготе', 'баррель'))
        # vintage and volume are not name tokens
        self.assertEqual(name_tokens('Рислинг 2021'), ('рислинг',))
        self.assertNotIn('полусухое', name_tokens('Рислинг полусухое'))

    def test_token_f1(self):
        self.assertEqual(token_f1(('алиготе', 'баррель'), ('алиготе', 'баррель')), 1.0)
        self.assertAlmostEqual(token_f1(('алиготе', 'баррель', '2024'),
                                        ('алиготе', 'баррель', '2025')), 2 / 3)
        self.assertEqual(token_f1((), ('любой',)), 0.0)

    def test_canonical_template_and_trimming(self):
        text = canonical_text(name_ru='Алиготе Баррель', producer='Коммуналка')
        self.assertEqual(text, 'Фотография бутылки вина «алиготе баррель», '
                               'производитель коммуналка')
        full = canonical_text(name_ru='Алиготе', producer='Коммуналка',
                              color='Белое', vintage='2024')
        self.assertTrue(full.endswith(', белое, 2024'))
        with self.assertRaises(ValueError):
            canonical_text(name_ru='', producer='Коммуналка')

    def test_producer_aliases(self):
        aliases = ProducerAliases({
            'fanagoria': 'фанагория', 'фанагория': 'фанагория'})
        self.assertEqual(aliases.canonical('Fanagoria'), 'фанагория')
        self.assertEqual(aliases.canonical('Фанагория'), 'фанагория')
        self.assertEqual(aliases.lookup('купить Fanagoria вино'), 'фанагория')
        self.assertIsNone(aliases.canonical(''))

    def test_volume_re_matches_larger_volumes(self):
        # P2-12: 3-4 digit volumes are recognized, 19xx/20xx stay vintages.
        self.assertEqual(name_tokens('Рислинг 1000 мл'), ('рислинг',))
        self.assertEqual(name_tokens('Магнум 1500'), ('магнум',))
        self.assertEqual(name_tokens('Каберне 750'), ('каберне',))
        self.assertEqual(name_tokens('Рислинг 2021'), ('рислинг',))
        self.assertEqual(extract_vintage('2024'), '2024')
        self.assertIsNone(VOLUME_RE.fullmatch('2024'))

    def test_canonical_query_without_producer_placeholder(self):
        # P2-7: an unknown producer is omitted, never invented as "Неизвестный".
        text = canonical_query(name_ru='Алиготе Баррель', color='Белое')
        self.assertTrue(text.startswith('Фотография бутылки вина «алиготе баррель»'))
        self.assertNotIn('производитель', text)
        self.assertNotIn('неизвестный', text)
        full = canonical_query(name_ru='Алиготе', producer='Коммуналка', vintage='2024')
        self.assertIn('производитель коммуналка', full)
        self.assertTrue(full.endswith(', 2024'))
        with self.assertRaises(ValueError):
            canonical_query(name_ru='')

    def test_sweetness_token(self):
        self.assertEqual(sweetness_token('suhoe'), 'suhoe')
        self.assertEqual(sweetness_token('polusladkoe'), 'polusladkoe')
        self.assertEqual(sweetness_token('что-то там'), None)


if __name__ == '__main__':
    unittest.main()