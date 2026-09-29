import unittest

from wineid.canonical import ProducerAliases, parse_aliases_lines
from wineid.catalog import Wine, normalize
from wineid.search import Index, evidence_tokens, build_idf


def wine(slug, name, producer, raw=None):
    return Wine(slug, slug, name, producer, normalize(name),
                normalize(producer), (2,), raw or {}, 'test')


class SearchV2Tests(unittest.TestCase):
    def setUp(self):
        raw_a = {'Сорт винограда': 'Сибирьковый', 'Категория': 'Белое',
                 'Цвет': 'Светло-соломенный'}
        raw_b = {'Сорт винограда': 'Алиготе', 'Категория': 'Белое',
                 'Цвет': 'Светло-соломенный'}
        self.wines = [
            wine('arpachino-aligote', 'Арпачино Иноходец', 'Вина Арпачина', raw_b),
            wine('arpachino-sibirkovyy', 'Арпачино Иноходец', 'Вина Арпачина', raw_a),
            wine('wine-2020', 'Вино 2020', 'Завод А'),
            wine('wine-2021', 'Вино 2021', 'Завод А'),
        ]
        self.index = Index(self.wines)

    def test_evidence_tokens_and_idf(self):
        tokens = evidence_tokens('Арпачино Иноходец Сибирьковый')
        self.assertIn('сибирьковый', tokens)
        idf = build_idf([tokens, ('сибирьковый',)])
        self.assertGreater(idf['сибирьковый'], idf.get('неизвестный', 0))

    def test_identical_names_broken_by_grape_token(self):
        matches = self.index.search('Арпачино Иноходец Сибирьковый', k=2)
        self.assertEqual(matches[0].slug, 'arpachino-sibirkovyy')
        self.assertGreater(matches[0].score, matches[1].score)
        self.assertIn('сибирьковый', matches[0].match)

    def test_name_only_query_stays_tied(self):
        matches = self.index.search('Арпачино Иноходец', k=2)
        self.assertEqual(matches[0].score, matches[1].score)
        self.assertNotEqual(matches[0].slug, matches[1].slug)

    def test_legacy_distance_ranking_preserved(self):
        matches = self.index.search('Вино 2021\nЗавод А', k=3)
        self.assertEqual(matches[0].slug, 'wine-2021')
        self.assertEqual(len({m.wine_id for m in matches}), 3)

    def test_vintage_hard_mode_drops_other_vintages(self):
        matches = self.index.search('Вино 2021', k=3, vintage_mode='hard')
        slugs = {m.slug for m in matches}
        self.assertIn('wine-2021', slugs)
        self.assertNotIn('wine-2020', slugs)

    def test_equivalence_groups_on_identical_canonical_texts(self):
        # Two identical name+producer+color+vintage => same canonical text.
        groups = self.index.equivalence_groups()
        self.assertTrue(any('arpachino-aligote' in slugs and
                            'arpachino-sibirkovyy' in slugs for slugs in groups.values()))
        stats = self.index.collision_stats()
        self.assertEqual(stats['groups'], 1)
        self.assertEqual(stats['affected_slugs'], 2)

    def test_fuzzy_prefix_filter_keeps_typo_candidates(self):
        # P2-18: a typo that keeps the token prefix stays in the fuzzy pool,
        # while unrelated SKUs (no shared >=3-char prefix) are pruned.
        matches = self.index.search('Сибирьквый', k=20)
        slugs = {m.slug for m in matches}
        self.assertIn('arpachino-sibirkovyy', slugs)
        self.assertNotIn('arpachino-aligote', slugs)

    def test_empty_query(self):
        self.assertEqual(self.index.search(''), [])


class AliasTests(unittest.TestCase):
    def setUp(self):
        raw = {'Сорт винограда': 'Алиготе', 'Категория': 'Белое'}
        self.wines = [wine('fanagoria-aligote', 'Алиготе', 'Фанагория', raw),
                      wine('other-aligote', 'Алиготе', 'Другой Завод', raw)]
        self.aliases = ProducerAliases(parse_aliases_lines(
            ['Фанагория, Fanagoria, Phanagoria']))

    def test_alias_boosts_prod_hit(self):
        with_alias = {c.slug: c.score for c in Index(self.wines, aliases=self.aliases)
                      .search('Fanagoria Алиготе', k=5)}
        without = {c.slug: c.score for c in Index(self.wines).search('Fanagoria Алиготе', k=5)}
        self.assertGreater(with_alias['fanagoria-aligote'],
                           without['fanagoria-aligote'])
        # The unrelated producer is not boosted by another producer's alias.
        self.assertEqual(with_alias['other-aligote'], without['other-aligote'])

    def test_alias_creates_no_second_card(self):
        index = Index(self.wines, aliases=self.aliases)
        self.assertEqual(len(index.wines), 2)
        self.assertEqual(len({w.slug for w in index.wines}), 2)

    def test_canonical_variant_maps_to_one_producer(self):
        self.assertEqual(self.aliases.canonical('Phanagoria'), 'фанагория')


if __name__ == '__main__':
    unittest.main()