"""Versioned, score-scale-aware retrieval fusion. Scores are NOT probabilities.

No photo reference is not a negative observation. A SKU outside a returned text
Top-K, by contrast, has a known lower bound. Keep these cases separate.
"""
import math
import re
from dataclasses import dataclass, field
from collections import defaultdict

from .canonical import canonical_normalize, name_tokens
from .catalog import structured_wine

VERSION = 'family-gap-v2'  # suffix alcohol disabled by default; rank validation added
SOURCES = {'text': ('text_score_0_100', 0, 100),
           'photo': ('cosine', -1, 1), 'class': ('cosine', -1, 1)}
_LAT2CYR = str.maketrans('aceopxykmthb', 'асеорхукмтнв')
_WORD = re.compile(r'[^\W_]+', re.UNICODE)
_SWEET = ((r'\bэкстра\s*брют\b', 'ekstra-bryut'), (r'\bбрют\b', 'bryut'),
          (r'\bполусладк\w*', 'polusladkoe'), (r'\bсладк\w*', 'sladkoe'),
          (r'\bполусух\w*', 'polusuhoe'), (r'\bсух\w*', 'suhoe'))
_COLOR = re.compile(r'\b(белое|красное|розовое|оранжевое)\b')
_ALCOHOL = re.compile(r'(?<![\d.,])(\d{1,2}(?:[.,]\d{1,2})?)\s*%')
_YEAR = re.compile(r'(?<!\d)(?:19|20)\d{2}(?!\d)')


def clean_ocr(text: str) -> str:
    """Fix mixed-script Cyrillic words, not genuine Latin brand/variety names."""
    def fix(match):
        token = match.group()
        if re.search('[а-яё]', token):
            return token.translate(_LAT2CYR).replace('0', 'о')
        if any(c.isdigit() for c in token) and re.fullmatch(r'[0-9oо]+', token):
            return token.replace('o', '0').replace('о', '0')
        return token
    return _WORD.sub(fix, (text or '').casefold())


def _sweet(text):
    for pattern, code in _SWEET:
        if re.search(pattern, text):
            return code
    return None


def _grapes(text):
    return {canonical_normalize(g) for g in re.split(r'[,;/]', text or '')
            if len(canonical_normalize(g)) >= 4}


def _grape_hits(text, vocabulary):
    # Prefer longest matches and avoid a shorter grape contained inside a longer one.
    found = set()
    for grape in sorted(vocabulary, key=lambda g: (-len(g), g)):
        if re.search(r'(?<!\w)' + re.escape(grape) + r'(?!\w)', text) and not any(
                grape in longer for longer in found):
            found.add(grape)
    return found


def ocr_attributes(text, vocabulary=()):
    t = canonical_normalize(clean_ocr(text))
    # Foundation dates often coexist with a vintage on the same label.
    # Ignore only dates explicitly introduced as foundation dates; an
    # unqualified second date stays ambiguous rather than a hard conflict.
    years = set()
    for match in _YEAR.finditer(t):
        prefix = t[max(0, match.start()-32):match.start()]
        if re.search(r'(?:основан\w*|since|estd?\.?|founded)\s*(?:в\s*)?$', prefix):
            continue
        years.add(match.group())
    alcohols = {float(m.group(1).replace(',', '.')) for m in _ALCOHOL.finditer(t)}
    alcohols = {a for a in alcohols if 5 <= a <= 25}
    color = _COLOR.search(t)
    return {'vintage': next(iter(years)) if len(years) == 1 else None,
            'sweet': _sweet(t), 'color': color.group(1).capitalize() if color else None,
            'grapes': _grape_hits(t, vocabulary),
            'alcohol': next(iter(alcohols)) if len(alcohols) == 1 else None,
            'text': t}


def _slug_alcohol(slug):
    # Only the usual 2/3 digit alcohol suffix; -1/-2 duplicates, years and
    # arbitrary numeric suffixes are not strengths. This remains an heuristic.
    match = re.search(r'-(\d{2,3})$', slug)
    if not match:
        return None
    token = match.group(1)
    value = float(token) if len(token) == 2 else float(token[:-1] + '.' + token[-1])
    return value if 5 <= value <= 25 else None


@dataclass(frozen=True)
class CandidateEvidence:
    slug: str
    source: str
    score: float
    rank: int
    score_type: str
    provenance: str  # e.g. search version or archive/checkpoint SHA
    reference_id: str | None = None
    query_view: str | None = None

    def __post_init__(self):
        if self.source not in SOURCES:
            raise ValueError('unknown source')
        kind, low, high = SOURCES[self.source]
        if (not self.slug or self.score_type != kind or not self.provenance
                or type(self.rank) is not int or self.rank < 1
                or not isinstance(self.score, (int, float)) or not math.isfinite(self.score)
                or not low - 1e-3 <= self.score <= high + 1e-3):
            raise ValueError('invalid candidate evidence')


@dataclass(frozen=True)
class BlendConfig:
    mode: str = 'gap'  # 'rrf' is the rank-only ablation
    top_k: int = 50
    max_candidates: int = 300
    tau: dict = field(default_factory=lambda: {'photo': .05, 'class': .05, 'text': 10.})
    weights: dict = field(default_factory=lambda: {'photo': 1., 'class': .3, 'text': .7})
    attributes: dict = field(default_factory=lambda: {
        'vintage': (1.5, -2.), 'sweet': (1., -1.5), 'grape': (.7, -.7),
        'color': (.5, -1.), 'alcohol': (.5, -.5), 'producer': (.5, -1.)})
    member_photo: float = .2
    member_text: float = .1
    rrf_k: int = 60
    use_slug_alcohol: bool = False  # unverified suffix convention; diagnostic opt-in only

    def __post_init__(self):
        if (self.mode not in ('gap', 'rrf') or type(self.top_k) is not int or self.top_k < 1
                or type(self.max_candidates) is not int or self.max_candidates < self.top_k
                or type(self.rrf_k) is not int or self.rrf_k < 1
                or type(self.use_slug_alcohol) is not bool
                or set(self.tau) != SOURCES.keys() or set(self.weights) != SOURCES.keys()
                or set(self.attributes) != {'vintage', 'sweet', 'grape', 'color', 'alcohol', 'producer'}
                or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
                       for x in self.tau.values())
                or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0
                       for x in (*self.weights.values(), self.member_photo, self.member_text))
                or any(len(pair) != 2 or not all(isinstance(x, (int, float)) and math.isfinite(x)
                                                 for x in pair) for pair in self.attributes.values())):
            raise ValueError('invalid blend configuration')


def _attribute_score(attrs, fields, wine, weights, canonical_producer, use_slug_alcohol):
    if attrs is None:
        return 0.
    # Missing/ambiguous on either side is neutral. Do not infer vintage from slug.
    target = {'vintage': fields['vintage'] or None, 'sweet': _sweet(
              canonical_normalize(wine.name) + ' ' + wine.slug.replace('-', ' ')) or fields['sweetness'],
              'color': fields['color'] or None, 'grape': _grapes((wine.raw or {}).get('Сорт винограда', '')),
              'alcohol': _slug_alcohol(wine.slug) if use_slug_alcohol else None,
              'producer': canonical_producer}
    total = 0.
    for key in weights:
        query = attrs['grapes'] if key == 'grape' else attrs.get(key)
        candidate = target[key]
        if not query or not candidate:
            continue
        match = bool(query & candidate) if key == 'grape' else (
            abs(query - candidate) <= .25 if key == 'alcohol' else query == candidate)
        total += weights[key][0 if match else 1]
    return total


class Blender:
    def __init__(self, wines, *, aliases=None, config=None):
        self.config = config or BlendConfig()
        self.wines = {w.slug: w for w in wines}
        if len(self.wines) != len(wines):
            raise ValueError('duplicate slugs')
        self.fields = {w.slug: structured_wine(w) for w in wines}
        self.producer_aliases = aliases.variants if aliases else {}
        self.producer_aliases = {**{canonical_normalize(w.producer): canonical_normalize(w.producer)
                                    for w in wines}, **self.producer_aliases}
        self.families = defaultdict(list)
        self.key = {}
        for w in wines:
            producer = aliases.canonical(w.producer) if aliases else canonical_normalize(w.producer)
            family = (producer, name_tokens(w.name))
            self.key[w.slug] = family
            self.families[family].append(w.slug)
        self.vocab = set().union(*(_grapes((w.raw or {}).get('Сорт винограда', '')) for w in wines)) if wines else set()

    def rank(self, evidence, ocr_text=None):
        """Union per-source top-K, expand full families, then score & rank.

        `evidence` can contain all class/gallery cosines; no need to serialize
        them all to the HTTP client. No candidates without observations.
        """
        per_source = defaultdict(list)
        seen = set()
        for item in evidence:
            if item.slug not in self.wines:
                raise ValueError('evidence references unknown slug')
            if (item.source, item.slug) in seen:
                raise ValueError('duplicate source/slug')
            seen.add((item.source, item.slug))
            per_source[item.source].append(item)
        for source, items in per_source.items():
            if len({item.rank for item in items}) != len(items):
                raise ValueError(f'duplicate ranks in {source} source')
            ordered_by_rank = sorted(items, key=lambda e: e.rank)
            if any(a.score + 1e-6 < b.score for a, b in zip(ordered_by_rank, ordered_by_rank[1:])):
                raise ValueError(f'inconsistent ranks and scores in {source} source')
        if not per_source:
            return []
        config = self.config
        selected = set()
        ordered = {}
        for source, items in per_source.items():
            items.sort(key=lambda e: (-e.score, e.slug))
            ordered[source] = {e.slug: e for e in items}
            selected.update(e.slug for e in items[:config.top_k])
        # Limit families atomically. Avoid cutting an otherwise retrievable member.
        candidates = set(selected)
        for family in sorted({self.key[s] for s in selected}):
            members = self.families[family]
            if len(candidates | set(members)) <= config.max_candidates:
                candidates.update(members)
        if len(candidates) > config.max_candidates:
            # Requested budget smaller than union: truncate deterministically by best rank.
            candidates = set(sorted(candidates, key=lambda s: (min(
                (i for source, items in per_source.items() for i, e in enumerate(items, 1)
                 if e.slug == s), default=10**9), s))[:config.max_candidates])
        if not candidates:
            return []
        gaps = defaultdict(dict)
        contributions = defaultdict(dict)
        for source, items in per_source.items():
            best = items[0].score
            floor = (items[min(config.top_k, len(items)) - 1].score - best) / config.tau[source]
            for slug in candidates:
                item = ordered[source].get(slug)
                # Photo/class without an indexed SKU: missing reference, not low cosine.
                gap = (item.score - best) / config.tau[source] if item else (
                    floor - 1 if source == 'text' else 0.)
                gaps[slug][source] = gap
                if config.mode == 'gap':
                    contributions[slug][source] = config.weights[source] * gap
                else:
                    # Missing has no rank contribution. Scaling by k keeps the
                    # attribute bonuses in approximately the same unit as gap.
                    contributions[slug][source] = (config.weights[source] * config.rrf_k /
                        (config.rrf_k + item.rank)) if item and item.rank <= config.top_k else 0.
        bases = {s: sum(contributions[s].values()) for s in candidates}
        family_score = {family: max(bases[s] for s in members if s in candidates)
                        for family, members in self.families.items() if any(s in candidates for s in members)}
        attrs = ocr_attributes(ocr_text, self.vocab) if ocr_text else None
        if attrs:
            found = [(len(variant), canonical) for variant, canonical in self.producer_aliases.items()
                     if variant and re.search(r'(?<!\w)' + re.escape(variant) + r'(?!\w)', attrs['text'])]
            top_len = max((length for length, _ in found), default=0)
            matched = {canonical for length, canonical in found if length == top_len}
            attrs['producer'] = next(iter(matched)) if len(matched) == 1 else None
        rows = []
        for slug in candidates:
            wine = self.wines[slug]
            bonus = _attribute_score(attrs, self.fields[slug], wine, config.attributes,
                                     self.key[slug][0], config.use_slug_alcohol)
            member = (config.member_photo * gaps[slug].get('photo', 0.) +
                      config.member_text * gaps[slug].get('text', 0.)) if config.mode == 'gap' else 0.
            # Multi-family rerank: label features can overturn a close family.
            score = family_score[self.key[slug]] + member + bonus
            rows.append({'slug': slug, 'wine_id': wine.wine_id, 'blend_score': score,
                         'attribute_score': bonus, 'signals': {
                             source: {'score': ordered[source][slug].score,
                                      'rank': ordered[source][slug].rank,
                                      'provenance': ordered[source][slug].provenance,
                                      'reference_id': ordered[source][slug].reference_id,
                                      'query_view': ordered[source][slug].query_view}
                             for source in ordered if slug in ordered[source]},
                         'family_size': len(self.families[self.key[slug]])})
        # Stable duplicate tie break: unsuffixed slug preferred to -1/-2,
        # then verified reference if present, then alphabetical slug.
        rows.sort(key=lambda r: (-r['blend_score'], bool(re.search(r'-[12]$', r['slug'])),
                                 'photo' not in r['signals'], r['slug']))
        for i, row in enumerate(rows, 1):
            row['rank'] = i
            row['score'] = round(max(0., 100. + 10. * (row['blend_score'] - rows[0]['blend_score'])), 2)
        return rows
