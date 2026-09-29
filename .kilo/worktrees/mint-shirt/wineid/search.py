"""Structured candidate generation and ТЗ lexical blending (tz-clip-text-ranking.md §4.1).

Text branch (offline lexical proxy for the CLIP ``s_clip`` term):

    evidence(w) = sum_{t in query_tokens ∩ tokens(w)} idf(t)

Rare, distinctive tokens (e.g. the grape variety on the label) dominate over
common words, so identical-name families can be told apart when the OCR text
contains the discriminative token. Fuzzy token-set ratio is used only to extend
the candidate pool under typos, never as primary evidence. Final order:
(evidence desc, fuzzy desc, edit-distance asc, slug).

Candidate.score = 100 * evidence / pool_max, clamped to [0,100]; it is a
retrieval score, not a calibrated probability. Sweetness/style tokens are kept
as evidence (broader than ТЗ name identity) so that OCR-readable label text can
break ``сухое/полусухое`` ties; a documented extension to be gated by val.

Fuzzy backend policy (review P1-6): ``fuzzywuzzy`` (GPL-2.0, kept by decision)
is the single production backend; ``Index.backend`` exposes the active one. When
it is missing, a non-score-compatible difflib fallback activates for offline
tests with an explicit warning — production must install fuzzywuzzy (license
note in requirements.txt).
"""
import math
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from .catalog import Wine, normalize
from .canonical import (canonical_normalize, canonical_text, extract_vintage,
                        name_tokens, token_f1, VOLUME_RE)

# Version of the lexical blend/scoring formula. Any change to ``_blend`` or the
# evidence/token rules must bump this, because calibrated text policies and
# fingerprints are bound to it (STATUS.md B05).
BLEND_VERSION = 'search-blend-v1'

try:
    from fuzzywuzzy import fuzz
    BACKEND = 'fuzzywuzzy (installed ratio backend)'
    def fuzzy(a, b):
        return fuzz.token_set_ratio(a, b)
except ImportError:
    # Offline-only fallback: scores are NOT FuzzyWuzzy-equivalent, so thresholds
    # calibrated against fuzzywuzzy must not be applied. Production requires
    # fuzzywuzzy to be installed (GPL-2.0 license note in requirements.txt).
    warnings.warn('fuzzywuzzy is not installed; falling back to a non-production '
                  'difflib token-set ratio that is NOT FuzzyWuzzy-equivalent',
                  RuntimeWarning, stacklevel=2)
    BACKEND = 'difflib token-set fallback (NOT FuzzyWuzzy-equivalent; offline only)'
    def fuzzy(a, b):
        if not a or not b:
            return 0
        sa, sb = set(a.split()), set(b.split())
        shared = ' '.join(sorted(sa & sb))
        x = ' '.join(filter(None, (shared, ' '.join(sorted(sa - sb)))))
        y = ' '.join(filter(None, (shared, ' '.join(sorted(sb - sa)))))
        return round(100 * max(SequenceMatcher(None, left, right).ratio()
                               for left, right in ((shared, x), (shared, y), (x, y))))


def levenshtein(a: str, b: str) -> int:
    """Unit insertion/deletion/substitution distance, independent of fuzzy backend."""
    if len(a) < len(b):
        a, b = b, a
    row = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        nxt = [i]
        for j, cb in enumerate(b, 1):
            nxt.append(min(row[j]+1, nxt[-1]+1, row[j-1]+(ca != cb)))
        row = nxt
    return row[-1]


@dataclass(frozen=True)
class Candidate:
    wine_id: str
    slug: str
    score: float  # normalized evidence score [0,100], not probability
    rank: int
    fuzzy_score: int
    edit_distance: int
    normalized_distance: float
    match: str


_WORD_RE = re.compile(r'[^\W_]+', flags=re.UNICODE)


def evidence_tokens(text: str) -> tuple[str, ...]:
    """Query/corpus tokens for IDF evidence: words >=3 chars, no volumes.

    Deliberately broader than ``name_tokens`` (ТЗ §3): keeps sweetness/style
    tokens (``сухое``, ``полусухое``, ``игристое``, ``брют``) because they are
    printed on the label and discriminate colliding SKUs.
    """
    norm = canonical_normalize(text)
    out = []
    for token in _WORD_RE.findall(norm):
        if len(token) <= 2:
            continue
        if VOLUME_RE.fullmatch(token):
            continue
        out.append(token)
    return tuple(dict.fromkeys(out))


def build_idf(token_docs) -> dict[str, float]:
    """Smoothed log IDF over the whole catalog token universe."""
    df = Counter()
    for tokens in token_docs:
        for token in set(tokens):
            df[token] += 1
    n = len(token_docs)
    return {token: math.log((n + 1) / (count + 1)) + 1.0
            for token, count in df.items()}


def _wine_field_text(wine: Wine) -> str:
    """All structured per-field text of one SKU (name, producer, grape,
    category, color, slug). The slug carries catalog-verified attributes and is
    treated as catalog ground truth, never as a filename feature."""
    r = wine.raw or {}
    return ' '.join(filter(None, (
        wine.name, wine.producer, r.get('Сорт винограда', ''),
        r.get('Категория', ''), r.get('Цвет', ''), wine.slug)))


class Index:
    def __init__(self, wines: list[Wine], aliases=None):
        self.wines = wines
        self.backend = BACKEND
        self.documents = [(w, self.documents_by_wine(w)) for w in wines]
        self._field_text = [_wine_field_text(w) for w in wines]
        self._token_docs = [evidence_tokens(text) for text in self._field_text]
        self._field_docs = [tuple(dict.fromkeys(tokens)) for tokens in self._token_docs]
        self._idf = build_idf(self._token_docs)
        self._by_slug = {w.slug: i for i, w in enumerate(wines)}
        # Optional producer aliases (ТЗ §3/§4.1): an alias never creates a second
        # card, it only expands prod_hit on the query side.
        self.aliases = aliases
        variant_map = aliases.variants_by_canonical() if aliases is not None else {}
        self._producer_terms = [
            {canonical_normalize(w.producer)} | variant_map.get(canonical_normalize(w.producer), set())
            for w in wines]

    def documents_by_wine(self, wine):
        # Producer-only identity matches every SKU of that winery. Use it only
        # alongside the title; a missing title must not look like an exact SKU.
        return tuple(dict.fromkeys(filter(None, (wine.name_search,
                    normalize(wine.name + ' ' + wine.producer)))))

    # ------------------------------------------------------------------
    # Equivalence groups (ТЗ refusal semantics: no forced exact Top-1)
    # ------------------------------------------------------------------
    def equivalence_groups(self, *, include_vintage: bool = True) -> dict[str, list[str]]:
        """Slugs whose canonical texts are identical => indistinguishable by index.

        Key is the canonical text of (name_ru, producer, color, vintage).
        Different vintages of one name stay different wine_ids (ТЗ §3).
        """
        groups: dict[str, list[str]] = {}
        for w in self.wines:
            r = w.raw or {}
            vintage = extract_vintage(w.name) if include_vintage else None
            color = canonical_normalize(r.get('Категория', '')).capitalize() or None
            key = canonical_text(name_ru=w.name, producer=w.producer,
                                 color=color, vintage=vintage)
            groups.setdefault(key, []).append(w.slug)
        return {key: slugs for key, slugs in groups.items() if len(slugs) > 1}

    def collision_stats(self) -> dict:
        groups = self.equivalence_groups()
        affected = sum(len(slugs) for slugs in groups.values())
        return {'groups': len(groups), 'affected_slugs': affected,
                'class_fraction': affected / len(self.wines) if self.wines else None,
                'largest_group': max((len(s) for s in groups.values()), default=0)}

    # ------------------------------------------------------------------
    # Text-branch search
    # ------------------------------------------------------------------
    def search(self, text: str, k: int = 20, pool: int = 100,
               vintage_mode: str = 'soft') -> list[Candidate]:
        """Rank wines by IDF token evidence; fuzzy extends the pool under typos.

        vintage_mode: 'soft' (default) applies a score term; 'hard' drops SKUs
        with a different vintage from the pool (ТЗ §4.1 step 2).
        """
        if vintage_mode not in ('soft', 'hard'):
            raise ValueError("vintage_mode must be 'soft' or 'hard'")
        q = normalize(text)
        if not q:
            return []
        q_canon = canonical_normalize(text)
        q_vintage = extract_vintage(q_canon)
        q_tokens = evidence_tokens(q_canon)
        q_name_tokens = name_tokens(q_canon)
        lines = [normalize(line) for line in text.splitlines()]
        snippets = tuple(dict.fromkeys(s for s in [q, *lines] if s))[:30]

        # Primary evidence: containment of (rare) query tokens in SKU fields.
        scored = []
        for i, w in enumerate(self.wines):
            tokens = self._token_docs[i]
            token_set = set(tokens)
            overlap = [t for t in q_tokens if t in token_set]
            if overlap:
                evidence = sum(self._idf.get(t, 0.0) for t in overlap)
                scored.append((i, w, evidence, tuple(overlap)))
        scored.sort(key=lambda item: (-item[2], item[1].slug))
        pool_max = scored[0][2] if scored else 0.0

        # Fuzzy extension only for recall (typos); never primary evidence.
        need = max(pool, k)
        hit_slugs = {w.slug for _, w, _, _ in scored}
        extra = []
        if len(scored) < need:
            # P2-18: cheap token-prefix pre-filter bounds the O(N * snippets *
            # docs) fuzzy pass to wines sharing a >=3-char token prefix with the
            # query — plausible typos keep the prefix, unrelated SKUs are pruned.
            # Larger catalogs should replace this with an n-gram/trie index.
            prefixable = self._prefix_candidate_indices(q_tokens) if q_tokens else None
            for i, w in enumerate(self.wines):
                if w.slug in hit_slugs or (prefixable is not None and i not in prefixable):
                    continue
                docs = self._field_docs[i]
                if not docs:
                    continue
                best = max(((fuzzy(s, d), s, d) for s in snippets for d in docs))
                extra.append((best[0], w, best[1], best[2]))
            extra.sort(key=lambda x: (-x[0], x[1].slug))

        raw = []
        for i, w, evidence, overlap in scored:
            edit, s, d = self._best_edit(snippets, w)
            distance = edit / max(len(s), len(d), 1) if (s or d) else 0.0
            fs = max((fuzzy(sn, dd) for sn in snippets for dd in self._field_docs[i]),
                     default=0)
            if vintage_mode == 'hard' and q_vintage:
                w_vintage = extract_vintage(w.name)
                if w_vintage is not None and w_vintage != q_vintage:
                    continue
            raw.append((evidence, fs, w, overlap, edit, distance))
        for fs, w, sn, doc in extra:
            edit, s, d = self._best_edit(snippets, w)
            distance = edit / max(len(s), len(d), 1) if (s or d) else 0.0
            if vintage_mode == 'hard' and q_vintage:
                w_vintage = extract_vintage(w.name)
                if w_vintage is not None and w_vintage != q_vintage:
                    continue
            raw.append((0.0, fs, w, (sn,), edit, distance))

        # ТЗ §4.1: score = s_clip + 0.15*token_f1 + 0.10*prod_hit.
        # Offline proxy for s_clip: retrieval = 0.5*distance_score + 0.5*evidence_norm
        # (weights are placeholders pending val calibration). token_f1 uses the
        # ТЗ name tokens; prod_hit checks whether the card producer string is
        # literally contained in the query.
        raw.sort(key=lambda item: (-self._blend(item, q_name_tokens, q_canon, pool_max),
                                   -item[1], item[4], item[2].slug))
        out = []
        for rank, (evidence, fs, w, overlap, edit, distance) in enumerate(raw[:k], 1):
            score = self._blend((evidence, fs, w, overlap, edit, distance),
                                q_name_tokens, q_canon, pool_max)
            out.append(Candidate(w.wine_id, w.slug, round(max(0.0, min(100.0, score)), 2),
                                 rank, fs, edit, round(distance, 4), ' | '.join(overlap)))
        return out

    def _blend(self, item, q_name_tokens, q_canon, pool_max):
        evidence, fs, w, overlap, edit, distance = item
        evidence_norm = 100.0 * evidence / pool_max if pool_max > 0 else 0.0
        distance_score = (1 - distance) * 100.0
        retrieval = 0.5 * distance_score + 0.5 * evidence_norm
        name_f1 = token_f1(q_name_tokens, name_tokens(w.name))
        index = self._by_slug.get(w.slug)
        terms = (self._producer_terms[index] if index is not None
                 else {canonical_normalize(w.producer)})
        prod_hit = 1.0 if any(term and term in q_canon for term in terms) else 0.0
        return max(0.0, min(100.0, retrieval + 15.0 * name_f1 + 10.0 * prod_hit))

    def _prefix_candidate_indices(self, q_tokens):
        """Indices of wines whose field tokens share a >=3-char prefix with the
        query tokens; bounds the fuzzy pool-extension pass (P2-18)."""
        prefixes = {t[:3] for t in q_tokens if len(t) >= 3}
        if not prefixes:
            return frozenset()
        return frozenset(i for i, docs in enumerate(self._field_docs)
                         if any(len(d) >= 3 and d[:3] in prefixes for d in docs))

    def _best_edit(self, snippets, wine):
        return min(((levenshtein(s, d), s, d) for s in snippets
                    for d in self.documents_by_wine(wine)),
                   key=lambda item: (item[0]/max(len(item[1]), len(item[2]), 1),
                                     -len(item[2]), item[0]))
