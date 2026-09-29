"""Canonical text and normalization per tz-clip-text-ranking.md / tz-wine-label-retrieval.md.

One template for index, training and text queries:

    Фотография бутылки вина «{name_ru}», производитель {producer}, {color}, {vintage}

Empty tails are trimmed identically. Marketing descriptions, review text,
back-label text and raw filenames never enter the index.

Normalization before building the string:
- NFKC, casefold, ``ё`` -> ``е``, quotes removed, whitespace collapsed;
- no automatic transliteration;
- latin/russian producer names are linked by an alias table;
- service tokens (``вино``, ``столовое``, ``сухое``, ``полусладкое``, ``згу``,
  ``знпм``, volume ``0.75``, ...) are not treated as part of the name.
"""
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# Service tokens that must not be considered part of a name (ТЗ §3, §3).
SERVICE_TOKENS = frozenset({
    'вино', 'вина', 'вин', 'столовое', 'столовый', 'столовая',
    'сухое', 'сухой', 'сухая', 'сухие',
    'полусухое', 'полусухой', 'полусухая', 'полусухие',
    'полусладкое', 'полусладкий', 'полусладкая', 'полусладкие',
    'сладкое', 'сладкий', 'сладкая', 'сладкие',
    'згу', 'знпм', 'тз', 'шт', 'л', 'мл',
    'игристое', 'игристый', 'игристая', 'игристые',
    'брют', 'экстра', 'брю', 'экстрабрют',
})

# Volume patterns such as "0.75", "0,75", "750 мл", "0.7", "1000 мл".
# 3-4 digit quantities qualify, but 19xx/20xx are always vintages (VINTAGE_RE)
# and must never be swallowed as volumes.
VOLUME_RE = re.compile(r'(?<!\d)(?:0[,.]\d{1,2}|(?!19\d\d|20\d\d)\d{3,4})(?!\d)\s*(?:л|мл|litre|литр)?', re.IGNORECASE)

# Vintage: four digits 19xx or 20xx, not part of a longer number.
VINTAGE_RE = re.compile(r'(?<!\d)(19|20)\d{2}(?!\d)')

# Canonical template. Empty tails are trimmed identically for every wine.
TEMPLATE = 'Фотография бутылки вина «{name_ru}», производитель {producer}{tail}'
# Query-side template used when the producer is not recognized: the clause is
# omitted rather than inventing a placeholder (canonical_query contract).
QUERY_TEMPLATE = 'Фотография бутылки вина «{name_ru}»{tail}'


def canonical_normalize(text: str) -> str:
    """ТЗ normalization: casefold, ё->е, strip quotes, collapse whitespace."""
    text = unicodedata.normalize('NFKC', text).casefold().replace('ё', 'е')
    text = text.replace('«', ' ').replace('»', ' ').replace('"', ' ').replace("'", ' ')
    return ' '.join(text.split())


def extract_vintage(text: str) -> str | None:
    """First standalone 19xx/20xx year, else None. Never read from slug."""
    match = VINTAGE_RE.search(canonical_normalize(text) if text else '')
    return match.group(0) if match else None


def name_tokens(text: str) -> tuple[str, ...]:
    """Tokens >2 chars without service tokens and volumes (ТЗ §4.1)."""
    norm = canonical_normalize(text)
    tokens = re.findall(r'[^\W_]+', norm, flags=re.UNICODE)
    out = []
    for token in tokens:
        token = token.rstrip('.')
        if len(token) <= 2 or token in SERVICE_TOKENS:
            continue
        if VOLUME_RE.fullmatch(token):
            continue
        if VINTAGE_RE.fullmatch(token):
            continue
        out.append(token)
    return tuple(dict.fromkeys(out))


def token_f1(query_tokens: tuple[str, ...], name_tokens_other: tuple[str, ...]) -> float:
    """F1 over token overlap (words >2 chars, no service tokens). 0..1."""
    if not query_tokens or not name_tokens_other:
        return 0.0
    q, n = set(query_tokens), set(name_tokens_other)
    common = len(q & n)
    if common == 0:
        return 0.0
    precision = common / len(q)
    recall = common / len(n)
    return 2 * precision * recall / (precision + recall)


def canonical_text(*, name_ru: str, producer: str,
                   color: str | None = None, vintage: str | None = None) -> str:
    """Build the one canonical string; empty tails are trimmed identically."""
    name_ru = (canonical_normalize(name_ru) or '').strip()
    producer = (canonical_normalize(producer) or '').strip()
    if not name_ru or not producer:
        raise ValueError('name_ru and producer are required')
    tail = ''
    if color:
        tail += ', ' + canonical_normalize(color)
    if vintage:
        tail += ', ' + vintage
    return TEMPLATE.format(name_ru=name_ru, producer=producer, tail=tail)


def canonical_query(name_ru: str, producer: str | None = None,
                    color: str | None = None, vintage: str | None = None) -> str:
    """Canonical string for a text query. Unrecognized words are not invented.

    Only the recognized name/producer (via aliases) plus optional color/vintage
    enter the string; unknown tokens are dropped (ТЗ §4.1 step 4). When the
    producer is unknown it is omitted entirely — a placeholder is never invented
    (an empty producer clause, not "Неизвестный").
    """
    name_ru = (canonical_normalize(name_ru) or '').strip()
    if not name_ru:
        raise ValueError('name_ru is required')
    producer = (canonical_normalize(producer) or '').strip() if producer else ''
    tail = ''
    if color:
        tail += ', ' + canonical_normalize(color)
    if vintage:
        tail += ', ' + vintage
    if producer:
        return TEMPLATE.format(name_ru=name_ru, producer=producer, tail=tail)
    return QUERY_TEMPLATE.format(name_ru=name_ru, tail=tail)


@dataclass(frozen=True)
class ProducerAliases:
    """Latin/russian producer names mapped to one canonical producer.

    An alias never creates a second card; it is substituted on the query side
    only. Mapping: each entry maps variants -> canonical producer.
    """
    variants: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> 'ProducerAliases':
        path = Path(path)
        variants = {}
        if not path.is_file():
            return cls(variants)
        for line in path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [p.strip() for p in line.split(',') if p.strip()]
            if len(parts) < 2:
                continue
            canonical = parts[0]
            canon_norm = canonical_normalize(canonical)
            variants[canon_norm] = canon_norm
            for variant in parts[1:]:
                variants[canonical_normalize(variant)] = canon_norm
        return cls(variants)

    def canonical(self, producer: str | None) -> str | None:
        if not producer:
            return None
        norm = canonical_normalize(producer)
        return self.variants.get(norm, norm)

    def variants_by_canonical(self) -> dict[str, set[str]]:
        """All known (normalized) variants per canonical producer."""
        out: dict[str, set[str]] = {}
        for variant, canonical in self.variants.items():
            out.setdefault(canonical, set()).add(variant)
        return out

    def lookup(self, text: str) -> str | None:
        """Find the canonical producer whose any variant appears in text."""
        norm = canonical_normalize(text)
        best, best_len = None, 0
        for variant, canonical in self.variants.items():
            if variant and variant in norm and len(variant) > best_len:
                best, best_len = canonical, len(variant)
        return best


def parse_aliases_lines(lines) -> dict[str, str]:
    """Parse alias lines for tests/examples: 'Canonical, variant1, variant2'."""
    variants = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = [p.strip() for p in line.split(',') if p.strip()]
        if len(parts) < 2:
            continue
        canonical = parts[0]
        canon_norm = canonical_normalize(canonical)
        variants[canon_norm] = canon_norm
        for variant in parts[1:]:
            variants[canonical_normalize(variant)] = canon_norm
    return variants


def sweetness_token(slug_or_text: str) -> str | None:
    """Extract sweetness/style token if present (suhoe/polusuhoe/polusladkoe/sladkoe/bryut)."""
    norm = canonical_normalize(slug_or_text)
    for token in ('polusladkoe', 'polusuhoe', 'sladkoe', 'suhoe', 'ekstra-bryut', 'bryut'):
        if token in norm:
            return token
    return None