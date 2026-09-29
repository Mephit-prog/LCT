"""CSV ingestion. A slug is the identifier; filenames are never search features."""
import csv
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


def normalize(text: str) -> str:
    text = unicodedata.normalize('NFKC', text).casefold()
    return ' '.join(re.findall(r'[^\W_]+', text, flags=re.UNICODE))


@dataclass(frozen=True)
class Wine:
    wine_id: str
    slug: str
    name: str
    producer: str
    name_search: str
    producer_search: str
    source_row_ids: tuple[int, ...]
    raw: dict
    csv_sha256: str


def load_catalog(path: str | Path) -> tuple[list[Wine], dict]:
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    groups = defaultdict(list)
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        required = {'Slug', 'Название вина', 'Винодельня'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'CSV missing columns: {required - set(reader.fieldnames or [])}')
        for number, row in enumerate(reader, 2):
            slug = row['Slug'].strip()
            if not slug:
                raise ValueError(f'empty Slug at CSV row {number}')
            groups[slug].append((number, row))
    conflicts = {}
    wines = []
    for slug, rows in groups.items():
        raw = rows[0][1]
        if any(row != raw for _, row in rows[1:]):
            conflicts[slug] = [number for number, _ in rows]
        name = raw['Название вина'].strip()
        producer = raw['Винодельня'].strip()
        wines.append(Wine(slug, slug, name, producer, normalize(name),
                          normalize(producer), tuple(n for n, _ in rows), raw, digest))
    # Never silently choose the first of contradictory rows.
    if conflicts:
        raise ValueError(f'conflicting duplicate slugs: {json.dumps(conflicts, ensure_ascii=False)}')
    return wines, {'csv_sha256': digest, 'rows': sum(len(r) for r in groups.values()),
                   'unique_slugs': len(wines), 'exact_duplicates': sum(len(r)-1 for r in groups.values()),
                   'empty_names': sum(not w.name_search for w in wines)}


def export_catalog(csv_path, output):
    wines, report = load_catalog(csv_path)
    with open(output, 'w', encoding='utf-8') as f:
        for w in wines:
            f.write(json.dumps(w.__dict__, ensure_ascii=False) + '\n')
    return report


# ---------------------------------------------------------------------------
# Structured attributes per tz-wine-label-retrieval.md (wines.csv schema).
# These helpers never change the Wine dataclass contract used by tests.
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {'белое': 'Белое', 'красное': 'Красное',
                   'розовое': 'Розовое', 'оранжевое': 'Оранжевое'}


def structured_fields(row: dict) -> dict:
    """Parse ТЗ fields from one raw CSV row (empty strings when absent).

    - name_ru: canonical name (no marketing tail is stripped automatically);
    - producer: canonical producer (aliases resolved elsewhere);
    - color: one of Красное/Белое/Розовое/Оранжевое from Категория, else '';
      the free-text ``Цвет`` field is deliberately not trusted here;
    - vintage: standalone 19xx/20xx from the name, else '';
    - grape: canonical grape variety field;
    - sweetness: suhoe/polusuhoe/polusladkoe/sladkoe/bryut token if present.
    """
    from .canonical import canonical_normalize, extract_vintage, sweetness_token
    name = (row.get('Название вина') or '').strip()
    producer = (row.get('Винодельня') or '').strip()
    category = (row.get('Категория') or '').strip().lower()
    return {
        'name_ru': canonical_normalize(name),
        'producer': canonical_normalize(producer),
        'color': CATEGORY_COLORS.get(category, ''),
        'vintage': extract_vintage(name) or '',
        'grape': canonical_normalize(row.get('Сорт винограда') or ''),
        'category': (row.get('Категория') or '').strip(),
        'sweetness': sweetness_token((row.get('Slug') or '') + ' ' + name) or '',
    }


def structured_wine(wine: Wine) -> dict:
    """Structured fields for an already loaded Wine (raw row may be shared)."""
    return structured_fields(wine.raw or {})
