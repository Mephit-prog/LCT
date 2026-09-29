"""Stage 4: label extraction -> dedup -> entity resolution -> enrichment with provenance -> features.
External calls (VLM/OCR, web lookup) are injected as callables; nothing here talks to the network."""
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, List, Optional, Any
import difflib
import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz
    def sim(a, b): return fuzz.token_set_ratio(a, b) / 100.0
except Exception:
    def sim(a, b): return difflib.SequenceMatcher(None, a, b).ratio()

LABEL_SCHEMA = {
    "producer": "str|null", "wine_name": "str|null", "grapes": "list[str]", "vintage": "int|null",
    "appellation": "str|null", "region": "str|null", "country": "str|null", "abv": "float|null",
    "confidence": "float 0..1",
}


@dataclass
class Field:
    value: Any
    source: str
    confidence: float
    url: Optional[str] = None
    retrieved: str = field(default_factory=lambda: date.today().isoformat())


def norm_key(rec: dict) -> str:
    s = " ".join(str(rec.get(k) or "") for k in ("producer", "wine_name"))
    return " ".join(s.lower().split()) + f"|{rec.get('vintage') or ''}"


def cluster_labels(records: List[dict], embeddings: Optional[np.ndarray] = None,
                   text_thr=0.9, emb_thr=0.92) -> np.ndarray:
    """Greedy grouping of photos into wines: same vintage & similar text, or very close image embedding."""
    n = len(records); keys = [norm_key(r) for r in records]
    if embeddings is not None:
        E = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    lab = -np.ones(n, int); reps: List[int] = []
    for i in range(n):
        for c, j in enumerate(reps):
            same_v = records[i].get("vintage") == records[j].get("vintage")
            t = sim(keys[i], keys[j]) >= text_thr and same_v
            e = embeddings is not None and E[i] @ E[j] >= emb_thr and same_v
            if t or e:
                lab[i] = c; break
        if lab[i] < 0:
            lab[i] = len(reps); reps.append(i)
    return lab


def resolve_entity(query: dict, candidates: List[dict], min_conf=0.8):
    """Pick best candidate by name similarity, gated by vintage & producer. Returns (cand, confidence)."""
    best, bs = None, 0.0
    for c in candidates:
        s = 0.6 * sim(str(query.get("producer") or "").lower(), str(c.get("producer") or "").lower()) \
            + 0.4 * sim(str(query.get("wine_name") or "").lower(), str(c.get("wine_name") or "").lower())
        if query.get("vintage") and c.get("vintage") and query["vintage"] != c["vintage"]:
            s *= 0.5
        if s > bs:
            best, bs = c, s
    return (best, bs) if bs >= min_conf else (None, bs)


SchemaFields = ["type", "country", "region", "appellation", "grapes", "abv", "price_eur",
                "critic_score", "avg_rating", "n_ratings", "acidity", "tannin", "body", "sweetness"]


class Enricher:
    """lookups: list of callables record -> dict(field -> Field). Higher confidence wins per field.
    Plug in: Wine-Searcher API client, open dataset joins, LLM web-extraction with strict schema."""
    def __init__(self, lookups: List[Callable[[dict], Dict[str, Field]]]):
        self.lookups = lookups

    def enrich(self, rec: dict) -> Dict[str, Field]:
        merged: Dict[str, Field] = {}
        for lk in self.lookups:
            try:
                got = lk(rec)
            except Exception:
                continue
            for k, f in got.items():
                if f.value is None:
                    continue
                if k not in merged or f.confidence > merged[k].confidence:
                    merged[k] = f
        return merged


def to_frame(records: List[dict], enriched: List[Dict[str, Field]], match_conf: List[float]) -> pd.DataFrame:
    rows = []
    for r, e, mc in zip(records, enriched, match_conf):
        row = dict(r); row["match_conf"] = mc
        for k in SchemaFields:
            f = e.get(k)
            row[k] = f.value if f else row.get(k)
            # Raw, uncalibrated source confidence. Entity-match confidence is
            # a separate decision and MUST NOT be treated as field noise.
            row[k + "_conf"] = f.confidence if f else 0.0
            row[k + "_src"] = f.source if f else None
        rows.append(row)
    return pd.DataFrame(rows)


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for k in SchemaFields:
        if k in df:
            out.append(dict(field=k, missing=df[k].isna().mean(), mean_conf=df[k + "_conf"].mean()))
    return pd.DataFrame(out).round(3)


def filter_for_training(df: pd.DataFrame, min_match=0.7) -> pd.DataFrame:
    """Abstain on uncertain entity matches; they are not noisy ABV readings."""
    return df[df["match_conf"] >= min_match].copy()


def audit_fields(labels: pd.DataFrame) -> pd.DataFrame:
    """Manual validation: columns field, source, predicted, truth, confidence.

    Reports accuracy for categories, absolute physical error for numbers, and
    retention under a confidence gate. It does NOT claim confidence calibration
    from a small sample. Entity matches should be audited as a separate field.
    """
    required = {"field", "source", "predicted", "truth", "confidence"}
    if not required <= set(labels):
        raise ValueError(f"missing annotation columns: {required - set(labels)}")
    rows = []
    for (field_name, source), group in labels.groupby(["field", "source"], dropna=False):
        p, t = group.predicted, group.truth
        numeric = pd.to_numeric(p, errors="coerce").notna().all() and pd.to_numeric(t, errors="coerce").notna().all()
        error = float(np.mean(np.abs(p.astype(float) - t.astype(float)))) if numeric else np.nan
        rows.append(dict(field=field_name, source=source, n=len(group),
                         exact_accuracy=float(np.mean(p.to_numpy() == t.to_numpy())),
                         mean_absolute_error=error,
                         retained_at_07=float((group.confidence >= 0.7).mean())))
    return pd.DataFrame(rows)
