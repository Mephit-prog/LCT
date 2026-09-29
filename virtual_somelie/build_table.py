"""Fit fixed encoders on *allowed* catalog rows; never on test ratings."""
import ast
import numpy as np
import pandas as pd
from .table import WineTable, onehot, multihot


def factorize(series):
    codes, uniq = pd.factorize(series.astype("string"), sort=True)
    return codes.astype(int), max(len(uniq), 1)


def parse_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    if x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NA:
        return []
    try:
        v = ast.literal_eval(x)
        return list(v) if isinstance(v, (list, tuple)) else [str(v)]
    except (ValueError, SyntaxError, TypeError):
        return [s.strip() for s in str(x).split(",") if s.strip()]


def _category(values, fit_values):
    fit_values = fit_values.astype('string')
    unique = sorted(fit_values.dropna().unique().tolist())
    lookup = {value: i for i, value in enumerate(unique)}
    n = max(len(unique), 1)
    codes = values.astype('string').map(lookup).fillna(-1).to_numpy(dtype=int)
    known = fit_values.map(lookup).dropna().to_numpy(dtype=int)
    P = onehot(codes, n)
    # Freeze unknown-category embedding from FIT rows, not the held-out catalog.
    marginal = np.bincount(known, minlength=n) / len(known) if len(known) else np.full(n, 1.0 / n)
    P[codes == -1] = marginal
    return P


def _blend_weights(x):
    """Share-valued blend: dict, [(grape, share)], or equally weighted list."""
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except (ValueError, SyntaxError):
            x = parse_list(x)
    if isinstance(x, dict):
        weights = x
    elif isinstance(x, (list, tuple)) and x and all(isinstance(v, (list, tuple)) and len(v) == 2 for v in x):
        weights = dict(x)
    else:
        values = parse_list(x)
        weights = {value: 1.0 for value in values}
    weights = {key: float(value) for key, value in weights.items()}
    if any(not np.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError('grape shares must be finite and nonnegative')
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()} if total else {}


def _list_factor(values, fit_values, mode='presence'):
    if mode == 'shares':
        train = [_blend_weights(x) for x in fit_values]
        parsed = [_blend_weights(x) for x in values]
        vocab = {g: i for i, g in enumerate(sorted({g for row in train for g in row}))}
        n = max(len(vocab), 1)
        counts = np.zeros(n)
        for row in train:
            for g, weight in row.items():
                counts[vocab[g]] += weight
        marginal = counts / counts.sum() if counts.sum() else np.full(n, 1 / n)
        P = np.zeros((len(parsed), n))
        for i, row in enumerate(parsed):
            unknown = 0.0
            for g, weight in row.items():
                if g in vocab:
                    P[i, vocab[g]] += weight
                else:
                    unknown += weight
            P[i] += (unknown if row else 1.0) * marginal
        return P
    if mode != 'presence':
        raise ValueError('multi mode must be presence or shares')
    parsed = [parse_list(x) for x in values]
    train = [parse_list(x) for x in fit_values]
    vocab = {g: i for i, g in enumerate(sorted({g for row in train for g in row}))}
    n = max(len(vocab), 1)
    encode = lambda row: [vocab[g] for g in row if g in vocab]
    P = multihot([encode(row) for row in parsed], n)
    counts = np.bincount([g for row in train for g in encode(row)], minlength=n)
    marginal = counts / counts.sum() if counts.sum() else np.full(n, 1.0 / n)
    empty = [i for i, row in enumerate(parsed) if not encode(row)]
    P[empty] = marginal / max(np.linalg.norm(marginal), 1e-12)
    return P


def _path(df, cols):
    values = df[list(cols)].astype('string')
    return values.apply(lambda row: '/'.join(row.astype(str)) if row.notna().all() else pd.NA, axis=1)


def build_table(df, cat_cols=(), ord_cols=None, geo_cols=(), num_cols=(), multi_cols=(),
                num_conf_cols=None, text_emb=None, min_conf=0.7, fit_rows=None):
    """Unknown categories -> FIT marginal; numeric -> FIT median + missing indicator.

    multi_cols: names (presence/L2), or {name: 'shares'} for physical blends
    with nonnegative known proportions. OCR probabilities are not blend shares.
    fit_rows: integer row positions used to fit vocabularies, marginals and scaling.
    For known-catalog temporal splits, all catalog rows are available. For a
    new-wine experiment pass ONLY training cuvées. Confidence is NOT variance:
    entries below min_conf are treated as missing. Exclude bad entity matches
    upstream. geo_cols must be coarse -> fine; never repeat country in cat_cols.
    """
    if not len(df):
        raise ValueError("catalog is empty")
    if set(cat_cols) & set(geo_cols):
        raise ValueError("geographic levels must not be duplicated in cat_cols")
    fit_rows = np.arange(len(df)) if fit_rows is None else np.asarray(fit_rows)
    if (fit_rows.ndim != 1 or not np.issubdtype(fit_rows.dtype, np.integer) or
            not len(fit_rows) or np.any((fit_rows < 0) | (fit_rows >= len(df)))):
        raise ValueError("invalid encoder fit rows")
    train = df.iloc[fit_rows]
    t = WineTable(ids=np.asarray(df.index))
    for c in cat_cols:
        t.cat.append(_category(df[c], train[c])); t.cat_names.append(c)
    for c in multi_cols:
        mode = multi_cols[c] if isinstance(multi_cols, dict) else 'presence'
        t.cat.append(_list_factor(df[c], train[c], mode)); t.cat_names.append(c)
    for c, levels in (ord_cols or {}).items():
        if levels is None:
            raw = pd.to_numeric(df[c], errors='coerce')
            known = raw.iloc[fit_rows].dropna()
            if raw.notna().any() and not np.all(raw.dropna().values == raw.dropna().values.astype(int)):
                raise ValueError('ordinal values must be integers')
            offset = int(known.min()) if len(known) else 0
            n = int(known.max()) - offset + 1 if len(known) else 1
            codes = raw.fillna(offset).to_numpy(dtype=int) - offset
            codes[raw.isna().to_numpy() | (codes < 0) | (codes >= n)] = -1
            # Values outside fitted range are unknown, not new graph vertices.
            fit_codes = codes[fit_rows]
            P = onehot(codes, n)
            marginal = np.bincount(fit_codes[fit_codes >= 0], minlength=n)
            marginal = marginal / marginal.sum() if marginal.sum() else np.full(n, 1 / n)
            P[codes < 0] = marginal
        else:
            if not levels:
                raise ValueError('empty ordinal levels')
            codes = df[c].map({v: i for i, v in enumerate(levels)}).fillna(-1).to_numpy(dtype=int)
            fit_codes = codes[fit_rows]
            P = onehot(codes, len(levels))
            marginal = np.bincount(fit_codes[fit_codes >= 0], minlength=len(levels))
            marginal = marginal / marginal.sum() if marginal.sum() else np.full(len(levels), 1 / len(levels))
            P[codes < 0] = marginal
        t.ordv.append(P); t.ord_names.append(c)
    for depth, c in enumerate(geo_cols):
        path = _path(df, geo_cols[:depth + 1])
        t.geo.append(_category(path, path.iloc[fit_rows])); t.geo_names.append(c)
    if num_cols:
        X = df[list(num_cols)].apply(pd.to_numeric, errors='coerce').copy()
        for c, cc in (num_conf_cols or {}).items():
            if c not in num_cols:
                raise ValueError(f'unknown numeric column: {c}')
            conf = pd.to_numeric(df[cc], errors='coerce')
            X.loc[conf.isna() | (conf < min_conf), c] = np.nan
        miss = X.isna().to_numpy(dtype=float)
        fit = X.iloc[fit_rows]
        med = fit.median().fillna(0.0)
        base = fit.fillna(med)
        sd = base.std().fillna(1.0).replace(0, 1.0)
        Z = ((X.fillna(med) - base.mean()) / sd).to_numpy(dtype=float)
        # Always allocate an indicator: even if train has no missing values,
        # new wines may have them. Shape is independent of held-out metadata.
        t.num = np.column_stack([Z, miss])
        t.num_names = list(num_cols) + [f'{c}_missing' for c in num_cols]
        t.num_sx = miss  # legacy diagnostic only; never passed to a kernel
    if text_emb is not None:
        T = np.asarray(text_emb, float)
        if T.ndim != 2 or T.shape[0] != len(df):
            raise ValueError('text embedding dimensions')
        t.text = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    return t.validate()
