"""Synthetic wine catalog + simulated taste for stage 2."""
import numpy as np
from .table import WineTable, onehot, multihot


def make_catalog(N=1500, seed=0):
    r = np.random.default_rng(seed)
    n_type, n_country, n_grape, n_prod = 4, 12, 40, 200
    typ = r.integers(n_type, size=N)
    country = r.integers(n_country, size=N)
    region = country * 5 + r.integers(5, size=N)
    grape = r.integers(n_grape, size=N)
    grape2 = np.where(r.random(N) < 0.4, r.integers(n_grape, size=N), -1)
    prod = r.integers(n_prod, size=N)
    vintage = r.integers(15, size=N)
    sweet = r.integers(5, size=N)
    num = r.standard_normal((N, 5))
    num[:, 0] += 0.5 * typ
    txt = r.standard_normal((N, 8))
    txt /= np.linalg.norm(txt, axis=1, keepdims=True)
    t = WineTable(
        cat=[onehot(typ, n_type), onehot(prod, n_prod),
             multihot([[g] + ([g2] if g2 >= 0 else []) for g, g2 in zip(grape, grape2)], n_grape)],
        cat_names=["type", "producer", "grapes"],
        ordv=[onehot(vintage, 15), onehot(sweet, 5)], ord_names=["vintage", "sweetness"],
        geo=[onehot(country, n_country), onehot(region, n_country * 5)],
        geo_names=["country", "region"],
        num=num, num_names=["abv", "logprice", "acidity", "tannin", "body"], text=txt,
    )
    hidden = dict(typ=typ, grape=grape, sweet=sweet, num=num)
    return t, hidden


def make_taste(hidden, seed=1, noise=0.3):
    r = np.random.default_rng(seed)
    g_eff = r.normal(0, 1.0, size=40)
    t_eff = r.normal(0, 0.7, size=4)
    w = np.zeros(5); w[3] = 1.0; w[4] = 0.8
    center = np.array([0, 0, 0, 1.0, 0.5])
    def f(idx):
        n = hidden["num"][idx]
        pref = -((n - center) ** 2 @ np.array([0, 0, 0, 0.8, 0.5]))
        return 3 + g_eff[hidden["grape"][idx]] + t_eff[hidden["typ"][idx]] + 0.5 * pref
    def rate(idx, rng):
        return f(idx) + noise * rng.standard_normal(np.shape(idx))
    return f, rate
