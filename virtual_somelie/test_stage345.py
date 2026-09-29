"""Ordinal, multi-user and evaluation regression tests."""
import numpy as np, pandas as pd
from .synthetic import make_catalog, make_taste
from .gp import GP
from .ordinal import OrdinalGP
from .enrich import (cluster_labels, Field, Enricher, to_frame,
                     filter_for_training, resolve_entity, quality_report)
from .build_table import build_table
from .evaluate import evaluate, summarize


def test_ordinal_laplace():
    t, h = make_catalog(300, seed=5)
    f, rate = make_taste(h, noise=0.2)
    rng = np.random.default_rng(0)
    tr = rng.choice(300, 60, replace=False); te = np.setdiff1d(np.arange(300), tr)
    yc = np.clip(np.round(rate(tr, rng)), 1, 5).astype(int)
    m = OrdinalGP(GP(t, n_restarts=0), 5).fit(tr, yc)
    P, mu, var = m.predict_proba(te)
    assert np.allclose(P.sum(1), 1) and (var > 0).all()
    r = np.corrcoef(m.expected_rating(te), f(te))[0, 1]
    assert r > 0.4, r


def test_multiuser_icm():
    t, h = make_catalog(300, seed=6)
    f1, r1 = make_taste(h, seed=1, noise=0.2); f2, r2 = make_taste(h, seed=1, noise=0.2)
    rng = np.random.default_rng(0)
    a = rng.choice(300, 8, replace=False); b = rng.choice(300, 40, replace=False)
    I = np.concatenate([a, b]); u = np.array([0] * 8 + [1] * 40)
    y = np.concatenate([r1(a, rng), r2(b, rng)])
    gp = GP(t, n_restarts=0).fit(I, y, users=u)
    te = np.setdiff1d(np.arange(300), a)
    mu, _ = gp.predict(te, user=0)
    r = np.corrcoef(mu, f1(te))[0, 1]
    solo = GP(t, n_restarts=0).fit(a, r1(a, np.random.default_rng(0))).predict(te)[0]
    r_solo = np.corrcoef(solo, f1(te))[0, 1]
    assert r > r_solo, (r, r_solo)


def test_pipeline_mock():
    recs = [dict(producer="Chateau Test", wine_name="Grand Vin", vintage=2018, grapes=["Merlot"]),
            dict(producer="Château Test", wine_name="Grand Vin", vintage=2018, grapes=["Merlot"]),
            dict(producer="Domaine X", wine_name="Blanc", vintage=2020, grapes=["Chardonnay"])]
    lab = cluster_labels(recs)
    assert lab[0] == lab[1] != lab[2]
    lk = lambda r: {"abv": Field(13.5, "mock", 0.9), "type": Field("red", "mock", 0.8)}
    lk2 = lambda r: {"abv": Field(14.0, "web", 0.6)}
    e = Enricher([lk, lk2]).enrich(recs[0])
    assert e["abv"].value == 13.5
    df = to_frame(recs, [e, e, e], [0.9, 0.9, 0.5])
    assert len(filter_for_training(df)) == 2
    tab = build_table(df, cat_cols=["type"], multi_cols=["grapes"], num_cols=["abv"],
                      num_conf_cols={"abv": "abv_conf"})
    assert len(tab) == 3 and tab.num_sx.shape == (3, 1)
    q, c = resolve_entity(recs[0], [dict(producer="Chateau Test", wine_name="Grand Vin", vintage=2019)])
    assert len(quality_report(df)) > 0


def test_eval_harness_on_synthetic():
    t, h = make_catalog(200, seed=7)
    rows = []
    for u in range(6):
        f, rate = make_taste(h, seed=50 + u, noise=0.3); rng = np.random.default_rng(u)
        idx = rng.choice(200, 40, replace=False)
        rows += [dict(user=u, wine_idx=int(i), rating=float(v), timestamp=str(pd.Timestamp('2024-01-01') + pd.Timedelta(days=j)))
                 for j, (i, v) in enumerate(zip(idx, rate(idx, rng)))]
    res = evaluate(pd.DataFrame(rows), t, n_train_list=(10,), n_users=6, min_test=10)
    assert not res.empty
    assert len(summarize(res)) > 0


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            print("OK ", n, fn())
