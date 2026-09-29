"""Regression checks for the mathematical contract and leakage boundaries."""
import numpy as np
import pandas as pd
import pytest
from scipy.optimize._numdiff import approx_derivative
from .acquisition import expected_improvement, maxmin_init, recommend_batch, thompson
from .build_table import build_table
from .enrich import audit_fields
from .evaluate import evaluate, make_manifest, paired_bootstrap
from .gp import GP, WineKernel
from .kernels import cat_factor
from .ordinal import OrdinalGP, _ll_terms, laplace_fit
from .synthetic import make_catalog
from .table import WineTable, onehot


def test_posterior_vs_independent_reference_and_soft_diagonal():
    t = WineTable(cat=[onehot(np.array([0, -1, 1, 2, -1]), 3)], cat_names=['type'])
    gp = GP(t).fit([0, 2, 3], [1.0, 2.0, 4.0], optimize=False)
    J = np.array([0, 1, 2, 3, 4])
    Ktt = gp.kern.gram(gp.p, gp.I, gp.I)
    Kxt = gp.kern.gram(gp.p, J, gp.I)
    Kxx = gp.kern.gram(gp.p, J, J)
    A = Ktt + gp.p['sn'] ** 2 * np.eye(len(gp.I))
    expected_mu = gp.y_mu + gp.y_sd * Kxt @ np.linalg.solve(A, gp.y)
    expected_cov = gp.y_sd ** 2 * (Kxx - Kxt @ np.linalg.solve(A, Kxt.T))
    mu, cov = gp.predict(J, full_cov=True)
    mu2, var = gp.predict(J)
    assert np.allclose(mu, expected_mu, atol=1e-8)
    assert np.allclose(cov, expected_cov, atol=1e-8)
    assert np.allclose(mu, mu2) and np.allclose(np.diag(cov), var)
    assert np.allclose(gp.kern.diag(gp.p, J), np.diag(Kxx))
    # Soft rows have smaller prior diagonal than hard rows in a delta kernel.
    assert gp.kern.diag(gp.p, [1])[0] < gp.kern.diag(gp.p, [0])[0]


def test_heterogeneous_missing_psd_and_geography_not_product():
    rng = np.random.default_rng(7)
    for _ in range(4):
        x = rng.normal(size=30)
        x[rng.random(30) < .45] = np.nan
        t = build_table(pd.DataFrame(dict(abv=x, type=rng.choice(['red', 'white', None], 30),
                                          country=rng.choice(['A', 'B'], 30),
                                          region=rng.choice(['R', 'S'], 30))),
                        cat_cols=['type'], geo_cols=['country', 'region'], num_cols=['abv'])
        k = WineKernel(t)
        p = k.unpack(k.default_theta())
        gram = k.gram(p, np.arange(30), np.arange(30))
        assert np.allclose(gram, gram.T)
        assert np.linalg.eigvalsh(gram).min() > -1e-9
        assert np.allclose(np.diag(gram), k.diag(p, np.arange(30)))
    with pytest.raises(ValueError, match='duplicated'):
        build_table(pd.DataFrame({'country': ['A']}), cat_cols=['country'], geo_cols=['country'])
    # All-missing numeric fields are explicit indicators, not NaN kernel inputs.
    assert np.isfinite(build_table(pd.DataFrame({'abv': [None, None]}), num_cols=['abv']).num).all()
    assert np.allclose(cat_factor(onehot([0, 1], 2), onehot([0, 1], 2), 1), 1)


def test_ordinal_integer_missing_and_full_path():
    df = pd.DataFrame({'level': [3, None, 5], 'country': ['A', 'B', 'C'],
                       'region': ['R', 'R', 'R'], 'sub': ['X', 'X', 'X']})
    t = build_table(df, ord_cols={'level': None}, geo_cols=['country', 'region', 'sub'])
    assert t.ordv[0].dtype.kind == 'f'
    assert np.allclose(t.ordv[0][1].sum(), 1)
    assert t.ordv[0].shape[1] == 3
    assert np.allclose(t.geo[-1], np.eye(3))
    with pytest.raises(ValueError):
        build_table(pd.DataFrame({'level': [1.5]}), ord_cols={'level': None})


def test_frozen_encoder_and_blend_shares():
    df = pd.DataFrame({'type': ['red', 'red', 'new'], 'abv': [12.0, 14.0, 99.0],
                       'grapes': [{'A': .25, 'B': .75}, {'A': 1.0}, {'Z': 1.0}]})
    t = build_table(df, cat_cols=['type'], num_cols=['abv'], multi_cols={'grapes': 'shares'},
                    fit_rows=[0, 1])
    assert np.allclose(t.cat[1].sum(1), 1)
    assert np.allclose(t.cat[1][2], [0.625, 0.375])  # trained marginal A/B
    assert np.allclose(t.cat[0][2], [1.0])  # unknown category -> FIT marginal
    assert np.allclose(t.num[:2, 0], [-.707106781, .707106781])
    altered = df.copy()
    altered.loc[2, ['type', 'abv', 'grapes']] = ['other', -123., {'Y': 1.0}]
    t2 = build_table(altered, cat_cols=['type'], num_cols=['abv'],
                     multi_cols={'grapes': 'shares'}, fit_rows=[0, 1])
    assert np.allclose(t.cat[0][:2], t2.cat[0][:2])
    assert np.allclose(t.cat[1][:2], t2.cat[1][:2])
    assert np.allclose(t.num[:2], t2.num[:2])


def test_multiuser_cross_contract():
    t = WineTable(cat=[onehot([0, 0, 1, 1], 2)], cat_names=['type'])
    gp = GP(t).fit([0, 1], [3., 5.], users=[10, 20], optimize=False, mean=0, scale=1)
    with pytest.raises(ValueError, match='specify'):
        gp.predict([0, 1])
    with pytest.raises(ValueError, match='unknown'):
        gp.predict([0], user=999)
    base = gp.kern.gram(gp.p, [0], gp.I)
    assert np.allclose(gp._cross([0], 10), base * [1, gp.p['omega']])
    assert np.allclose(gp._cross([0], kind='new'), base * gp.p['omega'])
    assert np.allclose(gp._cross([0], kind='population'), base * gp.p['omega'])
    _, vc = gp.predict([2], user=10)
    _, vn = gp.predict_new_user([2])
    _, vp = gp.predict_population([2])
    assert vp[0] <= vn[0] and np.all(np.isfinite([vc[0], vn[0], vp[0]]))


def test_ordinal_derivatives_and_extreme_tails():
    b = np.array([-1., 0., 1., 2.])
    for y in range(5):
        for f in [-10., -3., 0., 3., 10.]:
            z = np.array([f])
            logp, g, h = _ll_terms(z, np.array([y]), b)
            assert np.isfinite([logp[0], g[0], h[0]]).all()
            assert np.allclose(g[0], approx_derivative(lambda x: _ll_terms(x, np.array([y]), b)[0], z).ravel()[0], rtol=3e-5, atol=1e-5)
            if abs(f) <= 3:
                assert np.allclose(h[0], approx_derivative(lambda x: _ll_terms(x, np.array([y]), b)[1], z).ravel()[0], rtol=5e-4, atol=1e-5)
    K = np.eye(4) * 1.3 + .1
    fit = laplace_fit(K, np.array([0, 1, 3, 4]), b)
    assert all(np.isfinite(x).all() for x in fit)
    t = WineTable(cat=[onehot([0, 1, 1, 0, 1], 2)], cat_names=['a'])
    model = OrdinalGP(GP(t)).fit([0, 1, 2, 3], np.array([1, 2, 5, 4]), prefit=False)
    P, _, var = model.predict_proba(np.arange(5))
    assert np.allclose(P.sum(axis=1), 1) and (var >= 0).all()
    assert np.allclose(model.probability_at_least([4]), P[4, 3:].sum())
    with pytest.raises(ValueError):
        model.fit([0], [0])
    with pytest.raises(ValueError):
        OrdinalGP(GP(t), 1)


def test_short_history_optimizes_only_one_local_scale():
    t, _ = make_catalog(40, seed=1)
    for n in (2, 8, 20):
        gp = GP(t)
        original = gp.theta.copy()
        gp.fit(np.arange(n), np.arange(n, dtype=float) % 5 + 1, optimize=True)
        changed = np.flatnonzero(~np.isclose(gp.theta, original))
        scale_position = gp.kern.M + gp.kern.R + gp.kern.d
        if n < 5:
            assert len(changed) == 0
        else:
            assert set(changed) <= {scale_position}


def test_acquisition_limits_and_no_mutation():
    assert np.allclose(expected_improvement(np.array([0., 1.]), np.zeros(2), 0.5, xi=0), [0, .5])
    K = np.ones((5, 5))
    assert len(set(maxmin_init(K, 5, first=0))) == 5
    assert maxmin_init(K, 0) == []
    with pytest.raises(ValueError):
        maxmin_init(K, 6)
    t, _ = make_catalog(12, seed=4)
    gp = GP(t).fit([0, 1, 2], [2, 3, 4], optimize=False)
    theta, alpha, L, I = gp.theta.copy(), gp.alpha.copy(), gp.L.copy(), gp.I.copy()
    got = recommend_batch(gp, [0, 1, 2], [2, 3, 4], [0, 3, 3, 4, 5, 6], k=3)
    assert len(set(got)) == 3 and not set(got) & {0, 1, 2}
    assert np.array_equal(gp.theta, theta) and np.array_equal(gp.alpha, alpha)
    assert np.array_equal(gp.L, L) and np.array_equal(gp.I, I)
    assert np.array_equal(thompson(gp, [3, 4], rng=np.random.default_rng(4)),
                          thompson(gp, [3, 4], rng=np.random.default_rng(4)))
    with pytest.raises(ValueError):
        recommend_batch(gp, [0, 1, 2], [2, 3, 4], [3], k=2)


def test_evaluation_anti_leakage_and_manifest(tmp_path):
    t, _ = make_catalog(32, seed=2)
    rows = [dict(user=u, wine_idx=int(i), timestamp=str(pd.Timestamp('2024-01-01') + pd.Timedelta(days=j)),
                 rating=float((i + u) % 5 + 1))
            for u in range(6) for j, i in enumerate(np.roll(np.arange(32), u)[:20])]
    ratings = pd.DataFrame(rows)
    manifest = make_manifest(ratings, t, n_train=8, min_test=4, seed=42)
    for fold in manifest['folds'].values():
        assert set(fold['test']).isdisjoint(manifest['population'])
        assert max(pd.to_datetime(ratings.iloc[fold['train']].timestamp)) < min(pd.to_datetime(ratings.iloc[fold['test']].timestamp))
    res, preds = evaluate(ratings, t, n_train_list=(8,), min_test=4, seed=42, n_users=6,
                          return_predictions=True, manifest_dir=tmp_path)
    changed = ratings.copy()
    test_rows = [row for fold in manifest['folds'].values() for row in fold['test']]
    changed.loc[test_rows, 'rating'] = 1
    res2, preds2 = evaluate(changed, t, n_train_list=(8,), min_test=4, seed=42,
                            n_users=6, return_predictions=True)
    pd.testing.assert_frame_equal(preds, preds2)
    assert not res.equals(res2)
    assert len(list(tmp_path.glob('*.json'))) == 1
    assert not paired_bootstrap(res, repeats=20, n_train=8)['gate_g2']
    with pytest.raises(ValueError, match='timestamps'):
        make_manifest(ratings.drop(columns=['timestamp']), t)
    # New-wine split prevents different vintages of the same cuvee in both sets.
    groups = np.arange(len(t)) // 2
    mw = make_manifest(ratings, t, n_train=3, min_test=1, protocol='new_wine', groups=groups)
    assert mw['folds']
    for fold in mw['folds'].values():
        a = set(groups[ratings.iloc[fold['train']].wine_idx])
        b = set(groups[ratings.iloc[fold['test']].wine_idx])
        assert not a & b
    nu = make_manifest(ratings, t, n_train=8, min_test=4, seed=0, protocol='new_user', n_users=2)
    assert len(nu['folds']) == 2 and len(nu['population']) > 0
    for fold in nu['folds'].values():
        assert not set(fold['train'] + fold['test']) & set(nu['population'])


def test_manual_audit_not_fake_variance():
    labels = pd.DataFrame({'field': ['abv', 'abv', 'type'], 'source': ['OCR'] * 3,
                           'predicted': [13., 14., 'red'], 'truth': [13.2, 15., 'red'],
                           'confidence': [.8, .5, .9]})
    report = audit_fields(labels)
    assert report.loc[report.field == 'abv', 'mean_absolute_error'].iloc[0] == pytest.approx(.6)
