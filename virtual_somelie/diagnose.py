"""Controlled additive-vs-product ablation. NOT real-world G2 evidence.

python -m virtual_somelie.diagnose --seeds 0 1 2 --n 300 --train 60 > ablation.json
"""
import argparse
import json
import numpy as np
from .gp import GP
from .synthetic import make_catalog, make_taste
from .table import WineTable


def _subset(t, case):
    """Add one group of nuisance features at a time, without reading ratings."""
    cat = ['type', 'grapes'] + (['producer'] if case in ('producer', 'full') else [])
    ords = ['sweetness'] + (['vintage'] if case == 'full' else [])
    geo = t.geo_names if case in ('geography', 'producer', 'full') else []
    nums = ['tannin', 'body'] + (['abv', 'logprice', 'acidity'] if case == 'full' else [])
    return WineTable(cat=[t.cat[t.cat_names.index(c)] for c in cat], cat_names=cat,
                     ordv=[t.ordv[t.ord_names.index(c)] for c in ords], ord_names=ords,
                     geo=t.geo if geo else [], geo_names=list(geo),
                     num=t.num[:, [t.num_names.index(c) for c in nums]], num_names=nums,
                     text=t.text if case == 'full' else None)


def diagnose(seeds=(0, 1, 2), n=300, train=60, optimize=False):
    if not 3 <= train < n:
        raise ValueError('invalid train size')
    results = []
    for seed in seeds:
        table, hidden = make_catalog(n, seed=seed)
        f, rate = make_taste(hidden, seed=seed + 100, noise=0.05)
        rng = np.random.default_rng(seed)
        tr = rng.choice(n, train, replace=False)
        te = np.setdiff1d(np.arange(n), tr)
        y = rate(tr, rng)
        for case in ('taste', 'geography', 'producer', 'full'):
            features = _subset(table, case)
            for mode in ('additive', 'product') if case != 'full' else ('additive', 'additive_product', 'product'):
                gp = GP(features, mode=mode, seed=seed).fit(tr, y, optimize=optimize)
                gram = gp.kern.gram(gp.p, tr, tr)
                off = gram[np.triu_indices(train, 1)]
                # Remove weak global intercept before measuring factor collapse.
                structured_off = off - 0.05 * gp.p['sf'] ** 2
                eig = np.linalg.eigvalsh(gram)
                mu, var = gp.predict(te)
                results.append(dict(seed=int(seed), case=case, mode=mode, n_train=train,
                                    theta=gp.theta.tolist(), optimizer=gp.optimization_status,
                                    objective=float(gp._nll(gp.theta, gp.I, gp.y, gp.users)),
                                    noise=float(gp.p['sn']), amplitudes=gp.p['amps'].tolist(),
                                    diag=[float(np.min(np.diag(gram))), float(np.max(np.diag(gram)))],
                                    offdiag_quantiles=np.quantile(off, [0, .25, .5, .75, 1]).tolist(),
                                    effective_rank=float(eig.sum() ** 2 / np.square(eig).sum()),
                                    near_zero=float(np.mean(np.abs(structured_off) < 1e-3)),
                                    correlation=float(np.corrcoef(mu, f(te))[0, 1]),
                                    mean_var=float(var.mean()),
                                    coverage90=float(np.mean(np.abs(mu - f(te)) < 1.645 * np.sqrt(var + .05 ** 2)))))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--n', type=int, default=300)
    parser.add_argument('--train', type=int, default=60)
    parser.add_argument('--optimize', action='store_true')
    args = parser.parse_args()
    print(json.dumps(diagnose(args.seeds, args.n, args.train, args.optimize), indent=2))
