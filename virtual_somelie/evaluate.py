"""Leakage-free, *conditional on rated wines* offline ranking harness.

No offline counterfactual BO claim is possible without exposure/propensity logs.
Ratings (user, wine_idx, timestamp, rating) are explicit post-tasting feedback.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .gp import GP


def ndcg_at_k(y_true, y_score, k=5):
    """Linear gains max(stars-1, 0); equal gaps are a baseline, not an ordinal axiom."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    if len(y_true) != len(y_score) or k < 1 or not np.isfinite(y_true).all() or not np.isfinite(y_score).all():
        raise ValueError("invalid ranking")
    order = np.argsort(-y_score, kind="stable")[:k]
    gains = np.maximum(y_true - 1, 0)
    disc = 1.0 / np.log2(np.arange(2, len(order) + 2))
    idcg = (np.sort(gains)[::-1][:len(order)] * disc).sum()
    return (gains[order] * disc).sum() / idcg if idcg > 0 else np.nan


def _features(table):
    blocks = [*table.cat, *table.ordv, *table.geo]
    if table.num is not None:
        blocks.append(table.num / 2)
    if table.text is not None:
        blocks.append(table.text)
    return np.hstack(blocks) / np.sqrt(len(blocks))


def score_knn(K, tr_idx, tr_y, te_idx):
    S = np.maximum(K[np.ix_(te_idx, tr_idx)], 0)
    dev = tr_y - tr_y.mean()
    return tr_y.mean() + (S @ dev) / (S.sum(1) + 1e-9)


def score_ridge(X, tr_idx, tr_y, te_idx, prior=3.0, regularizer=1.0):
    """Bayesian linear weights N(0,I), Gaussian observation variance λ (dual form)."""
    tr, te = X[tr_idx], X[te_idx]
    return prior + te @ tr.T @ np.linalg.solve(tr @ tr.T + regularizer * np.eye(len(tr)), tr_y - prior)


def make_manifest(ratings, table, n_train=10, n_users=100, min_test=5, seed=0,
                  protocol="temporal", timestamp="timestamp", groups=None,
                  col_user="user", col_wine="wine_idx", col_rating="rating"):
    """Row positions, not labels, saved for replay. Test ratings are NEVER read for training.

    temporal: prefix/suffix within known users; random: synthetic-only; new_user:
    disjoint training and test users; new_wine: disjoint cuvée groups globally.
    """
    if protocol not in ("temporal", "random", "new_user", "new_wine") or n_train < 1 or min_test < 1:
        raise ValueError("invalid evaluation protocol")
    if protocol != "random" and timestamp not in ratings:
        raise ValueError("real evaluation requires timestamps; random split is synthetic-only")
    if ratings[[col_user, col_wine, col_rating]].isna().any().any():
        raise ValueError("missing user, wine or rating")
    wi = ratings[col_wine].to_numpy()
    if not np.issubdtype(wi.dtype, np.integer) or np.any((wi < 0) | (wi >= len(table))):
        raise ValueError("invalid wine indices")
    if not np.isfinite(ratings[col_rating].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite ratings")
    if ratings.duplicated([col_user, col_wine]).any():
        raise ValueError("aggregate repeated tastings per user/wine before splitting")
    if protocol != "random" and (ratings[timestamp].isna().any() or
                                 pd.to_datetime(ratings[timestamp], errors="coerce").isna().any()):
        raise ValueError("invalid timestamps")
    rng = np.random.default_rng(seed)
    positions = np.arange(len(ratings))
    users = ratings[col_user].to_numpy()
    unique = pd.unique(users)
    if protocol == "new_wine":
        if groups is None or len(groups) != len(table) or pd.isna(groups).any():
            raise ValueError("new_wine requires a cuvée group for every catalog object")
        labels = np.asarray(groups)
        unique_groups = np.unique(labels)
        held_out = set(rng.choice(unique_groups, max(1, len(unique_groups) // 5), replace=False))
        wine_test = np.array([x in held_out for x in labels])
    else:
        wine_test = None
    ordered = {}
    for u in unique:
        ix = positions[users == u]
        if protocol != "random":
            ix = ix[np.argsort(pd.to_datetime(ratings.iloc[ix][timestamp]).to_numpy(), kind="stable")]
        else:
            ix = rng.permutation(ix)
        if wine_test is None:
            tr, te = ix[:n_train], ix[n_train:]
        else:
            tr = np.array([i for i in ix if not wine_test[wi[i]]])[:n_train]
            # A query predating the last support event could not have been
            # recommended with that user's full support history.
            cutoff = pd.to_datetime(ratings.iloc[tr[-1]][timestamp]) if len(tr) else None
            te = np.array([i for i in ix if wine_test[wi[i]] and
                           (cutoff is None or pd.to_datetime(ratings.iloc[i][timestamp]) > cutoff)])
        if len(tr) == n_train and len(te) >= min_test:
            ordered[str(u)] = dict(train=tr.tolist(), test=te.tolist())
    limit = len(ordered) - 1 if protocol == "new_user" else len(ordered)
    selected = rng.choice(list(ordered), size=min(n_users, limit), replace=False) if ordered else []
    folds = {u: ordered[u] for u in selected}
    if protocol == "new_user":
        # Only *other* users' permitted train events contribute to population stats.
        population = [i for u, fold in ordered.items() if u not in folds for i in fold["train"]]
        if not population:
            raise ValueError("new_user needs separate training users")
    else:
        population = [i for fold in ordered.values() for i in fold["train"]]
    return dict(seed=seed, protocol=protocol, n_train=n_train,
                folds=folds, population=population,
                held_out_cuvees=sorted(map(str, held_out)) if wine_test is not None else [])


def save_manifest(manifest, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def fit_population_theta(ratings, table, manifest, col_user="user", col_wine="wine_idx",
                         col_rating="rating", max_events=300, seed=0):
    """Optional global MAP on support events ONLY. Never fit on query events.

    The ICM shared-user model is a training device; recommendation evaluation
    still uses an independent GP per user. Use separate validation/final users
    when selecting hyperparameters or kernel mode.
    """
    rows = np.asarray(manifest["population"], dtype=int)
    if len(rows) < 21:
        raise ValueError("population kernel training needs at least 21 observations")
    if len(rows) > max_events:
        rows = np.random.default_rng(seed).choice(rows, max_events, replace=False)
    gp = GP(table, seed=seed)
    gp.fit(ratings.iloc[rows][col_wine].to_numpy(dtype=int),
           ratings.iloc[rows][col_rating].to_numpy(dtype=float),
           users=ratings.iloc[rows][col_user].to_numpy(), optimize=True)
    return gp.theta.copy()


def _scores(ratings, table, manifest, col_wine, col_rating, X, kern, p, theta):
    wi = ratings[col_wine].to_numpy(dtype=int)
    yy = ratings[col_rating].to_numpy(dtype=float)
    tr_pool = np.asarray(manifest["population"], dtype=int)
    if not len(tr_pool):
        raise ValueError("no population train ratings")
    global_mean = float(yy[tr_pool].mean())
    global_sd = max(float(yy[tr_pool].std()), 1.0)
    counts = np.bincount(wi[tr_pool], minlength=len(table))
    totals = np.bincount(wi[tr_pool], weights=yy[tr_pool], minlength=len(table))
    popularity = (totals + 5 * global_mean) / (counts + 5)
    for user, fold in manifest["folds"].items():
        tr, te = np.asarray(fold["train"], int), np.asarray(fold["test"], int)
        tr_i, te_i, tr_y = wi[tr], wi[te], yy[tr]
        prior = (tr_y.sum() + 5 * global_mean) / (len(tr_y) + 5)
        # Small per-user cross block only, not an N×N Gram for every user.
        cross = kern.gram(p, te_i, tr_i)
        knn = tr_y.mean() + np.maximum(cross, 0) @ (tr_y - tr_y.mean()) / (np.maximum(cross, 0).sum(1) + 1e-9)
        gp = GP(table, theta=theta, mode=kern.mode)
        gp.fit(tr_i, tr_y, optimize=True, mean=global_mean, scale=global_sd)
        yield user, te, dict(popularity=popularity[te_i], user_mean=np.full(len(te), prior),
                             content_kNN=knn,
                             bayesian_linear=score_ridge(X, tr_i, tr_y, te_i, prior),
                             GP_additive=gp.predict(te_i)[0])


def evaluate(ratings, table, n_train_list=(5, 10, 20), n_users=100, min_test=5,
             col_user="user", col_wine="wine_idx", col_rating="rating", seed=0, k=5,
             protocol="temporal", timestamp="timestamp", groups=None,
             return_predictions=False, manifest_dir=None, global_theta=None):
    """Return metrics (and optionally scores for an anti-leakage audit).

    Catalog features must be available at recommendation time. Critic/public
    ratings are NOT allowed as features unless source/time is independently audited.
    """
    table.validate()
    X = _features(table)
    prototype = GP(table, theta=global_theta)
    p = prototype.kern.unpack(prototype.theta)
    out, predictions = [], []
    for n_tr in n_train_list:
        manifest = make_manifest(ratings, table, n_tr, n_users, min_test, seed,
                                 protocol, timestamp, groups, col_user, col_wine, col_rating)
        if manifest_dir is not None:
            from pathlib import Path
            Path(manifest_dir).mkdir(parents=True, exist_ok=True)
            save_manifest(manifest, Path(manifest_dir) / f"{protocol}_{n_tr}_{seed}.json")
        for user, te, scores in _scores(ratings, table, manifest, col_wine, col_rating, X,
                                       prototype.kern, p, prototype.theta):
            te_y = ratings.iloc[te][col_rating].to_numpy(dtype=float)
            for method, s in scores.items():
                rho = spearmanr(s, te_y).statistic if np.ptp(s) > 1e-9 and np.ptp(te_y) > 1e-9 else 0.0
                out.append(dict(n_train=n_tr, user=user, method=method, spearman=rho,
                                ndcg=ndcg_at_k(te_y, s, k), top5_ge4=float(np.mean(te_y[np.argsort(-s)[:5]] >= 4))))
                for pos, value in zip(te, s):
                    predictions.append(dict(n_train=n_tr, user=user, row=int(pos), method=method, score=float(value)))
    results = pd.DataFrame(out)
    return (results, pd.DataFrame(predictions)) if return_predictions else results


def paired_bootstrap(res, challenger="GP_additive", baseline="content_kNN", n_train=10,
                     metric="ndcg", seed=0, repeats=2000):
    """Paired user-level CI. Gate G2 needs >=300 real users and >0.02 lower bound.

    Run on an untouched final holdout ONCE after selecting the kernel/prior on
    validation users. This function does not fix rated-only selection bias.
    """
    if metric not in ("ndcg", "spearman", "top5_ge4") or repeats < 1:
        raise ValueError("unsupported metric or bootstrap count")
    scores = res[res.n_train == n_train].pivot(index="user", columns="method", values=metric)
    diff = (scores[challenger] - scores[baseline]).dropna().to_numpy()
    if not len(diff):
        raise ValueError("no paired users")
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(diff, len(diff), replace=True).mean() for _ in range(repeats)])
    lo, hi = np.quantile(means, [0.025, 0.975])
    return dict(n_users=len(diff), effect=float(diff.mean()), ci95=[float(lo), float(hi)],
                gate_g2=bool(len(diff) >= 300 and lo >= 0.02))


def summarize(res):
    return res.groupby(["n_train", "method"])[["spearman", "ndcg", "top5_ge4"]].mean().round(3)
