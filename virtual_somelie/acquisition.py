"""Finite-catalog acquisition; exploration is for simulations/pilots, not offline proof."""
import numpy as np
from scipy.stats import norm


def expected_improvement(mu, var, best, xi=0.01):
    """Gaussian EI relative to a *posterior mean* incumbent, not max noisy y."""
    mu, var = np.asarray(mu), np.asarray(var)
    if np.any(var < 0):
        raise ValueError("negative variance")
    s = np.sqrt(var)
    z = (mu - best - xi) / np.where(s > 0, s, 1)
    return np.where(s > 0, (mu - best - xi) * norm.cdf(z) + s * norm.pdf(z),
                    np.maximum(mu - best - xi, 0))


def ucb(mu, var, kappa=1.0):
    if kappa < 0 or np.any(np.asarray(var) < 0):
        raise ValueError("invalid UCB parameters")
    return mu + kappa * np.sqrt(var)


def thompson(gp, cand, user=None, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    cand = np.asarray(cand)
    if not len(cand):
        return np.array([])
    mu, cov = gp.predict(cand, user=user, full_cov=True)
    # Reject materially non-PSD posteriors; add only numerical jitter.
    w, U = np.linalg.eigh(cov)
    if min(w) < -1e-6 * max(1.0, max(w)):
        raise np.linalg.LinAlgError("Thompson requires PSD covariance")
    return mu + U @ (np.sqrt(np.maximum(w, 0)) * rng.standard_normal(len(cand)))


def maxmin_init(Kfull, k, first=None, rng=None):
    """Select unique objects by kernel distance; accept a matrix OR (diag, row) callable.

    A callable accepts an index and returns a kernel row; its .diag must be a
    vector of prior variances. No N×N allocation is required for this form.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    if callable(Kfull):
        diag = np.asarray(Kfull.diag)
        getrow = Kfull
    else:
        Kfull = np.asarray(Kfull)
        if Kfull.ndim != 2 or Kfull.shape[0] != Kfull.shape[1]:
            raise ValueError("kernel must be square")
        diag = np.diag(Kfull)
        getrow = lambda j: Kfull[j]
    N = len(diag)
    if not 0 <= k <= N or (first is not None and (not 0 <= first < N or k == 0)) or np.any(diag <= 0):
        raise ValueError("invalid k, first or kernel diagonal")
    if not k:
        return []
    sel = [int(rng.integers(N)) if first is None else int(first)]
    def distance(j):
        similarity = getrow(j) / np.sqrt(diag[j] * diag)
        return np.sqrt(np.maximum(2 - 2 * similarity, 0))
    dmin = distance(sel[0]); dmin[sel[0]] = -np.inf
    while len(sel) < k:
        j = int(np.argmax(dmin))
        sel.append(j)
        dmin = np.minimum(dmin, distance(j))
        dmin[sel] = -np.inf
    return sel


def recommend_batch(gp, I_obs, y_obs, candidates, k=5, kappa=1.0, user=None, users_obs=None,
                    available=None, budget_mask=None):
    """Kriging believer on a *frozen* GP posterior (no mutation/re-normalization).

    Observation noise of each fantasy is the fitted GP noise on the original
    rating scale. Availability/budget filters and observed wines are excluded
    before sampling. This is a heuristic, not integration over unknown ratings.
    """
    if not gp.fitted:
        raise ValueError("fit the GP first")
    I_obs, y_obs = list(I_obs), list(y_obs)
    fitted_y = gp.y_mu + gp.y_sd * gp.y
    if (len(I_obs) != len(y_obs) or not np.array_equal(np.asarray(I_obs), gp.I) or
            not np.allclose(np.asarray(y_obs, float), fitted_y)):
        raise ValueError("observations must match fitted GP")
    if gp.users is not None and users_obs is None:
        raise ValueError("users_obs required for a multi-user GP")
    if users_obs is not None and (gp.users is None or not np.array_equal(np.asarray(users_obs), gp.users)):
        raise ValueError("users_obs must match fitted GP")
    cand = np.asarray(candidates)
    if cand.ndim != 1 or not np.issubdtype(cand.dtype, np.integer) or np.any((cand < 0) | (cand >= len(gp.kern.t))):
        raise ValueError("invalid candidates")
    cand = np.array(list(dict.fromkeys(int(x) for x in cand if x not in I_obs)), dtype=int)
    for mask in (available, budget_mask):
        if mask is not None:
            mask = np.asarray(mask, bool)
            if mask.shape != (len(gp.kern.t),):
                raise ValueError("filter must cover the whole catalog")
            cand = cand[mask[cand]]
    if not 0 <= k <= len(cand):
        raise ValueError("k exceeds eligible candidates")
    if k == 0:
        return []
    mu, cov = gp.predict(cand, user=user, full_cov=True)
    cov = cov.copy()
    noise = (gp.p["sn"] * gp.y_sd) ** 2
    chosen, eligible = [], np.ones(len(cand), dtype=bool)
    for _ in range(k):
        scores = ucb(mu, np.maximum(np.diag(cov), 0), kappa)
        scores[~eligible] = -np.inf
        j = int(np.argmax(scores))
        chosen.append(int(cand[j])); eligible[j] = False
        column = cov[:, j].copy()
        denom = cov[j, j] + noise
        cov -= np.outer(column, column) / denom
        cov = (cov + cov.T) / 2
        # Fantasy observation equals mu[j]: posterior means stay unchanged.
    return chosen
