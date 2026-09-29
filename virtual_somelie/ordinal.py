"""Ordinal probit with fixed unit probit noise and zero latent mean.

Gaussian GP prefit is optional and only initializes kernel hyperparameters; Laplace
optimizes the latent mode and threshold evidence. It is not exact marginalization.
"""
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import norm
from .gp import GP


def _ll_terms(f, y, b):
    """Per-item log likelihood, gradient and Hessian in f, with stable tail CDFs."""
    edges = np.concatenate(([-np.inf], b, [np.inf]))
    u, l = edges[y + 1] - f, edges[y] - f
    # On the positive tail compute survival-function differences, not 1 - CDF.
    hi, lo = np.where(l > 0, -l, u), np.where(l > 0, -u, l)
    a, c = log_ndtr(hi), log_ndtr(lo)
    logz = a + np.log(-np.expm1(np.minimum(c - a, 0.0)))
    ru = np.exp(norm.logpdf(u) - logz)
    rl = np.exp(norm.logpdf(l) - logz)
    g = rl - ru
    h = -(np.where(np.isfinite(u), u, 0.0) * ru -
          np.where(np.isfinite(l), l, 0.0) * rl) - g ** 2
    return logz, g, h


def laplace_fit(K, y, b, iters=80):
    """Return mode, derivative, W, Cholesky(B), approximate log evidence.

    Raises if Newton/line search does not converge. All returned quantities are
    recomputed at the final mode, not mixed between consecutive iterations.
    """
    y = np.asarray(y, int)
    K = np.asarray(K, float)
    n = len(y)
    if K.shape != (n, n) or not n or not np.isfinite(K).all():
        raise ValueError("invalid ordinal covariance")
    LK = cholesky(K, lower=True)
    Kinv = cho_solve((LK, True), np.eye(n))
    f = np.zeros(n)

    def objective(x):
        return -0.5 * x @ Kinv @ x + _ll_terms(x, y, b)[0].sum()

    for _ in range(iters):
        logp, g, h = _ll_terms(f, y, b)
        if not np.isfinite(logp).all():
            raise RuntimeError("nonfinite ordinal log likelihood")
        W = np.maximum(-h, 1e-10)
        step = np.linalg.solve(Kinv + np.diag(W), g - Kinv @ f)
        old = objective(f)
        if np.max(np.abs(step)) < 1e-7:
            break
        scale = 1.0
        for _ in range(30):
            trial = f + scale * step
            if objective(trial) >= old - 1e-11:
                f = trial
                break
            scale /= 2
        else:
            raise RuntimeError("ordinal Laplace line search failed")
        if abs(objective(f) - old) < 1e-9 and np.max(np.abs(scale * step)) < 1e-5:
            break
    else:
        raise RuntimeError("ordinal Laplace did not converge")
    logp, g, h = _ll_terms(f, y, b)
    W = np.maximum(-h, 1e-10)
    sW = np.sqrt(W)
    B = np.eye(n) + sW[:, None] * K * sW[None, :]
    L = cholesky(B, lower=True)
    logq = -0.5 * f @ Kinv @ f + logp.sum() - np.log(np.diag(L)).sum()
    return f, g, W, L, logq


class OrdinalGP:
    def __init__(self, gp: GP, n_levels=5):
        if n_levels < 2:
            raise ValueError("ordinal scale requires at least two levels")
        self.gp, self.C = gp, n_levels
        self.fitted = False

    def _thr(self, z):
        return np.concatenate(([z[0]], z[0] + np.cumsum(np.exp(z[1:]))))

    def fit(self, I, ratings, users=None, prefit=True):
        ratings = np.asarray(ratings)
        if ratings.ndim != 1 or not len(ratings) or not np.isfinite(ratings.astype(float)).all() or not np.all(
                ratings == ratings.astype(int)) or np.any((ratings < 1) | (ratings > self.C)):
            raise ValueError("ratings must be integer stars 1..C")
        y = ratings.astype(int) - 1
        self.gp.fit(I, y.astype(float), users=users, optimize=prefit)
        gp = self.gp
        K = gp._K_obs(gp.p, gp.I, gp.users) + 1e-8 * np.eye(len(y))
        self.K, self.y = K, y
        # Approximate class-quantile thresholds; fixed zero latent mean, probit sigma=1.
        counts = np.bincount(y, minlength=self.C)
        q = (np.cumsum(counts)[:-1] + 0.5) / (len(y) + 1.0)
        b0 = np.maximum.accumulate(norm.ppf(q) * np.sqrt(1 + np.mean(np.diag(K))))
        b0 += np.arange(self.C - 1) * 1e-3
        z0 = np.r_[b0[0], np.log(np.maximum(np.diff(b0), 0.1))]
        def objective(z):
            try:
                # Explicit weak prior on threshold coordinates; limits empty-class collapse.
                return -laplace_fit(K, y, self._thr(z))[-1] + 0.01 * np.sum((z - z0) ** 2)
            except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError):
                return 1e12
        result = minimize(objective, z0, method="L-BFGS-B",
                          bounds=[(-10, 10)] + [(-4, 3)] * (self.C - 2),
                          options={"maxiter": 80})
        self.threshold_status = result.message
        z = result.x if np.isfinite(result.fun) and result.fun <= objective(z0) else z0
        self.b = self._thr(z)
        self.f, self.g, self.W, self.L, self.log_evidence = laplace_fit(K, y, self.b)
        self.fitted = True
        return self

    def predict_proba(self, J, user=None):
        if not self.fitted:
            raise ValueError("fit the ordinal GP first")
        J = np.asarray(J)
        Ks = self.gp._cross(J, user)
        mu = Ks @ self.g
        sW = np.sqrt(self.W)
        V = solve_triangular(self.L, sW[:, None] * Ks.T, lower=True)
        prior = self.gp.kern.diag(self.gp.p, J)
        var = prior - (V ** 2).sum(0)
        if len(J) and var.min() < -1e-6 * max(1, prior.max()):
            raise np.linalg.LinAlgError("negative ordinal posterior variance")
        var = np.maximum(var, 0)
        edges = np.concatenate(([-np.inf], self.b, [np.inf]))
        cdf = norm.cdf((edges[None, :] - mu[:, None]) / np.sqrt(1 + var[:, None]))
        P = np.maximum(np.diff(cdf, axis=1), 0)
        return P, mu, var

    def probability_at_least(self, J, level=4, user=None):
        if not 1 <= level <= self.C:
            raise ValueError("level out of range")
        P, _, _ = self.predict_proba(J, user=user)
        return P[:, level - 1:].sum(axis=1)

    def expected_rating(self, J, user=None):
        P, _, _ = self.predict_proba(J, user=user)
        return P @ np.arange(1, self.C + 1)
