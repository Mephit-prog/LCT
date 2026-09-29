"""Exact GP on a finite catalog; additive main effects are the default.

Hyperparameters are fixed for a short user history. Fit global parameters on TRAIN
users only; calling fit on the same instance for another user is not an evaluation API.
"""
import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize, minimize_scalar
from .kernels import (sigmoid, softplus, path_laplacian, diffusion_gram,
                      normalize_gram, cat_factor, graph_factor, matern52)


class WineKernel:
    def __init__(self, table, mode="additive"):
        table.validate()
        if mode not in ("additive", "additive_product", "product"):
            raise ValueError("invalid kernel mode")
        self.t, self.mode = table, mode
        self.M, self.R, self.L, self.d = len(table.cat), len(table.ordv), len(table.geo), table.d
        self.has_text = table.text is not None
        self.Lap = [path_laplacian(P.shape[1]) for P in table.ordv]
        self.F = self.M + self.R + self.L + (self.d > 0) + self.has_text
        if not self.F:
            raise ValueError("catalog needs at least one feature")
        # tau, beta, lengthscales, signal/noise, user shared fraction, main-effect
        # log-amplitudes, product log-amplitude. All are unconstrained log coordinates.
        self.n_par = self.M + self.R + self.d + 3 + self.F + 1

    def default_theta(self):
        return np.array([0.0] * self.M + [-0.5] * self.R +
                        [np.log(1.5)] * self.d + [0.0, np.log(0.3), 0.0] +
                        [-0.5 * np.log(self.F)] * self.F + [-1.0])

    def unpack(self, th):
        th = np.asarray(th, float)
        if th.shape != (self.n_par,) or not np.isfinite(th).all():
            raise ValueError("invalid kernel parameters")
        i = 0
        p = {}
        p["tau"] = sigmoid(th[i:i + self.M]); i += self.M
        p["beta"] = softplus(th[i:i + self.R]); i += self.R
        p["ell"] = np.exp(th[i:i + self.d]); i += self.d
        p["sf"], p["sn"], p["omega"] = np.exp(th[i]), np.exp(th[i + 1]), sigmoid(th[i + 2]); i += 3
        p["amps"] = np.exp(th[i:i + self.F]); i += self.F
        p["prod"] = np.exp(th[i])
        p["Kp"] = [normalize_gram(diffusion_gram(L, b)) for L, b in zip(self.Lap, p["beta"])]
        return p

    def _factors(self, p, I, J):
        t = self.t
        factors = [cat_factor(P[I], P[J], tau) for P, tau in zip(t.cat, p["tau"])]
        factors += [graph_factor(P[I], P[J], Kp) for P, Kp in zip(t.ordv, p["Kp"])]
        # Geography is additive: different countries still share the global taste kernel.
        factors += [P[I] @ P[J].T for P in t.geo]
        if self.d:
            factors.append(matern52(t.num[I], t.num[J], p["ell"]))
        if self.has_text:
            factors.append(t.text[I] @ t.text[J].T)
        return factors

    def gram(self, p, I, J):
        I, J = np.asarray(I), np.asarray(J)
        factors = self._factors(p, I, J)
        K = np.zeros((len(I), len(J)))
        if self.mode != "product":
            for a, block in zip(p["amps"], factors):
                K += a ** 2 * block
        if self.mode != "additive":
            product = np.ones_like(K)
            for block in factors:
                product *= block
            K += (p["prod"] if self.mode == "additive_product" else 1.0) ** 2 * product
        return p["sf"] ** 2 * (K + 0.05)  # weak global intercept

    def diag(self, p, idx):
        """O(n d), not diag(gram(idx,idx)). Important for soft/unknown categories."""
        idx = np.asarray(idx)
        t = self.t
        parts = [(1 - tau) * (P[idx] ** 2).sum(1) + tau * P[idx].sum(1) ** 2
                 for P, tau in zip(t.cat, p["tau"])]
        parts += [np.sum((P[idx] @ Kp) * P[idx], axis=1) for P, Kp in zip(t.ordv, p["Kp"])]
        parts += [(P[idx] ** 2).sum(1) for P in t.geo]
        if self.d:
            parts.append(np.ones(len(idx)))
        if self.has_text:
            parts.append((t.text[idx] ** 2).sum(1))
        result = np.full(len(idx), 0.05)
        if self.mode != "product":
            for a, block in zip(p["amps"], parts):
                result += a ** 2 * block
        if self.mode != "additive":
            product = np.prod(parts, axis=0)
            result += (p["prod"] if self.mode == "additive_product" else 1.0) ** 2 * product
        return p["sf"] ** 2 * result

    def prior_nlp(self, th, p):
        """Explicit Gaussian prior on *logit/log coordinates*, not a Horseshoe.

        MAP is with respect to Lebesgue measure in th. No implicit Jacobian or
        claim of parametrization invariance. Local log-amplitudes shrink to the
        shared defaults; tune this prior on training users only.
        """
        delta = th - self.default_theta()
        scales = np.array([1.5] * self.M + [1.0] * self.R + [0.8] * self.d +
                          [0.5, 0.5, 1.0] + [0.6] * self.F + [0.6])
        return 0.5 * np.sum((delta / scales) ** 2)


def _factor(K):
    """Bounded adaptive jitter; large negative eigenvalues are errors, not variances."""
    scale = max(float(np.max(np.diag(K))), 1e-12)
    for rel in (0.0, 1e-10, 1e-8, 1e-6, 1e-5):
        try:
            return cholesky(K + rel * scale * np.eye(len(K)), lower=True), rel * scale
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("kernel not PSD at permitted jitter (1e-5 of diagonal)")


class GP:
    def __init__(self, table, n_restarts=0, seed=0, mode="additive", theta=None):
        self.kern = WineKernel(table, mode)
        self.n_restarts = n_restarts
        self.rng = np.random.default_rng(seed)
        self.theta = self.kern.default_theta() if theta is None else np.array(theta, float, copy=True)
        self.kern.unpack(self.theta)
        self.fitted = False

    def _K_obs(self, p, I, users):
        K = self.kern.gram(p, I, I)
        if users is not None:
            same = (users[:, None] == users[None, :]).astype(float)
            K = K * (p["omega"] + (1 - p["omega"]) * same)
        return K

    def _nll(self, th, I, y, users):
        p = self.kern.unpack(th)
        K = self._K_obs(p, I, users) + p["sn"] ** 2 * np.eye(len(I))
        try:
            L, _ = _factor(K)
        except np.linalg.LinAlgError:
            return 1e10
        a = cho_solve((L, True), y)
        return 0.5 * y @ a + np.log(np.diag(L)).sum() + self.kern.prior_nlp(th, p)

    def fit(self, I, y, users=None, optimize=True, mean=None, scale=None):
        I, y = np.asarray(I), np.asarray(y, float)
        if I.ndim != 1 or not np.issubdtype(I.dtype, np.integer) or np.any((I < 0) | (I >= len(self.kern.t))):
            raise ValueError("invalid wine indices")
        if y.shape != I.shape or not np.isfinite(y).all() or len(y) == 0:
            raise ValueError("nonempty aligned finite ratings required")
        if users is not None:
            users = np.asarray(users)
            if users.shape != I.shape or pd_isna_any(users):
                raise ValueError("invalid user identifiers")
        # Freeze normalization for the entire posterior; fantasies never call fit().
        self.I, self.users = I.copy(), None if users is None else users.copy()
        self.y_mu = float(np.mean(y) if mean is None else mean)
        self.y_sd = float((y.std() if len(y) > 2 and y.std() > 1e-6 else 1.0) if scale is None else scale)
        if self.y_sd <= 0 or not np.isfinite(self.y_sd):
            raise ValueError("invalid rating scale")
        self.y = (y - self.y_mu) / self.y_sd
        self.optimization_status = "fixed (1-4 ratings or optimize=False)"
        if optimize and 5 <= len(I) <= 20:
            # Only ONE local signal-scale parameter; global structure, ARD and
            # noise stay fixed. No individual 20-dimensional search at n=5.
            j = self.kern.M + self.kern.R + self.kern.d
            base = self.theta.copy()
            def score(z):
                th = base.copy(); th[j] = z
                return self._nll(th, I, self.y, users)
            result = minimize_scalar(score, bounds=(-3, 3), method='bounded')
            if result.success and result.fun < score(base[j]):
                self.theta[j] = result.x
                self.optimization_status = f"local scale: {result.message}"
        elif optimize and len(I) > 20:
            bounds = [(-5, 5)] * self.kern.n_par
            best = (self._nll(self.theta, I, self.y, users), self.theta.copy())
            starts = [self.theta] + [self.theta + 0.3 * self.rng.standard_normal(self.kern.n_par)
                                     for _ in range(self.n_restarts)]
            for s in starts:
                r = minimize(self._nll, np.clip(s, -5, 5), args=(I, self.y, users),
                             method="L-BFGS-B", bounds=bounds, options={"maxiter": 50})
                if np.isfinite(r.fun) and r.fun < best[0]:
                    best = (r.fun, r.x.copy())
                    self.optimization_status = f"{r.success}: {r.message}"
            self.theta = best[1]
        self.p = self.kern.unpack(self.theta)
        K = self._K_obs(self.p, I, users) + self.p["sn"] ** 2 * np.eye(len(I))
        self.L, self.jitter = _factor(K)
        self.alpha = cho_solve((self.L, True), self.y)
        self.fitted = True
        return self

    def _cross(self, J, user=None, kind="known"):
        if not self.fitted:
            raise ValueError("fit the GP first")
        Ks = self.kern.gram(self.p, np.asarray(J), self.I)
        if self.users is not None:
            if kind in ("population", "new"):
                Ks *= self.p["omega"]
            elif user is None:
                raise ValueError("specify a known user, or use predict_new_user / predict_population")
            elif user not in self.users:
                raise ValueError("unknown user: use predict_new_user")
            else:
                Ks *= self.p["omega"] + (1 - self.p["omega"]) * (self.users[None, :] == user)
        return Ks

    def _predict(self, J, user=None, kind="known", full_cov=False):
        J = np.asarray(J)
        if J.ndim != 1 or not np.issubdtype(J.dtype, np.integer) or np.any((J < 0) | (J >= len(self.kern.t))):
            raise ValueError("invalid candidate indices")
        Ks = self._cross(J, user, kind)
        mu = Ks @ self.alpha
        V = solve_triangular(self.L, Ks.T, lower=True)
        prior_scale = self.p["omega"] if kind == "population" and self.users is not None else 1.0
        if full_cov:
            Kss = prior_scale * self.kern.gram(self.p, J, J)
            cov = (Kss - V.T @ V) * self.y_sd ** 2
            cov = (cov + cov.T) / 2
            if len(J) and np.min(np.diag(cov)) < -1e-6 * max(1.0, np.max(np.diag(Kss))):
                raise np.linalg.LinAlgError("negative posterior variance")
            return mu * self.y_sd + self.y_mu, cov
        var = prior_scale * self.kern.diag(self.p, J) - (V ** 2).sum(0)
        if len(J) and np.min(var) < -1e-6 * max(1.0, np.max(self.kern.diag(self.p, J))):
            raise np.linalg.LinAlgError("negative posterior variance")
        return mu * self.y_sd + self.y_mu, np.maximum(var, 0) * self.y_sd ** 2

    def predict(self, J, user=None, full_cov=False):
        return self._predict(J, user, full_cov=full_cov)

    def predict_new_user(self, J, full_cov=False):
        """New user has independent deviation; prior variance is the full kernel."""
        return self._predict(J, kind="new", full_cov=full_cov)

    def predict_population(self, J, full_cov=False):
        """Only the shared component; smaller prior variance than a new user."""
        return self._predict(J, kind="population", full_cov=full_cov)

    def relevance(self, names=None):
        if not self.fitted:
            raise ValueError("fit the GP first")
        names = names or [f"factor{i}" for i in range(self.kern.F)]
        return dict(zip(names, np.round(self.p["amps"] ** 2, 3)))  # associative, not causal


def pd_isna_any(a):
    # No pandas dependency in inference; object NaN is different from itself.
    return any(x is None or x != x for x in a)


def full_gram(kern, p, chunk=256):
    N = len(kern.t)
    idx = np.arange(N)
    K = np.empty((N, N))
    for s in range(0, N, chunk):
        K[s:s + chunk] = kern.gram(p, idx[s:s + chunk], idx)
    return K
