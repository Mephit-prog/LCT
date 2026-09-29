"""Stage 1 kernel and synthetic tests; run pytest virtual_somelie/."""
import numpy as np
from scipy.linalg import expm
from .kernels import (complete_laplacian, path_laplacian, diffusion_gram,
                      normalize_gram, tau_from_beta, cartesian_laplacian,
                      cat_factor, graph_factor, hier_factor, matern52)
from .table import onehot, multihot
from .synthetic import make_catalog
from .gp import WineKernel, GP


def psd(K, tol=1e-8):
    return np.linalg.eigvalsh(0.5 * (K + K.T)).min() >= -tol


def test_complete_graph_closed_form():
    for n, b in [(3, 0.3), (7, 1.1), (20, 0.05)]:
        K = diffusion_gram(complete_laplacian(n), b)
        rho = np.exp(-n * b)
        assert np.allclose(np.diag(K), (1 + (n - 1) * rho) / n)
        assert np.allclose(K[0, 1], (1 - rho) / n)
        Kn = normalize_gram(K)
        assert np.isclose(Kn[0, 1], tau_from_beta(n, b))


def test_kronecker_factorization():
    Ls = [complete_laplacian(3), path_laplacian(4), complete_laplacian(2)]
    bs = [0.4, 0.9, 0.2]
    Lc = cartesian_laplacian([b * L for b, L in zip(bs, Ls)])
    Kfull = expm(-Lc)
    Kk = np.kron(np.kron(expm(-bs[0] * Ls[0]), expm(-bs[1] * Ls[1])), expm(-bs[2] * Ls[2]))
    assert np.allclose(Kfull, Kk, atol=1e-10)


def test_hamming_equals_shortest_path():
    Lc = cartesian_laplacian([complete_laplacian(3), complete_laplacian(3), complete_laplacian(2)])
    A = np.diag(np.diag(Lc)) - Lc
    n = A.shape[0]
    D = np.full((n, n), np.inf); np.fill_diagonal(D, 0); D[A > 0] = 1
    for k in range(n):
        D = np.minimum(D, D[:, [k]] + D[[k], :])
    idx = np.array([(a, b, c) for a in range(3) for b in range(3) for c in range(2)])
    H = (idx[:, None, :] != idx[None, :, :]).sum(-1)
    assert np.array_equal(D, H)


def test_irrelevance_limits():
    P = onehot(np.array([0, 1, 2, 0]), 3)
    assert np.allclose(cat_factor(P, P, 1.0), 1.0)
    assert np.allclose(cat_factor(P, P, 0.0), P @ P.T)
    assert tau_from_beta(5, 50.0) > 0.99 and tau_from_beta(5, 1e-6) < 1e-4


def test_soft_and_multihot_psd():
    r = np.random.default_rng(0)
    codes = r.integers(-1, 6, size=60)
    P = onehot(codes, 6)
    assert psd(cat_factor(P, P, 0.3))
    M = multihot([list(r.integers(0, 10, size=r.integers(0, 3))) for _ in range(60)], 10)
    assert psd(cat_factor(M, M, 0.2))


def test_path_and_hierarchy_psd():
    r = np.random.default_rng(1)
    P = onehot(r.integers(0, 8, size=50), 8)
    Kp = normalize_gram(diffusion_gram(path_laplacian(8), 0.7))
    assert psd(graph_factor(P, P, Kp))
    c = r.integers(0, 4, size=50); reg = c * 3 + r.integers(0, 3, size=50)
    Ps = [onehot(c, 4), onehot(reg, 12)]
    K = hier_factor(Ps, Ps, [0.3, 0.7])
    assert psd(K) and np.allclose(np.diag(K), 1.0)


def test_matern_and_input_noise():
    r = np.random.default_rng(2)
    X = r.standard_normal((40, 3)); ell = np.array([1.0, 2.0, 0.5])
    assert psd(matern52(X, X, ell)) and np.allclose(np.diag(matern52(X, X, ell)), 1.0)
    # Heterogeneous pairwise lengthscale inflation is deliberately unsupported:
    # it is not a PSD kernel. Missingness is an explicit fixed input feature.
    import pytest
    with pytest.raises(TypeError):
        matern52(X, X, ell, X, X)


def test_full_kernel_psd_and_symmetry():
    t, _ = make_catalog(200, seed=3)
    k = WineKernel(t)
    th = k.default_theta() + 0.3 * np.random.default_rng(0).standard_normal(k.n_par)
    p = k.unpack(th); idx = np.arange(200)
    K = k.gram(p, idx, idx)
    assert np.allclose(K, K.T) and psd(K, 1e-6)


def test_gp_interpolates_and_calibrates():
    t, h = make_catalog(300, seed=4)
    from .synthetic import make_taste
    f, rate = make_taste(h, noise=0.05)
    rng = np.random.default_rng(0)
    tr = rng.choice(300, 60, replace=False); te = np.setdiff1d(np.arange(300), tr)
    gp = GP(t, n_restarts=1).fit(tr, rate(tr, rng))
    mu, var = gp.predict(te)
    truth = f(te)
    rho = np.corrcoef(mu, truth)[0, 1]
    cover = np.mean(np.abs(mu - truth) < 1.645 * np.sqrt(var + 0.05 ** 2))
    assert rho > 0.6, rho
    assert 0.7 < cover <= 1.0, cover


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            out = fn(); print("OK ", name, "" if out is None else out)
