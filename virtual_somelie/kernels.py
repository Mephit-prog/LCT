"""Positive semidefinite feature kernels (rows are fixed catalog objects)."""
import numpy as np
from scipy.linalg import eigh


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softplus(z):
    return np.logaddexp(0.0, z)


def complete_laplacian(n):
    return n * np.eye(n) - np.ones((n, n))


def path_laplacian(m):
    if m == 1:
        return np.zeros((1, 1))
    A = np.eye(m, k=1) + np.eye(m, k=-1)
    return np.diag(A.sum(1)) - A


def diffusion_gram(L, beta):
    """exp(-beta L); beta=0 is the identity, NOT a constant kernel."""
    w, U = eigh(L)
    return (U * np.exp(-beta * w)) @ U.T


def normalize_gram(K):
    d = np.sqrt(np.diag(K))
    return K / np.outer(d, d)


def tau_from_beta(n, beta):
    rho = np.exp(-n * beta)
    return (1.0 - rho) / (1.0 + (n - 1) * rho)


def cartesian_laplacian(Ls):
    total = np.zeros((1, 1))
    for L in Ls:
        n1, n2 = total.shape[0], L.shape[0]
        total = np.kron(total, np.eye(n2)) + np.kron(np.eye(n1), L)
    return total


def cat_factor(P, Q, tau):
    """Mean-embedding kernel P K_cat Q.T. Soft rows are mixtures, not posterior inference."""
    return (1.0 - tau) * (P @ Q.T) + tau * np.outer(P.sum(1), Q.sum(1))


def graph_factor(P, Q, Kg):
    return P @ Kg @ Q.T


def hier_factor(Ps, Qs, a):
    """Additive hierarchy, retained for old callers; levels must have full path keys."""
    a = np.asarray(a) / np.sum(a)
    return sum(al * (P @ Q.T) for al, P, Q in zip(a, Ps, Qs))


def matern52(X, Y, ell):
    """ARD Matern-5/2 on *fixed*, imputed inputs. No pairwise noise inflation (not PSD)."""
    X, Y, ell = np.asarray(X), np.asarray(Y), np.asarray(ell)
    if np.any(ell <= 0) or not np.isfinite(ell).all():
        raise ValueError("ell must be finite and positive")
    r = np.sqrt((((X[:, None, :] - Y[None, :, :]) / ell) ** 2).sum(-1))
    s5 = np.sqrt(5.0) * r
    return (1.0 + s5 + 5.0 * r ** 2 / 3.0) * np.exp(-s5)


def cosine_kernel(T, U):
    return T @ U.T
