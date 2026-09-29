"""Stage 2: BO loop on synthetic taste. Compares GP-UCB, GP-EI, random, kNN-greedy."""
import sys, time, numpy as np
from .synthetic import make_catalog, make_taste
from .gp import GP
from .acquisition import ucb, expected_improvement, maxmin_init


def run(method, table, hidden, seed, T=25, n_init=6, refit_every=3):
    N = len(table)
    f, _ = make_taste(hidden, seed=100 + seed)
    rng = np.random.default_rng(seed)
    truth = f(np.arange(N))
    # Common random numbers: observation noise belongs to a (run, wine), not
    # to the sequence of RNG calls made by each policy.
    observed = truth + 0.3 * np.random.default_rng(seed + 10000).standard_normal(N)
    gp = GP(table, n_restarts=1, seed=seed)
    p0 = gp.kern.unpack(gp.kern.default_theta())
    class Rows:
        diag = gp.kern.diag(p0, np.arange(N))
        def __call__(self, j):
            return gp.kern.gram(p0, np.array([j]), np.arange(N))[0]
    obs = maxmin_init(Rows(), n_init, rng=rng)
    y = list(observed[obs])
    best_true = truth.max()
    regret = []
    for t in range(T):
        rest = np.setdiff1d(np.arange(N), obs)
        if method == "random":
            j = int(rng.choice(rest))
        else:
            gp.fit(obs, y, optimize=(t % refit_every == 0))
            mu, var = gp.predict(rest)
            if method == "gp_ucb":
                s = ucb(mu, var, 1.5 * 0.95 ** t)
            else:
                s = expected_improvement(mu, var, max(gp.predict(np.asarray(obs))[0]))
            j = int(rest[np.argmax(s)])
        obs.append(j); y.append(float(observed[j]))
        regret.append(best_true - truth[obs].max())
    return np.array(regret), gp


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    table, hidden = make_catalog(N)
    res = {}
    for m in ["random", "gp_ucb", "gp_ei"]:
        t0 = time.time(); R = []
        for s in range(seeds):
            r, gp = run(m, table, hidden, s); R.append(r)
        R = np.array(R); res[m] = R
        print(f"{m:8s} regret@5={R[:,4].mean():.2f} @15={R[:,14].mean():.2f} @25={R[:,-1].mean():.2f}  ({time.time()-t0:.0f}s)")
        if m == "gp_ucb":
            print("learned relevance:", gp.relevance(table.cat_names), "ell:", np.round(gp.p["ell"], 2))
