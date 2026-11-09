"""Wall time on two 10,000-point examples (run with one thread: OMP_NUM_THREADS=1).
"separated": 6 well-separated blobs in 10-D (the certificate fires); "mixed": 8 blobs of unequal spread in 5-D.
Usage: python tests/bench_10k.py [n]   (n = number of points, default 10000)"""
import sys
import time

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs

from minpts_check import check_min_samples, sweep

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000


def example(name, n=N):
    rng = np.random.default_rng(0)
    if name == "separated":
        X, _ = make_blobs(int(0.95 * n), n_features=10, centers=6, center_box=(-15, 15), random_state=1)
    else:
        X, _ = make_blobs(int(0.9 * n), n_features=5, centers=8, cluster_std=rng.uniform(0.5, 2.0, 8), random_state=0)
    X = np.vstack([X, rng.uniform(X.min(0), X.max(0), (n - len(X), X.shape[1]))])
    return X + rng.normal(0, 1e-9, X.shape), 30 if name == "separated" else 15


def timed(f):
    t = time.perf_counter()
    r = f()
    return time.perf_counter() - t, r


for name in ("separated", "mixed"):
    X, m = example(name)
    s, _ = timed(lambda: HDBSCAN(min_samples=m + 1, min_cluster_size=m).fit(X))
    print(f"{name:9s} plain sklearn.cluster.HDBSCAN fit        {s:7.2f} s")
    runs = [("fast", dict(mode="fast")),
            ("fast + certified range", dict(mode="fast", range_search="binary")),
            ("fast + tie check", dict(mode="fast", ties="check")),
            ("verify", dict(mode="verify")),
            ("full", dict(mode="full")),
            ("full, canonical + chain", dict(mode="full", ties="canonical", refit="chain"))]
    for label, kw in runs:
        s, rep = timed(lambda: check_min_samples(X, m + 1, m, backend="sklearn", steps=2, coupled=True, **kw))
        print(f"{name:9s} {label:39s} {s:7.2f} s  fits={rep.fits} skipped={rep.fits_skipped}  {rep.verdict}")
    grid = list(range(m - 4, m + 9))
    s1, _ = timed(lambda: [HDBSCAN(min_samples=g, min_cluster_size=g).fit(X) for g in grid])
    s2, _ = timed(lambda: sweep(X, grid))
    print(f"{name:9s} {len(grid)} values: sklearn refits {s1:.2f} s, sweep (chained) {s2:.2f} s ({s1 / s2:.1f}x)")
rep.summary()
