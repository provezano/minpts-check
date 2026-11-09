"""Wall time of the three modes on two 10,000-point examples (run with one thread: OMP_NUM_THREADS=1).
"separated": 6 well-separated blobs in 10-D (the certificate fires); "mixed": 8 blobs of unequal spread in 5-D."""
import time

import numpy as np
from sklearn.datasets import make_blobs

from minpts_check import check_min_samples


def example(name):
    rng = np.random.default_rng(0)
    if name == "separated":
        X, _ = make_blobs(9500, n_features=10, centers=6, center_box=(-15, 15), random_state=1)
    else:
        X, _ = make_blobs(9000, n_features=5, centers=8, cluster_std=rng.uniform(0.5, 2.0, 8), random_state=0)
    X = np.vstack([X, rng.uniform(X.min(0), X.max(0), (10000 - len(X), X.shape[1]))])
    return X + rng.normal(0, 1e-9, X.shape), 30 if name == "separated" else 15


for name in ("separated", "mixed"):
    X, m = example(name)
    for backend, ms, kw in [("sklearn", m + 1, {}), ("hdbscan", m, dict(approx_min_span_tree=False))]:
        for mode in ("fast", "verify", "full"):
            t = time.perf_counter()
            rep = check_min_samples(X, ms, m, backend=backend, steps=2, mode=mode, coupled=True, **kw)
            print(f"{name:9s} {backend:8s} {mode:6s} {time.perf_counter() - t:6.2f} s  fits={rep.fits} "
                  f"skipped={rep.fits_skipped}  {rep.verdict}")
rep.summary()
