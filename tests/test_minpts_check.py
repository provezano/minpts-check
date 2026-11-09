"""Tests: backend equivalence on synthetic data, and the paper's stored runs as an oracle for s_hat, the at-risk
flags and the certificate.  Nothing is written to disk.  The oracle tests are skipped when the stored study outputs
are not present (set MINPTS_ORACLE_DIR to the study folder holding p7_data/ and p7_out/)."""
import hashlib
import json
import os

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from minpts_check import check_min_samples

ORACLE = os.environ.get("MINPTS_ORACLE_DIR", "")
DATASETS = ["oml_31", "oml_181", "oml_36", "oml_1466"]
hdbscan = pytest.importorskip("hdbscan")
EXACT = dict(algorithm="prims_kdtree", approx_min_span_tree=False, core_dist_n_jobs=1)


def synthetic(seed=0, n=1500):
    X, _ = make_blobs(n, centers=[[0, 0], [5, 0], [0, 5], [6, 6], [9, 1]], cluster_std=[1, 1, .5, 1.5, .4],
                      random_state=seed)
    rng = np.random.default_rng(seed)
    return np.vstack([X, rng.uniform(-4, 11, (n // 10, 2))]) + rng.normal(0, 1e-9, (n + n // 10, 2))


def same_partition(a, b):
    return a.max() == b.max() and np.array_equal(a < 0, b < 0) and adjusted_rand_score(a, b) == 1.0


@pytest.mark.parametrize("m", [5, 10, 20])
def test_backends_agree_once_convention_is_normalised(m):
    X = synthetic()
    for coupled in (True, False):
        _agree(X, m, coupled)


def _agree(X, m, coupled):
    sk = check_min_samples(X, m + 1, m, backend="sklearn", steps=2, mode="full", coupled=coupled)
    hd = check_min_samples(X, m, m, backend="hdbscan", steps=2, mode="full", coupled=coupled, **EXACT)
    assert same_partition(sk.labels[m + 1], hd.labels[m])
    for a, b in zip(sk.steps, hd.steps):
        assert a["min_samples"] == b["min_samples"] + 1 and a["min_cluster_size"] == b["min_cluster_size"]
        assert same_partition(sk.labels[a["min_samples"]], hd.labels[b["min_samples"]])
        assert a["certified"] == b["certified"] and a["U_size"] == b["U_size"]
    assert [c["s_hat"] for c in sk.clusters] == [c["s_hat"] for c in hd.clusters]


def test_sklearn_path_equals_public_estimator():
    from sklearn.cluster import HDBSCAN
    X = synthetic(1)
    for ms, mcs in [(5, 5), (11, 10), (21, 20)]:
        rep = check_min_samples(X, ms, mcs, backend="sklearn", steps=1)
        assert same_partition(rep.labels[ms], HDBSCAN(min_cluster_size=mcs, min_samples=ms).fit(X).labels_)


def test_fast_mode_does_not_refit_and_verify_skips_certified():
    X = synthetic(2)
    fast = check_min_samples(X, 10, 10, backend="hdbscan", steps=2, mode="fast", **EXACT)
    assert fast.fits == 1 and all(s["ari"] is None for s in fast.steps)
    ver = check_min_samples(X, 10, 10, backend="hdbscan", steps=2, mode="verify", u_cap=1.0, **EXACT)
    n_cert = sum(bool(s["certified"]) for s in ver.steps)
    assert ver.fits_skipped == n_cert and ver.fits == 1 + len(ver.steps) - n_cert
    d = ver.to_dict()
    json.dumps(d)
    assert "verdict" in d and ver.summary(file=open(os.devnull, "w"))


def test_approximate_tree_warns():
    rep = check_min_samples(synthetic(3), 10, 10, backend="hdbscan", steps=1)
    assert not rep.exact_tree and any("29.3%" in w for w in rep.warnings)
    assert all(s["certified"] is None for s in rep.steps)


# ---------------------------------------------------------------- oracle: the paper's stored runs
def _s64(k):
    return int.from_bytes(hashlib.sha256(k.encode()).digest()[:8], "big")


def r6_data(key, s=0):
    """the preprocessing of the paper's analysis code."""
    Z = np.load(os.path.join(ORACLE, "p7_data", f"{key}.npz"))
    X0 = Z["X"].astype(np.float64)
    mn, mx = X0.min(0), X0.max(0)
    X0 = np.unique((X0 - mn) / np.where(mx > mn, mx - mn, 1.0), axis=0)
    k = min(8000, int(0.9 * len(X0)))
    rows = np.sort(np.random.default_rng(_s64(f"P7/{key}/sub/{s}")).choice(len(X0), k, replace=False))
    return np.ascontiguousarray(X0[rows] + np.random.default_rng(_s64(f"P7/{key}/{s}/jitter")).normal(0, 1e-9, (k, X0.shape[1])))


def oracle_steps(key, backend):
    """yield (oracle step, our report, our labels at m, oracle labels at m and m')."""
    out = os.path.join(ORACLE, "p7_out")
    if not os.path.exists(os.path.join(out, f"{key}_s0.json")):
        pytest.skip("oracle outputs not available")
    J = json.load(open(os.path.join(out, f"{key}_s0.json")))
    L = np.load(os.path.join(out, f"{key}_s0_labels.npz"))["L"]
    X = r6_data(key)
    assert len(X) == J["n"]
    for i, st in enumerate(J["steps"]):
        m, m1 = st["m0"], st["m1"]
        if backend == "hdbscan":
            rep = check_min_samples(X, m, m, backend="hdbscan", steps=m1 - m, mode="fast", **EXACT)
        else:   # same run in scikit-learn's convention: min_samples m + 1, min_cluster_size m, moved together
            rep = check_min_samples(X, m + 1, m, backend="sklearn", steps=m1 - m, mode="fast", coupled=True)
        yield st, rep, L[i], L[i + 1]


def holds_outside(lab0, lab1, U):
    """the certificate's claim: same number of clusters, labels equal up to renaming and noise equal outside U."""
    out = ~U
    if lab0.max() != lab1.max():
        return False
    a, b = lab0[out], lab1[out]
    if not np.array_equal(a < 0, b < 0):
        return False
    pairs = set(zip(a[a >= 0].tolist(), b[b >= 0].tolist()))
    return len(pairs) == len({p[0] for p in pairs}) == len({p[1] for p in pairs})


@pytest.mark.parametrize("backend", ["hdbscan", "sklearn"])
@pytest.mark.parametrize("key", DATASETS)
def test_oracle_shat_flags_and_certificate(key, backend):
    n_cl = n_steps = n_cert = 0
    off = int(backend == "sklearn")
    for st, rep, l0, l1 in oracle_steps(key, backend):
        assert same_partition(rep.labels[st["m0"] + off], l0)
        j = st["m1"] - st["m0"]
        ours = [(c["size"], c["s_hat"][j - 1], c["at_risk"][j - 1]) for c in rep.clusters]
        theirs = [(c["s"], c["shat"], c["shat"] < st["m1"]) for c in st["cl"]]
        assert ours == theirs
        up = [s for s in rep.steps if s["min_samples"] == st["m1"] + off][0]
        if st["K0"] >= 1:
            assert bool(up["certified"]) == bool(st["cert"])
            if st["cert"]:
                assert abs(up["U_share"] - st["U_frac"]) < 1e-12
                U = np.zeros(len(l0), bool); U[up["U"]] = True
                assert holds_outside(rep.labels[st["m0"] + off], l1, U)   # never violated
                n_cert += 1
        n_cl += len(ours); n_steps += 1
    print(f"{key} ({backend}): {n_steps} steps, {n_cl} clusters, {n_cert} certified")
