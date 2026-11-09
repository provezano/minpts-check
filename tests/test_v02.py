"""Tests of the 0.2.0 additions: range search, tie-merged canonical labels, chained spanning trees and the
root-preference flag.  Nothing is written to disk.  The oracle test at the end is skipped unless MINPTS_ORACLE_DIR
points to the stored study outputs (p7_data/ and p7_out/)."""
import json
import os

import numpy as np
import pytest
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score

import minpts_check as mc
from minpts_check import canonical_labels, certified_range, check_min_samples, sweep

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def random_set(seed):
    """small mixtures of Gaussian blobs plus uniform noise, 1e-9 jitter (as in the paper)."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(80, 300)); kc = int(rng.integers(2, 7)); d = int(rng.choice([2, 2, 3, 5]))
    sizes = rng.multinomial(int(n * rng.uniform(0.6, 0.95)), rng.dirichlet(np.ones(kc) * rng.uniform(0.5, 3)))
    pts = [rng.normal(rng.uniform(0, 10, d), rng.uniform(0.2, 1.2), (s, d)) for s in sizes if s > 0]
    pts.append(rng.uniform(-1, 11, (n - sum(sizes), d)))
    X = np.vstack(pts)
    return X + rng.normal(0, 1e-9, X.shape)


def lattice(seed=0):
    """integer lattice patches plus scattered lattice points: many exactly equal distances (deliberate ties)."""
    rng = np.random.default_rng(seed)
    pats = [np.array([(cx + i * st, cy + j * st) for i in range(w) for j in range(h)], float)
            for cx, cy, w, h, st in [(0, 0, 12, 10, 1), (20, 0, 8, 8, 1), (0, 20, 15, 6, 1), (18, 16, 6, 6, 2),
                                     (40, 40, 5, 5, 1)]]
    noise = np.array(sorted({(int(a), int(b)) for a, b in rng.integers(-10, 55, (80, 2))}), float)
    return np.unique(np.vstack(pats + [noise]), axis=0)


def same_partition(a, b):
    return a.max() == b.max() and np.array_equal(a < 0, b < 0) and adjusted_rand_score(a, b) == 1.0


# ---------------------------------------------------------------- range search
SEEDS = range(40)


@pytest.mark.parametrize("seed", SEEDS)
def test_binary_search_equals_linear_scan(seed):
    """brute force: the linear scan evaluates every top; the certified tops must form a prefix (monotonicity) and
    binary search must find the same largest top and the same tolerance set."""
    X = random_set(seed)
    for m in (3, 6):
        lin = certified_range(X, m, m, max_steps=8, search="linear")
        ev = [lin["evaluated"][v] for v in sorted(lin["evaluated"])]
        assert all(ev[i] or not ev[i + 1] for i in range(len(ev) - 1)), f"non-monotone tops: {lin['evaluated']}"
        bi = certified_range(X, m, m, max_steps=8, search="binary")
        assert bi["top_min_samples"] == lin["top_min_samples"]
        assert bi["U"] == lin["U"]
        assert bi["checks"] <= lin["checks"]


def test_binary_search_logic():
    """the search alone, on monotone predicates with every possible threshold."""
    for max_u in range(1, 40):
        for t in range(0, max_u + 1):
            u, memo = mc._search_top(lambda x: x <= t, max_u, "binary")
            assert u == t and len(memo) <= 2 * int(np.log2(max_u)) + 3


@pytest.mark.parametrize("seed", range(12))
def test_range_claim_holds_on_every_refit(seed):
    """soundness of the range certificate: refit every value inside the certified range (library labels)."""
    X = random_set(100 + seed)
    for m in (3, 5):
        rep = check_min_samples(X, m, m, steps=6, mode="full", range_search="binary", range_steps=6)
        assert not any("claim failed" in w for w in rep.warnings)
        for s in rep.steps:
            if s["in_certified_range"]:
                assert s["range_holds_on_refit"] is True


@pytest.mark.parametrize("seed", range(20))
def test_fixed_edge_set_not_weaker_than_v01_step(seed):
    """the fixed edge set contains v0.1's, so a step v0.1 certifies is inside the certified range."""
    X = random_set(200 + seed)
    for m in (3, 6):
        rep = check_min_samples(X, m, m, steps=1, range_search="binary", range_steps=6)
        if rep.steps and rep.steps[-1]["certified"]:
            assert rep.certified_range["steps_certified"] >= 1


def test_range_in_verify_mode_skips_refits():
    for seed in range(30):
        X = random_set(300 + seed)
        rep = check_min_samples(X, 4, 4, steps=3, mode="verify", u_cap=1.0, range_search="binary", range_steps=3)
        r = rep.certified_range
        if r["steps_certified"] >= 2:
            inside = [s for s in rep.steps if s["in_certified_range"]]
            assert len(inside) == r["steps_certified"] and all(s["ari"] is None for s in inside)
            return
    pytest.skip("no seed with a certified range of two steps")


# ---------------------------------------------------------------- tie-merged canonical labels
@pytest.mark.parametrize("seed", range(20))
def test_canonical_equals_library_when_weights_are_distinct(seed):
    """min_samples = 1 (sklearn convention): edge weights are plain distances, pairwise distinct with the jitter,
    and the n-ary tree is the binary one; the canonical labels must equal the library's."""
    X = random_set(400 + seed)
    for mcs in (3, 5, 9):
        assert same_partition(canonical_labels(X, 1, mcs), HDBSCAN(min_samples=1, min_cluster_size=mcs).fit(X).labels_)


@pytest.mark.parametrize("m", [4, 6, 8, 10])
def test_canonical_labels_stable_under_ties(m):
    """deliberate ties (integer lattice): the canonical labels are the same for every row order and for both
    backends (whose exact spanning trees differ), while the library's labels may change with the row order."""
    X = lattice()
    n = len(X)
    base = canonical_labels(X, m, m)
    assert base.max() >= 1
    for s in range(5):
        p = np.random.default_rng(s).permutation(n)
        inv = np.argsort(p)
        assert same_partition(base, canonical_labels(X[p], m, m)[inv])
        assert same_partition(base, canonical_labels(X[p], m - 1, m, backend="hdbscan",
                                                     approx_min_span_tree=False)[inv])


def test_library_labels_can_change_under_ties():
    """the phenomenon the canonical option removes (documented, so asserted only as 'at least once')."""
    X = lattice()
    n = len(X)
    changed = False
    for m in (4, 6, 8, 10):
        ref = HDBSCAN(min_samples=m, min_cluster_size=m).fit(X).labels_
        for s in range(5):
            p = np.random.default_rng(s).permutation(n)
            changed |= not same_partition(ref, HDBSCAN(min_samples=m, min_cluster_size=m).fit(X[p]).labels_[np.argsort(p)])
    assert changed


def test_ties_check_reports_agreement():
    X = lattice()
    rep = check_min_samples(X, 6, 6, steps=1, ties="check")
    t = rep.ties
    assert t["tied_edges_share"] > 0.5 and 0 <= t["ari_library_vs_canonical"] <= 1
    assert t["same_partition"] == same_partition(rep.labels[6], canonical_labels(X, 6, 6))
    assert rep.labels[6].max() + 1 == t["library_n_clusters"]


# ---------------------------------------------------------------- chained spanning trees
@pytest.mark.parametrize("seed", range(10))
def test_sweep_equals_direct_canonical_fits(seed):
    X = random_set(500 + seed)
    grid = [g for g in (2, 3, 5, 8, 12, 20) if 4 * g <= len(X)]
    sw = sweep(X, grid)
    for g in grid:
        assert same_partition(sw[g], canonical_labels(X, g, g))
    fixed = sweep(X, grid, min_cluster_size=6, backend="hdbscan", approx_min_span_tree=False)
    for g in grid:
        assert same_partition(fixed[g], canonical_labels(X, g, 6, backend="hdbscan", approx_min_span_tree=False))


@pytest.mark.parametrize("seed", range(6))
def test_chain_refits_equal_backend_refits(seed):
    X = random_set(600 + seed)
    a = check_min_samples(X, 6, 6, steps=3, mode="full", ties="canonical", refit="chain")
    b = check_min_samples(X, 6, 6, steps=3, mode="full", ties="canonical", refit="backend")
    assert a.labels.keys() == b.labels.keys()
    for k in a.labels:
        assert same_partition(a.labels[k], b.labels[k])
    assert a.fits_chained >= 1 and b.fits_chained == 0 and a.fits == b.fits


# ---------------------------------------------------------------- root preference
@pytest.mark.parametrize("seed", range(15))
def test_root_flag_matches_allow_single_cluster(seed):
    """the flag says EOM would pick the root; that is exactly when allow_single_cluster=True changes the output."""
    X = random_set(700 + seed)
    for m in (3, 6, 10):
        rep = check_min_samples(X, m, m, steps=0)
        single = HDBSCAN(min_samples=m, min_cluster_size=m, allow_single_cluster=True).fit(X).labels_
        assert rep.root_preference["flag"] == (not same_partition(rep.labels[m], single))
        assert "empirical" in rep.root_preference["note"]


# ---------------------------------------------------------------- API
def test_defaults_add_nothing_and_options_serialise():
    X = random_set(1)
    rep = check_min_samples(X, 5, 5, steps=1)
    assert rep.ties is None and rep.certified_range is None and rep.fits_chained == 0
    assert rep.version == mc.__version__ == "0.2.0"
    rep = check_min_samples(X, 5, 5, steps=2, mode="full", ties="check", range_search="binary")
    json.dumps(rep.to_dict())
    assert "certified range" in rep.summary(file=open(os.devnull, "w"))
    rep = check_min_samples(X, 5, 5, steps=2, mode="verify", ties="canonical", refit="chain")
    json.dumps(rep.to_dict(labels=True))
    assert all(s["certified"] is None for s in rep.steps)


def test_invalid_options():
    X = random_set(2)
    with pytest.raises(ValueError):
        check_min_samples(X, 5, 5, refit="chain")
    with pytest.raises(ValueError):
        check_min_samples(X, 5, 5, ties="canonical", backend="hdbscan")       # approximate tree by default
    with pytest.raises(ValueError):
        check_min_samples(X, 5, 5, range_search="ternary")
    with pytest.raises(ValueError):
        sweep(X, [3, 5], cluster_selection_method="leaf")


# ---------------------------------------------------------------- oracle: range claims against the paper's runs
ORACLE = os.environ.get("MINPTS_ORACLE_DIR", "")


@pytest.mark.parametrize("key", ["oml_31", "oml_181", "oml_36", "oml_1466"])
def test_oracle_range_claims_hold_on_stored_runs(key):
    """certified ranges from every stored grid value, checked against the paper's stored labels at every grid value
    inside the range (hdbscan convention, coupled, exact tree)."""
    from test_minpts_check import EXACT, holds_outside, r6_data
    out = os.path.join(ORACLE, "p7_out")
    if not os.path.exists(os.path.join(out, f"{key}_s0.json")):
        pytest.skip("oracle outputs not available")
    J = json.load(open(os.path.join(out, f"{key}_s0.json")))
    L = np.load(os.path.join(out, f"{key}_s0_labels.npz"))["L"]
    grid = [s["m0"] for s in J["steps"]] + [J["steps"][-1]["m1"]]
    X = r6_data(key)
    for i, m in enumerate(grid[:-1]):
        r = certified_range(X, m, m, backend="hdbscan", max_steps=min(12, grid[-1] - m), **EXACT)
        for j in range(i + 1, len(grid)):
            if grid[j] <= r["top_min_samples"]:
                U = np.zeros(len(X), bool); U[r["U"]] = True
                assert holds_outside(L[i], L[j], U)
