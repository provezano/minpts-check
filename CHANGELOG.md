# Changelog

## 0.2.0 (2026-11-09)

### Added

- `certified_range(X, min_samples, min_cluster_size, ...)` and `check_min_samples(..., range_search="binary")`:
  the largest `min_samples` up to which the range form of the certificate holds, found by doubling and bisection
  (a few certificate checks instead of one per value). `search="linear"` evaluates every top. The upper
  hierarchy is now built on one fixed edge set (observed spanning tree and k-NN graph at the largest value
  tried), which makes the certificate monotone in the top; the docstring of `certified_range` gives the
  argument and its one unproved step (the choice of anchors), which the tests check by brute force.
- Tie-merged canonical labels: `canonical_labels(...)` and `check_min_samples(..., ties="canonical")`.
  Edges of equal weight are removed simultaneously, as in the definition of HDBSCAN* (Campello et al. 2015,
  Sec. 3.2), so the labels are the same for every exact spanning tree and every row order.
  `ties="check"` keeps the library's labels and reports their agreement with the canonical ones.
- `sweep(X, values, ...)` and `check_min_samples(..., refit="chain")`: canonical labels for many values of
  `min_samples` from one exact fit, each spanning tree computed from the next larger one (CORE-SG containment,
  Neto et al. 2022).
- `MinPtsReport.root_preference`: whether the excess-of-mass extraction would select the root. Reported as an
  empirical diagnostic only.
- `MinPtsReport` fields `ties`, `certified_range`, `root_preference`, `fits_chained` and `version`; in
  `"verify"` mode, steps inside a certified range with a small tolerance set skip their refit; in `"full"`
  mode, range claims are checked against the refits (`range_holds_on_refit`).
- Tests for all of the above, including a lattice data set with deliberate ties and an oracle test of the
  certified ranges against the paper's stored runs.

### Unchanged

- The 0.1.0 call `check_min_samples(X, min_samples, min_cluster_size, backend, steps, mode, coupled, u_cap,
  **kwargs)` and its defaults give the same labels, `s_hat`, flags and per-step certificates as 0.1.0. The new
  options are keyword-only.

## 0.1.0 (2026-11-09)

First release: `check_min_samples` with the one-run predictor `s_hat`, the per-step certificate and optional
refits, for scikit-learn and the `hdbscan` package.
