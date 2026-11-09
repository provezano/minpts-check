# minpts_check

Does your HDBSCAN* clustering survive a small change of `min_samples`? One call answers this for
`sklearn.cluster.HDBSCAN` and for the `hdbscan` package. It implements the routine of Coutinho (2026):

- At-risk clusters: for each selected cluster, `s_hat` counts its points that stay dense enough at the next
  `min_samples` (core distance within the cluster's birth radius). A cluster with `s_hat` below the next
  `min_cluster_size` is flagged as at risk of being lost.
- Certificate: a proof, from the same run, that raising `min_samples` changes no label outside an explicit
  set `U` of points.
- Neighbour agreement (optional refits): ARI between the chosen value and its neighbours, over all points and
  over the points clustered in both outputs, plus the change in the number of clusters.

`s_hat` and the certificate need one fit and one nearest-neighbour query, with no refit.

Version 0.2.0 adds four things:

- `certified_range` finds the largest `min_samples` up to which the certificate holds, with a few certificate
  checks instead of one per value.
- `canonical_labels` and the `ties=` option give HDBSCAN* labels as defined by Campello et al. (2015), with edges
  of equal weight removed at once, so the labels do not depend on which exact spanning tree a library builds.
- `sweep` and `refit="chain"` compute labels for many values of `min_samples` from one exact fit.
- The report flags runs where the extraction would prefer the root, as an empirical diagnostic.

## Install

```
pip install git+https://github.com/provezano/minpts-check            # numpy, scipy, scikit-learn
pip install "minpts-check[hdbscan] @ git+https://github.com/provezano/minpts-check"   # also the hdbscan package
```

## Usage

```python
from minpts_check import check_min_samples

rep = check_min_samples(X, min_samples=11, min_cluster_size=10, backend="sklearn", steps=2)
rep.summary()        # plain-text table and a one-line verdict
d = rep.to_dict()    # the same as a dict (tolerance sets as point indices)

# 0.2.0 options (keyword-only; the defaults give the 0.1.0 output)
rep = check_min_samples(X, 11, 10, range_search="binary", ties="check")
```

Modes:

| `mode` | what it fits | what it reports |
|---|---|---|
| `"fast"` (default) | the chosen value only | `s_hat`, at-risk flags and the certificate for every higher neighbour |
| `"verify"` | also the neighbours not certified with a tolerance set of at most `u_cap` (default 1%) of the points | as above, plus ARI for the refitted neighbours and the number of fits skipped |
| `"full"` | every neighbour | as above, and each certificate is checked against its refit |

With the scikit-learn backend, refits reuse the core distances from the single nearest-neighbour query and run
scikit-learn's own exact spanning-tree routine, so the labels equal those of `sklearn.cluster.HDBSCAN`. The
`hdbscan` backend refits with the package itself.

## Certified range

```python
from minpts_check import certified_range

r = certified_range(X, min_samples=11, min_cluster_size=10, max_steps=10)
r["top_min_samples"], r["U_share"], r["checks"]
```

Every `min_samples` in `(11, r["top_min_samples"]]` is certified to keep the labels of all points outside the
set `r["U"]`. The same search runs inside `check_min_samples(..., range_search="binary")`, where it adds a line
to the summary. In `"verify"` mode, steps inside a certified range with a small `U` skip their refit. In
`"full"` mode, the range claim is checked against every refit inside the range.

The search is valid for two reasons, which the docstring of `certified_range` explains in full.

- Soundness does not depend on the search. Only tops where the certificate was evaluated and held are
  reported, and a certificate at one top covers every value below it.
- The search finds the largest certified top when the certificate is monotone in the top. Version 0.2.0
  builds the upper hierarchy on one fixed edge set: the observed spanning tree and the k-NN graph at
  `min_samples + max_steps`. With that set fixed, every edge weight grows with the top, so every merge height
  grows too. Version 0.1.0 used the k-NN graph of each step, so its edge set changed with the top and this
  argument did not apply.

The test suite compares binary search with a linear scan of every top on random data and on the paper's stored
runs. `search="linear"` is also available.

## Ties

With the mutual reachability distance, many spanning-tree edges share their weight, because their weight is
the core distance of a shared point. A graph with tied weights has several minimum spanning trees.
scikit-learn and the `hdbscan` package condense a binary tree, so their labels can depend on which spanning
tree they built and on the order of the rows. The definition of HDBSCAN* removes edges of equal weight
simultaneously (Campello et al. 2015, Sec. 3.2). That gives one n-ary tree for all spanning trees, and so one
set of labels.

| `ties=` | labels reported | certificate |
|---|---|---|
| `"library"` (default) | the backend's | evaluated |
| `"check"` | the backend's, plus the canonical labels at the chosen value and their agreement (`report.ties`) | evaluated |
| `"canonical"` | canonical, for the chosen value and every refit | not evaluated (it covers the backend's binary tree) |

`canonical_labels(X, min_samples, min_cluster_size)` returns the canonical labels for one value. With pairwise
distinct edge weights they equal the library's labels. On the 10,000-point benchmark examples the two agreed
with ARI of 0.998 or more. On small data sets, and on data with many exactly equal distances, the gap can be
larger.

## Sweeps over min_samples

```python
from minpts_check import sweep

labels = sweep(X, range(5, 31))          # {min_samples: canonical labels}
```

`sweep` runs one nearest-neighbour query and one exact fit, at the largest value. Then, going down the values,
it computes each spanning tree from the tree of the next larger value and its k-NN graph. That union contains
an exact minimum spanning tree (Neto et al. 2022, Theorem 1). The labels are the canonical ones, because a
different exact spanning tree could change the library's labels. Inside `check_min_samples`, `refit="chain"`
does the same for the refits (it needs `ties="canonical"`). With one thread, five values took 36 s against 156 s
for five scikit-learn fits on a 100,000-point example in 5-D. On the two 10,000-point examples, 13 values took
1.0 to 1.7 s against 4.2 to 5.6 s.

## Root preference

`report.root_preference["flag"]` is true when the excess-of-mass extraction would select the root if it were
allowed to, so that `allow_single_cluster=True` would return a single cluster. It costs nothing. It is an
empirical signal, not a guarantee, and it is common. In an exploratory screen by the author (423 runs on 47
benchmark data sets, `min_cluster_size` equal to `min_samples`), 227 runs had the flag. In those runs, the step
to the next value of `min_samples` had ARI below 0.8 in 30.8% of cases, against 14.3% in the other 196 runs.

## Conventions

scikit-learn's `min_samples` counts the point itself and the `hdbscan` package's does not, so scikit-learn's
`min_samples = m + 1` equals the `hdbscan` package's `min_samples = m`. Report and input use the convention of the
chosen backend, and the report says which convention that is. `fast_plscan` follows `hdbscan`.

By default `min_cluster_size` moves with `min_samples` when the two are equal in the backend's convention, which is
what both libraries do when `min_samples` is left unset. Pass `coupled=False` to hold `min_cluster_size` fixed.

## Scope

As in the paper, `s_hat` and the certificate cover increases of `min_samples`, the excess-of-mass extraction, an
exact spanning tree and the euclidean metric.

- Lower neighbours are reported only when they are refitted.
- Approximate trees: the `hdbscan` package uses an approximate spanning tree by default. On the paper's
  exploratory data that tree changed the partition on 29.3% of steps, against 23.4% for the exact tree. The tool
  warns and skips the certificate; pass `approx_min_span_tree=False`.
- Other settings: with leaf extraction, `cluster_selection_epsilon` or `allow_single_cluster`, `s_hat` is
  reported but the certificate is not evaluated.
- Tolerance set: a certified step can still have ARI below 0.8 when `U` is large, so read `|U|` with the
  verdict.

## Note for BERTopic users

BERTopic clusters the UMAP embedding of the documents with HDBSCAN (the `hdbscan` package,
`min_samples = min_cluster_size` by default). Check the clustering on that embedding:

```python
topic_model = BERTopic()
topics, _ = topic_model.fit_transform(docs)
emb = topic_model.umap_model.embedding_          # the reduced embedding HDBSCAN clustered
hd = topic_model.hdbscan_model
rep = check_min_samples(emb, hd.min_samples or hd.min_cluster_size, hd.min_cluster_size, backend="hdbscan",
                        approx_min_span_tree=False)
rep.summary()
```

The check refits HDBSCAN on the embedding with an exact tree, so its labels can differ slightly from the topics of
a model fitted with the default approximate tree. In the paper, two UMAP seeds at the same `min_samples` usually gave
different BERTopic partitions, so fix the UMAP seed before comparing values of `min_samples`.

## Tests

```
pip install .[test]
pytest tests
```

The oracle tests compare `s_hat`, the at-risk flags, the certificate and the certified ranges with the stored
outputs of the paper's analysis code. They are skipped when those outputs are not present (`MINPTS_ORACLE_DIR`).
`python tests/bench_10k.py` times the modes and options on two 10,000-point examples.

## Citation

Software: Coutinho, F. P. (2026). minpts-check (version 0.2.0). Zenodo. https://doi.org/10.5281/zenodo.24094267 (all versions: https://doi.org/10.5281/zenodo.24092807)
(the record of version 0.1.0; Zenodo lists every version there)

Method: Coutinho, F. P. (2026). Why HDBSCAN*'s flat clustering changes with minPts: shrinking clusters, a one-run predictor
and a certificate. Zenodo. https://doi.org/10.5281/zenodo.24039671

Ties: Campello, R. J. G. B., Moulavi, D., Zimek, A., Sander, J. (2015). Hierarchical density estimates for data
clustering, visualization, and outlier detection. ACM Transactions on Knowledge Discovery from Data 10(1), 5.
https://doi.org/10.1145/2733381

Chained spanning trees: Neto, A. C. A., Naldi, M. C., Campello, R. J. G. B., Sander, J. (2022). CORE-SG: Efficient
computation of multiple MSTs for density-based methods. IEEE ICDE 2022, 951-964. https://doi.org/10.1109/ICDE53745.2022.00076

## Licence

MIT, Felipe Provezano Coutinho.
