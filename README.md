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

## Install

```
pip install git+https://github.com/provezano/minpts-check            # numpy, scikit-learn
pip install "minpts-check[hdbscan] @ git+https://github.com/provezano/minpts-check"   # also the hdbscan package
```

## Usage

```python
from minpts_check import check_min_samples

rep = check_min_samples(X, min_samples=11, min_cluster_size=10, backend="sklearn", steps=2)
rep.summary()        # plain-text table and a one-line verdict
d = rep.to_dict()    # the same as a dict (tolerance sets as point indices)
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

The oracle tests compare `s_hat`, the at-risk flags and the certificate with the stored outputs of the paper's
analysis code. They are skipped when those outputs are not present (`MINPTS_ORACLE_DIR`).

## Citation

Software: Coutinho, F. P. (2026). minpts-check (version 0.1.0). Zenodo. https://doi.org/10.5281/zenodo.24092808

Method: Coutinho, F. P. (2026). Why HDBSCAN*'s flat clustering changes with minPts: shrinking clusters, a one-run predictor
and a certificate. Zenodo. https://doi.org/10.5281/zenodo.24039671

## Licence

MIT, Felipe Provezano Coutinho.
