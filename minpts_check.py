"""minpts_check: is an HDBSCAN* flat clustering stable under a step of min_samples?

One call, ``check_min_samples``, applies the routine of Coutinho (2026), "Why HDBSCAN*'s flat clustering changes
with minPts: shrinking clusters, a one-run predictor and a certificate" (doi:10.5281/zenodo.24039671):

* the one-run predictor s_hat(C): the points of a selected cluster C whose core distance at the next value of
  min_samples is still within C's birth radius; C is at risk when s_hat(C) < the next min_cluster_size;
* the certificate (practical form, Corollary 2 of the paper): from the observed run and one k-NN query it proves that
  an increase of min_samples changes no label outside an explicit tolerance set U;
* optional refits of the neighbouring values, with the ARI between neighbouring outputs.

Added in 0.2.0:

* ``certified_range`` (and ``check_min_samples(..., range_search="binary")``): the largest value of min_samples up
  to which the range form of the certificate holds, found with O(log) certificate checks (see ``certified_range``
  for why the search is valid);
* tie-merged canonical labels (``ties="canonical"``, ``canonical_labels``): edges of equal weight are removed
  simultaneously, as in the definition of HDBSCAN* (Campello et al. 2015, Sec. 3.2), so the labels do not depend
  on which of several exact spanning trees a library happens to build; ``ties="check"`` compares the two;
* ``sweep`` and ``refit="chain"``: labels for many values of min_samples from one exact fit, with each spanning
  tree computed from the next larger one (the containment theorem of CORE-SG, Neto et al. 2022, applied value by
  value);
* a root-preference flag (empirical diagnostic, see ``MinPtsReport.root_preference``).

Conventions.  The paper writes k for the neighbour count that includes the point itself.  scikit-learn's
``min_samples`` is k; the ``hdbscan`` package's ``min_samples`` is k - 1 (sklearn m + 1 == hdbscan m).  Everything is
computed in k internally and reported in the convention of the chosen backend.
"""
import math
from dataclasses import dataclass, field
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

__all__ = ["check_min_samples", "certified_range", "canonical_labels", "sweep", "MinPtsReport"]
__version__ = "0.2.0"

INF = np.inf
CITATION = "Coutinho (2026), doi:10.5281/zenodo.24039671"
APPROX_WARNING = ("approximate spanning tree: on the paper's exploratory data the approximate tree changed the "
                  "partition on 29.3% of min_samples steps against 23.4% for the exact tree; the certificate is "
                  "not evaluated. Pass approx_min_span_tree=False for an exact tree.")
_OFFSET = {"sklearn": 0, "hdbscan": 1}          # k = min_samples + offset
_SK_OK = {"cluster_selection_method", "allow_single_cluster", "cluster_selection_epsilon", "max_cluster_size",
          "alpha", "algorithm", "leaf_size", "metric", "n_jobs"}


# ============================================================ fits (normalised to one internal record)
def _knn(X, kmax, algorithm="auto", leaf_size=40):
    """distances to the kmax nearest points, the point itself included (column k-1 = core distance for k)."""
    algo = algorithm if algorithm in ("kd_tree", "ball_tree", "brute") else "kd_tree"
    nn = NearestNeighbors(n_neighbors=kmax, algorithm=algo, leaf_size=leaf_size).fit(X)
    return nn.kneighbors(X, kmax)


def _fit_sklearn(X, k, mcs, core, kw):
    """scikit-learn's own exact path (Prim on the data matrix) fed with the cached core distances; this is what
    sklearn.cluster.HDBSCAN.fit runs for a euclidean metric, without its own k-NN query."""
    from sklearn.cluster._hdbscan._linkage import mst_from_data_matrix
    from sklearn.cluster._hdbscan._tree import _condense_tree, tree_to_labels
    from sklearn.cluster._hdbscan.hdbscan import _process_mst
    from sklearn.metrics import DistanceMetric
    mst = mst_from_data_matrix(X, np.ascontiguousarray(core), DistanceMetric.get_metric("euclidean"),
                               float(kw.get("alpha", 1.0)))
    slt = _process_mst(mst)
    lab, _ = tree_to_labels(slt, mcs, kw.get("cluster_selection_method", "eom"),
                            kw.get("allow_single_cluster", False), float(kw.get("cluster_selection_epsilon", 0.0)),
                            kw.get("max_cluster_size", None))
    ct = _condense_tree(slt, mcs)
    return dict(lab=np.asarray(lab, np.int64), ct=dict(parent=ct["parent"], child=ct["child"],
                lambda_val=ct["value"], child_size=ct["cluster_size"]),
                mst=(mst["current_node"].astype(np.int64), mst["next_node"].astype(np.int64), mst["distance"]))


def _fit_hdbscan(X, k, mcs, kw):
    import hdbscan
    md = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=k - 1, gen_min_span_tree=True, **kw).fit(X)
    ct = md.condensed_tree_.to_numpy()
    mst = md.minimum_spanning_tree_.to_numpy()
    return dict(lab=md.labels_.astype(np.int64), ct={f: ct[f] for f in ("parent", "child", "lambda_val", "child_size")},
                mst=(mst[:, 0].astype(np.int64), mst[:, 1].astype(np.int64), mst[:, 2]))


def _tree_info(backend, kw, d):
    """(extraction, tree description, exact?) of a fit with these settings."""
    ext = kw.get("cluster_selection_method", "eom")
    if backend == "sklearn":
        return ext, "exact minimum spanning tree (Prim)", True
    algo = kw.get("algorithm", "best")
    boruvka = algo.startswith("boruvka") or (algo == "best" and d <= 60)
    if boruvka and kw.get("approx_min_span_tree", True):
        return ext, f"approximate minimum spanning tree ({algo}, approx_min_span_tree=True)", False
    return ext, f"exact minimum spanning tree ({algo})", True


# ============================================================ condensed tree and hierarchies (paper's checker)
class Dendro:
    """Kruskal single-linkage dendrogram on n leaves from edge list (u, v, w).  Internal nodes n..2n-2."""

    def __init__(self, n, u, v, w):
        o = np.argsort(w, kind="stable")
        u, v, w = np.asarray(u)[o], np.asarray(v)[o], np.asarray(w, float)[o]
        par = np.full(2 * n - 1, -1, np.int64)
        h = np.zeros(2 * n - 1)
        sz = np.ones(2 * n - 1, np.int64)
        left = np.full(2 * n - 1, -1, np.int64); right = np.full(2 * n - 1, -1, np.int64)
        uf = np.arange(2 * n - 1)
        top = np.arange(n)

        def find(a):
            r = a
            while uf[r] != r:
                r = uf[r]
            while uf[a] != r:
                uf[a], a = r, uf[a]
            return r
        nxt = n
        for a, b, ww in zip(u.tolist(), v.tolist(), w.tolist()):
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            ta, tb = top[ra], top[rb]
            par[ta] = nxt; par[tb] = nxt; h[nxt] = ww; sz[nxt] = sz[ta] + sz[tb]
            left[nxt], right[nxt] = ta, tb
            uf[rb] = ra; top[ra] = nxt; nxt += 1
        if nxt != 2 * n - 1:
            raise ValueError("edge set does not span")
        self.n, self.par, self.h, self.sz, self.left, self.right = n, par, h, sz, left, right
        self.root = 2 * n - 2
        start = np.zeros(2 * n - 1, np.int64); end = np.zeros(2 * n - 1, np.int64)
        order = np.empty(n, np.int64); pos = 0
        stack = [(self.root, 0)]
        while stack:
            node, st = stack.pop()
            if node < n:
                start[node] = pos; order[pos] = node; pos += 1; end[node] = pos
                continue
            if st == 0:
                start[node] = pos
                stack.append((node, 1)); stack.append((right[node], 0)); stack.append((left[node], 0))
            else:
                end[node] = pos
        self.start, self.end, self.order = start, end, order
        self._paths = {}

    def path(self, x):
        p = self._paths.get(x)
        if p is None:
            nodes = []
            a = self.par[x]
            while a >= 0:
                nodes.append(a); a = self.par[a]
            nodes = np.array(nodes, np.int64)
            p = (nodes, self.h[nodes], self.sz[nodes])
            self._paths[x] = p
        return p

    def node_at(self, x, r, closed=True):
        """highest ancestor of leaf x with height <= r (closed) or < r (open); leaf if none."""
        nodes, hh, _ = self.path(x)
        i = np.searchsorted(hh, r, "right" if closed else "left") - 1
        return x if i < 0 else int(nodes[i])

    def leaves(self, node):
        return self.order[self.start[node]:self.end[node]]

    def core_node(self, mc):
        """for every leaf: (id, height) of its lowest ancestor of size >= mc (-1, inf if none)."""
        cn = np.full(2 * self.n - 1, -1, np.int64)
        for v in range(2 * self.n - 2, -1, -1):
            p = self.par[v]
            cn[v] = v if (self.sz[v] >= mc and v >= self.n) else (cn[p] if p >= 0 else -1)
        cn = cn[:self.n]
        return cn, np.where(cn >= 0, self.h[np.maximum(cn, 0)], INF)


def parse_condensed(ct, n):
    par, ch = np.asarray(ct["parent"], np.int64), np.asarray(ct["child"], np.int64)
    lam, sz = np.asarray(ct["lambda_val"], float), np.asarray(ct["child_size"], np.int64)
    root = int(par.min())
    nodes = sorted(set(par.tolist()) | set(ch[ch >= n].tolist()))
    birth = {root: 0.0}
    for c_, l_ in zip(ch[ch >= n].tolist(), lam[ch >= n].tolist()):
        birth[c_] = l_
    parent = {c_: p_ for p_, c_ in zip(par[ch >= n].tolist(), ch[ch >= n].tolist())}
    kids = {c: [] for c in nodes}
    for p_, c_ in zip(par[ch >= n].tolist(), ch[ch >= n].tolist()):
        kids[p_].append(c_)
    S = {c: 0.0 for c in nodes}; death = {c: 0.0 for c in nodes}
    for p_, l_, s_ in zip(par.tolist(), lam.tolist(), sz.tolist()):
        S[p_] += (l_ - birth[p_]) * s_
        if l_ > death[p_]:
            death[p_] = l_
    ptkids = {c: [] for c in nodes}
    ptpar = {}
    for p_, c_ in zip(par[ch < n].tolist(), ch[ch < n].tolist()):
        ptkids[p_].append(c_); ptpar[c_] = p_
    pts = {}
    for c in sorted(nodes, reverse=True):
        a = list(ptkids[c])
        for k in kids[c]:
            a.extend(pts[k])
        pts[c] = a
    pts = {c: np.array(v, np.int64) for c, v in pts.items()}
    return dict(root=root, nodes=nodes, birth=birth, death=death, parent=parent, kids=kids, S=S, pts=pts,
                ptpar=ptpar)


def selected_nodes(T, lab):
    """condensed-tree node of each label (label order): the node whose points are exactly that label's points."""
    out = []
    for j in range(int(lab.max()) + 1):
        mem = np.flatnonzero(lab == j)
        c = T["ptpar"].get(int(mem[0]))
        while c is not None and len(T["pts"][c]) < len(mem):
            c = T["parent"].get(c)
        if c is None or len(T["pts"][c]) != len(mem) or not np.array_equal(np.sort(T["pts"][c]), mem):
            return None
        out.append(c)
    return out


def anchor_ultra(D, xs):
    """ultrametric (merge heights) of hierarchy D restricted to the anchor leaves xs."""
    A = len(xs); H = np.zeros((A, A))
    pos = {int(x): i for i, x in enumerate(xs)}
    lst = {int(x): [pos[int(x)]] for x in xs}
    for v in range(D.n, 2 * D.n - 1):
        l, r = int(D.left[v]), int(D.right[v])
        a, b = lst.pop(l, None), lst.pop(r, None)
        if a and b:
            H[np.ix_(a, b)] = D.h[v]; H[np.ix_(b, a)] = D.h[v]
            lst[v] = a + b
        elif a or b:
            lst[v] = a or b
    return H


def blocks_at(D, xs, P):
    """node ids and sizes of the closed blocks of leaves xs at radii P: arrays (len(xs), len(P))."""
    ids = np.empty((len(xs), len(P)), np.int64); szs = np.empty((len(xs), len(P)), np.int64)
    for i, x in enumerate(xs):
        nodes, hh, ss = D.path(int(x))
        k = np.searchsorted(hh, P, "right") - 1
        ids[i] = np.where(k < 0, x, nodes[np.maximum(k, 0)])
        szs[i] = np.where(k < 0, 1, ss[np.maximum(k, 0)])
    return ids, szs


def distinct_sum(ids, szs):
    o = np.argsort(ids, axis=0, kind="stable")
    si = np.take_along_axis(ids, o, 0); ss = np.take_along_axis(szs, o, 0)
    first = np.ones_like(si, bool); first[1:] = si[1:] != si[:-1]
    return (ss * first).sum(0)


def breakpoints(r_lo, r_hi, paths):
    if not r_lo < r_hi:
        return np.array([])
    bp = [np.array([r_lo])]
    for (_, hh, _) in paths:
        bp.append(hh[(hh > r_lo) & (hh < r_hi)])
    return np.unique(np.concatenate(bp))


def lam_lengths(P, r_hi):
    nxt = np.append(P[1:], r_hi)
    return 1.0 / P - np.where(np.isinf(nxt), 0.0, 1.0 / nxt)


# ============================================================ certificate (Theorem 1, practical form)
def certify(T, selected, Kd, Fd, mc, mp, mb, rel_tol=1e-6, phantom=None, max_classes=12, refine_budget=400):
    """General certificate of the paper (Theorem 1).  mc: every anchor core must keep >= mc points; mp: phantom
    blocks must stay below mp; mb: block threshold of the descendant bound.  Returns dict(cert, reason, U, anchor)."""
    n = Kd.n
    root, kids, parent = T["root"], T["kids"], T["parent"]
    res = dict(cert=False, reason=None)
    stats = dict(refined=0, refined_ok=0, groups=0)
    if len(selected) < 2:
        res["reason"] = "fewer than two selected clusters"; return res
    sel = set(selected)
    tops = set()
    for L in selected:
        c = L
        while c != root:
            tops.add(c); c = parent[c]
    tops.add(root)
    internal = [X for X in tops if X not in sel]
    if any(k not in tops for X in internal for k in kids[X]):
        res["reason"] = "selection is not a cut"; return res
    cnF, hcF = Fd.core_node(mc)
    anchor = {}
    for L in selected:
        P = T["pts"][L]
        ind = np.zeros(n, np.int64); ind[P] = 1
        cs = np.concatenate([[0], np.cumsum(ind[Fd.order])])
        cn = np.maximum(cnF[P], 0)
        ok = (cnF[P] >= 0) & (cs[Fd.end[cn]] - cs[Fd.start[cn]] == Fd.sz[cn])
        if not ok.any():
            res["reason"] = "no anchor keeps the next min_cluster_size (cluster may cross the size threshold)"
            return res
        Pc = P[ok]
        anchor[L] = int(Pc[np.argmin(hcF[Pc])])
    order = sorted(selected)
    xs = np.array([anchor[L] for L in order]); A = len(xs)
    aidx = {L: i for i, L in enumerate(order)}
    HK = anchor_ultra(Kd, xs); HF = anchor_ultra(Fd, xs)
    if np.any(HF < HK * (1 - 1e-12)):
        res["reason"] = "internal error: lower hierarchy above the observed one"; return res
    below = {}
    for X in sorted(tops, reverse=True):
        below[X] = [aidx[X]] if X in sel else sorted(sum((below[k] for k in kids[X]), []))
    rsplit = {}
    for X in internal:
        ks = kids[X]; rs = 0.0
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                rs = max(rs, HK[below[ks[i]][0], below[ks[j]][0]])
        rsplit[X] = rs
    rtop = {X: (INF if X == root else rsplit[parent[X]]) for X in tops}
    mask = np.zeros(n, bool)

    def phantom_set(kx, r_k, r_core, q, core_x):
        if phantom is None or stats["refined"] >= refine_budget:
            return False
        stats["refined"] += 1
        mask[:] = False
        mask[Kd.leaves(Kd.node_at(kx, r_k))] = True
        for y in core_x:
            mask[Fd.leaves(Fd.node_at(int(y), r_core))] = False
        ok = phantom(np.flatnonzero(mask), q) < mp
        stats["refined_ok"] += int(ok)
        return ok

    def first_bad(kx, lo, hi, core_x):
        def rec(a, b):
            if phantom_set(kx, lo[a:b].max(), lo[a:b].min(), hi[a:b].max(), core_x):
                return None
            if b - a == 1:
                return a
            mid = (a + b) // 2
            r1 = rec(a, mid)
            return r1 if r1 is not None else rec(mid, b)
        return rec(0, len(lo)) if len(lo) else None
    # (C) inside every unselected ancestor: cores keep >= mc points, no phantom block reaches mp
    for X in internal:
        ax = xs[below[X]]; x0 = int(ax[0])
        r_lo, r_hi = rsplit[X], rtop[X]
        P = breakpoints(r_lo, r_hi, [Kd.path(x0)] + [Fd.path(int(x)) for x in ax])
        if len(P) == 0:
            continue
        ids, szs = blocks_at(Fd, ax, P)
        if szs.min() < mc:
            res["reason"] = "a certified core drops below the next min_cluster_size"; return res
        sK = blocks_at(Kd, [x0], P)[1][0]
        bad = np.flatnonzero(sK - distinct_sum(ids, szs) >= mp)
        if len(bad):
            up = np.append(P[1:], r_hi)
            for rn in np.split(bad, np.flatnonzero(np.diff(bad) > 1) + 1):
                if first_bad(x0, P[rn], up[rn], ax) is not None:
                    res["reason"] = "a new cluster could reach the size threshold (phantom check)"; return res
    # trunks of the selected clusters: certified stability S_lo
    Slo, lamT = {}, {}
    for L in selected:
        x = anchor[L]; r_hi = rtop[L]; r_d = (1.0 / T["death"][L]) * (1 + 1e-12)
        P = breakpoints(r_d, r_hi, [Kd.path(x), Fd.path(x)])
        if len(P) == 0:
            Slo[L] = 0.0; lamT[L] = 1.0 / r_hi; continue
        P = P[::-1]
        sF = blocks_at(Fd, [x], P)[1][0]; sK = blocks_at(Kd, [x], P)[1][0]
        ub = np.append(r_hi, P[:-1])
        good = np.ones(len(P), bool)
        bad_core = np.flatnonzero(sF < mc)
        stop = bad_core[0] if len(bad_core) else len(P)
        bad = np.flatnonzero(sK - sF >= mp)
        bad = bad[bad < stop]
        if len(bad):
            for rn in np.split(bad, np.flatnonzero(np.diff(bad) > 1) + 1):
                fb = first_bad(x, P[rn], ub[rn], [x])
                if fb is not None:
                    stop = rn[fb]; break
        good[stop:] = False
        ll = 1.0 / P - np.where(np.isinf(ub), 0.0, 1.0 / ub)
        Slo[L] = float((sF * ll)[good].sum())
        lamT[L] = 1.0 / (P[stop - 1] if stop > 0 else r_hi)
    # descendant bound D_hi
    internal_nodes = np.arange(n, 2 * n - 1)
    Dhi = {}
    for L in selected:
        vL = Kd.node_at(anchor[L], rtop[L], closed=False)
        if vL < n:
            Dhi[L] = 0.0; continue
        s0, e0 = Kd.start[vL], Kd.end[vL]
        mm = (Kd.start[internal_nodes] >= s0) & (Kd.end[internal_nodes] <= e0) & (Kd.sz[internal_nodes] >= mb)
        v = internal_nodes[mm]
        hp = np.where(Kd.par[v] >= 0, Kd.h[np.maximum(Kd.par[v], 0)], INF)
        lo = np.maximum(np.where(np.isinf(hp), 0.0, 1.0 / hp), lamT[L])
        Dhi[L] = float((Kd.sz[v] * np.clip(1.0 / Kd.h[v] - lo, 0, None)).sum())
    worst = INF
    for L in selected:                                  # (D1)
        g = Slo[L] - Dhi[L]; rel = g / max(Slo[L], 1e-300)
        worst = min(worst, rel)
        if not (g > 0 and rel > rel_tol):
            res["reason"] = "descendant bound (D1): a selected cluster could be replaced by its sub-clusters"
            return res
    slo_vec = np.array([Slo[L] for L in order])
    for X in internal:                                  # (D2) every feasible group of anchors
        r0 = rsplit[X]; mem = below[X]
        cid = np.full(A, -1, np.int64)
        for ci, k in enumerate(kids[X]):
            cid[below[k]] = ci
        fid = np.array([Fd.node_at(int(xs[i]), r0) for i in mem])
        classes = [list(np.array(mem)[fid == f]) for f in np.unique(fid)]
        if len(classes) > max_classes:
            res["reason"] = "too many anchor groups to enumerate"; return res
        for sub in range(1, 1 << len(classes)):
            G = sorted(sum((classes[c] for c in range(len(classes)) if sub >> c & 1), []))
            if len(set(cid[G].tolist())) < 2 or (X == root and len(G) == A):
                continue
            notG = np.setdiff1d(np.arange(A), G)
            sep = HF[np.ix_(G, notG)].min() if len(notG) else INF
            if not sep > r0:
                continue
            stats["groups"] += 1
            xg = int(xs[G[0]])
            others = [int(xs[j]) for j in notG if HK[G[0], j] < sep]
            P = breakpoints(r0, sep, [Kd.path(xg)] + [Fd.path(y) for y in others])
            sK = blocks_at(Kd, [xg], P)[1][0]
            sub_ = np.zeros(len(P))
            if others:
                ids, szs = blocks_at(Fd, others, P)
                js = np.array([j for j in notG if HK[G[0], j] < sep])
                inblk = HK[G[0], js][:, None] <= P[None, :]
                ids = np.where(inblk, ids, -1 - np.arange(len(others))[:, None]); szs = np.where(inblk, szs, 0)
                sub_ = distinct_sum(ids, szs)
            Shi = float((np.maximum(sK - sub_, 0) * lam_lengths(P, sep)).sum())
            lo = float(slo_vec[G].sum())
            rel = (lo - Shi) / max(lo, 1e-300)
            worst = min(worst, rel)
            if not (lo > Shi and rel > rel_tol):
                res["reason"] = "group margin (D2): EOM could merge selected clusters"; return res
    U = np.zeros(n, bool)                               # tolerance set
    for L in selected:
        i = aidx[L]; x = anchor[L]; r_b = rtop[L]
        others = np.delete(np.arange(A), i)
        o = np.zeros(n, bool)
        o[Kd.leaves(Kd.node_at(x, HF[i, others].min(), closed=True))] = True
        for j in others:
            o[Fd.leaves(Fd.node_at(int(xs[j]), r_b, closed=False))] = False
        o[Fd.leaves(Fd.node_at(x, r_b, closed=False))] = False
        U |= o
    res.update(cert=True, reason="ok", U=U, anchor={L: anchor[L] for L in selected}, worst_rel=worst)
    return res


def verify(res, selected, lab0, lab1):
    """claims of the theorem against a true run: same number of clusters, anchors in distinct clusters, labels
    agree (after matching) outside U."""
    mp = {j: int(lab1[res["anchor"][L]]) for j, L in enumerate(selected)}
    ok_count = int(lab1.max() + 1) == len(selected)
    ok_anchor = all(v >= 0 for v in mp.values()) and len(set(mp.values())) == len(mp)
    exp = np.array([mp.get(int(l), -99) if l >= 0 else -1 for l in lab0])
    bad = (~res["U"]) & (exp != lab1)
    return bool(ok_count and ok_anchor and bad.sum() == 0), int(bad.sum())


def _lower_hierarchy(X, mst, dist, nbr, k2):
    """F of Corollary 2: observed MST edges plus k2-NN edges, weighted by the mutual reachability at k2."""
    n = len(X); c1 = dist[:, k2 - 1]
    aa = np.concatenate([np.repeat(np.arange(n), k2 - 1), mst[0]])
    bb = np.concatenate([nbr[:, 1:k2].ravel(), mst[1]])
    lo, hi = np.minimum(aa, bb), np.maximum(aa, bb)
    keep = lo != hi
    kq = np.unique(lo[keep] * n + hi[keep]); lo, hi = kq // n, kq % n
    ww = np.maximum(np.maximum(c1[lo], c1[hi]), np.linalg.norm(X[lo] - X[hi], axis=1))
    G = minimum_spanning_tree(coo_matrix((ww, (lo, hi)), shape=(n, n)).tocsr()).tocoo()
    return Dendro(n, G.row, G.col, G.data)


def _phantom(X, cc, cap=2500):
    """Lemma 2 (phantom bound): largest component, at radius q, of the d_k2 graph on a point set R (crude |R| above the cap)."""
    def f(R, q):
        R2 = R[cc[R] <= q]
        if len(R2) <= 1 or len(R2) > cap:
            return len(R2)
        pr = cKDTree(X[R2]).query_pairs(q, output_type="ndarray")
        if len(pr) == 0:
            return 1
        A = coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(R2), len(R2)))
        return int(np.bincount(connected_components(A, directed=False)[1]).max())
    return f


# ============================================================ fixed edge set and range search (0.2.0)
def _pairs(n, a, b):
    """unique undirected pairs (lo < hi) of the edge list (a, b), self-loops removed."""
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    keep = lo != hi
    key = np.unique(lo[keep] * n + hi[keep])
    return key // n, key % n


def _mreach(X, core, a, b):
    """mutual reachability distance of the pairs (a, b) at the core distances ``core`` (one formula everywhere)."""
    return np.maximum(np.maximum(core[a], core[b]), np.linalg.norm(X[a] - X[b], axis=1))


class _FixedEdges:
    """the edge set E = (observed spanning tree) + (k-NN graph at kE), fixed for every top of a range search.

    F(t), the lower hierarchy of Corollary 2 for the top t, is the single-linkage hierarchy of E weighted by the
    mutual reachability at t.  v0.1 used the k-NN graph at the step itself, so E grew with t; with E fixed, every
    edge weight is non-decreasing in t, which is what the range search relies on (see ``certified_range``)."""

    def __init__(self, X, mst, nbr, kE):
        n = len(X)
        a = np.concatenate([np.repeat(np.arange(n), kE - 1), mst[0]])
        b = np.concatenate([nbr[:, 1:kE].ravel(), mst[1]])
        self.n = n
        self.lo, self.hi = _pairs(n, a, b)
        self.len = np.linalg.norm(X[self.lo] - X[self.hi], axis=1)

    def hierarchy(self, core):
        w = np.maximum(np.maximum(core[self.lo], core[self.hi]), self.len)
        G = minimum_spanning_tree(coo_matrix((w, (self.lo, self.hi)), shape=(self.n, self.n)).tocsr()).tocoo()
        return Dendro(self.n, G.row, G.col, G.data)


def _search_top(ok, max_u, search):
    """largest u in 1..max_u with ok(u) (0 if none).  'linear' evaluates every u; 'binary' doubles u while ok holds,
    then bisects between the last success and the first failure.  Returns (u*, {u: ok(u)} of the evaluated u)."""
    memo = {}

    def f(u):
        if u not in memo:
            memo[u] = bool(ok(u))
        return memo[u]
    if search == "linear":
        for u in range(1, max_u + 1):
            f(u)
        return max([u for u, v in memo.items() if v], default=0), memo
    lo, u = 0, 1
    while u <= max_u and f(u):
        lo, u = u, 2 * u
    if lo == max_u:
        return lo, memo
    hi = u
    if u > max_u:                       # doubling stepped over max_u: test max_u itself
        if f(max_u):
            return max_u, memo
        hi = max_u
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if f(mid):
            lo = mid
        else:
            hi = mid
    return lo, memo


# ============================================================ tie-merged canonical condensing (0.2.0)
def _fsum(vals):
    """order-independent sum (exactly rounded when finite), so stabilities do not depend on the point order."""
    if not vals:
        return 0.0
    a = np.asarray(vals, float)
    return math.fsum(a.tolist()) if np.isfinite(a).all() else float(np.sort(a).sum())


def _canonical(n, u, v, w, mcs):
    """HDBSCAN* flat clustering (EOM) on the n-ary single-linkage tree of a spanning tree (u, v, w).

    All edges of equal weight are merged at once, so the tree is that of the ultrametric of the graph, the same for
    every minimum spanning tree of it (Campello et al. 2015, Sec. 3.2: "in case of ties, edges are removed
    simultaneously").  Condensing and EOM follow the libraries: a node with two or more children of size >= mcs
    splits into new clusters born at lambda = 1/w; with one such child the cluster continues; points of smaller
    children leave at lambda; stability = sum of (lambda - birth) over leaving points and child clusters; a cluster
    is replaced by its children when their summed best stability is strictly larger; the root is not eligible.
    With pairwise distinct weights the result equals the libraries' binary condensing.

    Returns dict(labels, clusters=[dict(birth_lambda, members)] in label order, root_preferred, root_stability,
    children_stability).  Labels are numbered by the first point of each cluster."""
    o = np.argsort(w, kind="stable")
    ul = np.asarray(u, np.int64)[o].tolist(); vl = np.asarray(v, np.int64)[o].tolist()
    w = np.asarray(w, float)[o]; wl = w.tolist()
    uf = list(range(n)); top = list(range(n)); size = [1] * n
    children, height = [], []

    def find(a):
        r = a
        while uf[r] != r:
            r = uf[r]
        while uf[a] != r:
            uf[a], a = r, uf[a]
        return r
    cuts = (np.flatnonzero(np.diff(w) != 0) + 1).tolist()
    for i, j in zip([0] + cuts, cuts + [len(wl)]):
        if j - i == 1:
            a, b = find(ul[i]), find(vl[i])
            if a == b:
                continue
            children.append([top[a], top[b]]); height.append(wl[i]); size.append(size[top[a]] + size[top[b]])
            uf[b] = a; top[a] = n + len(children) - 1
            continue
        loc = {}

        def lf(x):
            while loc[x] != x:
                loc[x] = loc[loc[x]]; x = loc[x]
            return x
        for e in range(i, j):
            a, b = find(ul[e]), find(vl[e])
            if a == b:
                continue
            loc.setdefault(a, a); loc.setdefault(b, b)
            ra, rb = lf(a), lf(b)
            if ra != rb:
                loc[ra] = rb
        groups = {}
        for r in loc:
            groups.setdefault(lf(r), []).append(r)
        for g in groups.values():
            children.append([top[r] for r in g]); height.append(wl[i]); size.append(sum(size[top[r]] for r in g))
            for r in g[1:]:
                uf[r] = g[0]
            top[g[0]] = n + len(children) - 1
    if not children or size[-1] != n:
        raise ValueError("edge set does not span")

    def leaves(x):
        out, st = [], [x]
        while st:
            y = st.pop()
            if y < n:
                out.append(y)
            else:
                st.extend(children[y - n])
        return out
    birth, cpar, kids, contrib = [0.0], [-1], [[]], [[]]
    fall = np.full(n, -1, np.int64)
    stack = [(n + len(children) - 1, 0)]
    while stack:
        node, c = stack.pop()
        h = height[node - n]
        L = 1.0 / h if h > 0 else INF
        ch = children[node - n]
        big = [x for x in ch if size[x] >= mcs]
        for x in ch:
            if size[x] < mcs:
                p = leaves(x)
                fall[p] = c
                contrib[c].append((L - birth[c]) * len(p))
        if len(big) >= 2:
            for x in big:
                cid = len(birth)
                birth.append(L); cpar.append(c); kids.append([]); contrib.append([]); kids[c].append(cid)
                contrib[c].append((L - birth[c]) * size[x])
                stack.append((x, cid))
        elif len(big) == 1:
            stack.append((big[0], c))
    nc = len(birth)
    S = [_fsum(contrib[c]) for c in range(nc)]
    best = list(S); sel = [False] * nc
    for c in range(nc - 1, 0, -1):                      # child ids are larger than their parent's
        sub = _fsum([best[k] for k in kids[c]])
        if kids[c] and sub > S[c]:
            best[c] = sub
        else:
            sel[c] = True
    sub_root = _fsum([best[k] for k in kids[0]])
    lab_c = np.full(nc, -1, np.int64); chosen = [False] * nc
    for c in range(1, nc):
        p = cpar[c]
        chosen[c] = sel[c] and lab_c[p] < 0
        lab_c[c] = c if chosen[c] else lab_c[p]
    if (fall < 0).any():
        raise ValueError("internal error: a point never left the condensed tree")
    raw = lab_c[fall]
    ids, first = np.unique(raw[raw >= 0], return_index=True)
    order = ids[np.argsort(np.flatnonzero(raw >= 0)[first])]
    remap = {int(c): i for i, c in enumerate(order.tolist())}
    lab = np.array([remap.get(int(r), -1) for r in raw.tolist()], np.int64)
    clusters = [dict(birth_lambda=float(birth[c]), members=np.flatnonzero(lab == i)) for i, c in enumerate(order)]
    return dict(labels=lab, clusters=clusters, root_preferred=bool(sub_root <= S[0]), root_stability=float(S[0]),
                children_stability=float(sub_root))


def _mst_of(X, core, lo, hi):
    """an exact minimum spanning tree of the graph (lo, hi) weighted by the mutual reachability at ``core``.
    Returns (u, v, w) with w recomputed by ``_mreach`` (zero weights are kept as zeros)."""
    n = len(X)
    w = _mreach(X, core, lo, hi)
    pos = w[w > 0]
    tiny = pos.min() / 2 if len(pos) else 1.0
    G = minimum_spanning_tree(coo_matrix((np.where(w > 0, w, tiny), (lo, hi)), shape=(n, n)).tocsr()).tocoo()
    if G.nnz != n - 1:
        raise ValueError("edge set does not span")
    a, b = G.row.astype(np.int64), G.col.astype(np.int64)
    return a, b, _mreach(X, core, a, b)


def _tree_from(X, dist, nbr, k, k_edges, mst):
    """minimum spanning tree at k from (spanning tree mst at k_edges >= k) + (k-NN graph at k_edges).

    By the containment theorem of CORE-SG (Neto et al. 2022, Thm 1), MST_k' + kNN_k' contains a minimum spanning
    tree of the complete mutual-reachability graph at every k <= k', so the ultrametric (and the canonical labels)
    equal those of the complete graph.  With k_edges == k this also re-weights a library tree with ``_mreach``."""
    n = len(X)
    lo, hi = _pairs(n, np.concatenate([np.repeat(np.arange(n), k_edges - 1), mst[0]]),
                    np.concatenate([nbr[:, 1:k_edges].ravel(), mst[1]]))
    return _mst_of(X, dist[:, k - 1], lo, hi)


def _tied_share(w):
    """share of spanning-tree edges whose weight equals that of another edge."""
    _, inv, cnt = np.unique(np.asarray(w, float), return_inverse=True, return_counts=True)
    return float((cnt[inv] > 1).mean()) if len(w) else 0.0


EMPIRICAL_ROOT_NOTE = ("empirical signal, not a guarantee: in an exploratory screen by the author (423 runs on 47 "
                       "benchmark datasets, min_cluster_size = min_samples), the step to the next value of min_samples "
                       "had ARI < 0.8 in 30.8% of the 227 runs with this flag, against 14.3% of the other 196")


# ============================================================ report
def _fmt(v, nd=3):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


@dataclass
class MinPtsReport:
    backend: str
    convention: str
    min_samples: int
    min_cluster_size: int
    coupled: bool
    mode: str
    extraction: str
    tree: str
    exact_tree: bool
    n: int
    n_clusters: int
    noise_share: float
    clusters: list
    steps: list
    warnings: list
    fits: int
    fits_skipped: int
    seconds: float
    verdict: str = ""
    labels: dict = field(default_factory=dict, repr=False)
    # added in 0.2.0 (None when not requested or not applicable)
    ties: dict = None
    certified_range: dict = None
    root_preference: dict = None
    fits_chained: int = 0
    version: str = __version__

    def to_dict(self, labels=False):
        d = {k: v for k, v in self.__dict__.items() if k != "labels"}
        if labels:
            d["labels"] = {int(k): v.tolist() for k, v in self.labels.items()}
        return d

    def summary(self, file=None):
        L = [f"min_samples={self.min_samples}, min_cluster_size={self.min_cluster_size} "
             f"({self.backend} convention: {self.convention}); {'coupled' if self.coupled else 'min_cluster_size fixed'}",
             f"extraction={self.extraction}; tree: {self.tree}; mode={self.mode}; n={self.n}; "
             f"{self.n_clusters} clusters, noise {100 * self.noise_share:.1f}%",
             "", "clusters at the chosen value (s_hat: points still dense enough at each higher value)",
             f"{'label':>5} {'size':>6} {'margin':>6}  s_hat vs next min_cluster_size"]
        for c in self.clusters:
            sh = "  ".join(f"+{j + 1}:{s}/{t}{'!' if r else ''}" for j, (s, t, r) in
                           enumerate(zip(c["s_hat"], c["next_mcs"], c["at_risk"])))
            L.append(f"{c['label']:>5} {c['size']:>6} {c['margin']:>6}  {sh}")
        L += ["", "neighbouring values (vs the chosen value)",
              f"{'min_samples':>11} {'mcs':>4} {'ARI':>6} {'ARI_nn':>6} {'dK':>4}  {'at risk':>7}  certificate"]
        for s in self.steps:
            if s["certified"] is None:
                cert = "n/a" + (f" ({s['cert_reason']})" if s["cert_reason"] else "")
            elif s["certified"]:
                cert = f"certified, |U|={s['U_size']} ({100 * s['U_share']:.2f}%)"
            else:
                cert = f"not certified: {s['cert_reason']}"
            L.append(f"{s['min_samples']:>11} {s['min_cluster_size']:>4} {_fmt(s['ari']):>6} {_fmt(s['ari_nn']):>6} "
                     f"{_fmt(s['delta_k']):>4}  {_fmt(s['n_at_risk']):>7}  {cert}")
        r = self.certified_range
        if r is not None:
            L.append("")
            if r.get("steps_certified"):
                L.append(f"certified range: every min_samples in ({self.min_samples}, {r['top_min_samples']}] keeps "
                         f"the labels outside U, |U|={r['U_size']} ({100 * r['U_share']:.2f}%); "
                         f"{r['checks']} certificate check{'s' * (r['checks'] != 1)} ({r['search']} search up to "
                         f"+{r['max_steps']})")
            else:
                L.append(f"certified range: none ({r['reason']}); {r['checks']} certificate "
                         f"check{'s' * (r['checks'] != 1)}")
        t = self.ties
        if t is not None:
            L.append("")
            L.append(f"ties: {100 * t['tied_edges_share']:.1f}% of spanning-tree edges share their weight; "
                     f"library {t['library_n_clusters']} clusters, tie-merged canonical {t['canonical_n_clusters']}, "
                     f"ARI {t['ari_library_vs_canonical']:.3f}"
                     + ("" if t["same_partition"] else "; the partitions differ, so the library's output depends "
                                                          "on how its spanning tree broke ties"))
        rp = self.root_preference
        if rp is not None and rp["flag"]:
            L += ["", "root preference: EOM would select the root (allow_single_cluster=True would return one cluster); "
                  + rp["note"]]
        L += [""] + [f"warning: {w}" for w in self.warnings]
        ch = f", {self.fits_chained} of them by the chained spanning-tree update" if self.fits_chained else ""
        L += [f"fits: {self.fits}{ch} (skipped thanks to the certificate: {self.fits_skipped}); {self.seconds:.2f} s",
              f"verdict: {self.verdict}", f"method: {CITATION}"]
        txt = "\n".join(L)
        print(txt, file=file)
        return txt


# ============================================================ shared set-up
def _prepare(X, backend, kwargs):
    if backend not in _OFFSET:
        raise ValueError("backend must be 'sklearn' or 'hdbscan'")
    if kwargs.get("metric", "euclidean") != "euclidean":
        raise NotImplementedError("only the euclidean metric is supported")
    if kwargs.get("match_reference_implementation"):
        raise NotImplementedError("match_reference_implementation changes the min_samples convention")
    kwargs.pop("metric", None)
    if backend == "sklearn" and set(kwargs) - _SK_OK:
        raise ValueError(f"unsupported sklearn arguments: {sorted(set(kwargs) - _SK_OK)}")
    X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("X must be a finite 2-D array")
    return X


def _fitter(X, backend, kwargs):
    def fit(k, mcs, core):
        if backend == "sklearn":
            return _fit_sklearn(X, k, mcs, core, kwargs)
        return _fit_hdbscan(X, k, mcs, kwargs)
    return fit


def _root_pref_tree(T):
    """would EOM select the root if it were eligible (allow_single_cluster)?  Same rule as the libraries."""
    best = dict(T["S"])
    root = T["root"]
    for c in sorted(T["nodes"], reverse=True):
        if c == root:
            continue
        sub = sum(best[k] for k in T["kids"][c])
        if T["kids"][c] and sub > T["S"][c]:
            best[c] = sub
    sub_root = sum(best[k] for k in T["kids"][root])
    return dict(flag=bool(sub_root <= T["S"][root]), root_stability=float(T["S"][root]),
                children_stability=float(sub_root), note=EMPIRICAL_ROOT_NOTE)


# ============================================================ main entry point
def check_min_samples(X, min_samples, min_cluster_size, backend="sklearn", steps=2, mode="fast", coupled=None,
                      u_cap=0.01, *, ties="library", refit="backend", range_search=None, range_steps=10, **kwargs):
    """Check how the HDBSCAN* flat clustering at ``min_samples`` behaves at the neighbouring values.

    Parameters
    ----------
    X : array (n, d), euclidean features (e.g. a UMAP embedding).
    min_samples, min_cluster_size : int, in the convention of ``backend``.
    backend : "sklearn" (sklearn.cluster.HDBSCAN; min_samples counts the point itself) or "hdbscan" (the hdbscan
        package; it does not, so its m equals sklearn's m + 1).
    steps : neighbouring values min_samples - steps .. min_samples + steps (unit steps).
    mode : "fast" -- one fit at min_samples and one k-NN query; s_hat and the certificate for every higher value,
        no refit.  "verify" -- also refit the neighbours the certificate did not certify with |U| <= u_cap * n
        (all lower neighbours, which no certificate covers).  "full" -- refit every neighbour (and check every
        certificate claim against the refit).
    coupled : move min_cluster_size together with min_samples (default: True when the two are equal in the
        backend's convention, else min_cluster_size stays fixed).
    u_cap : largest tolerance set, as a share of n, for which a certified step skips its refit in "verify" mode.

    Keyword-only options added in 0.2.0 (the defaults reproduce 0.1.0):

    ties : "library" -- the labels of the backend (binary single-linkage tree; with tied edge weights they depend on
        the spanning tree the library built).  "check" -- as "library", plus the tie-merged canonical labels at the
        chosen value and their agreement with the library's (``report.ties``).  "canonical" -- every reported
        labelling (chosen value and refits) is the tie-merged canonical one (``canonical_labels``); s_hat uses the
        canonical clusters; the certificate is not evaluated (it covers the library's binary tree).
    refit : "backend" -- refits run the backend.  "chain" (needs ties="canonical") -- one exact backend fit at the
        largest refitted value, then each spanning tree from the next larger one (see ``sweep``).
    range_search : None, "binary" or "linear" -- also find the certified range (``certified_range``, reported in
        ``report.certified_range``) over min_samples + 1 .. min_samples + range_steps.  In "verify" mode a step inside
        the certified range with |U| <= u_cap * n skips its refit; in "full" mode the range claim is checked against
        every refit inside it.
    range_steps : largest step tried by the range search.
    **kwargs : passed to the backend estimator (e.g. cluster_selection_method, approx_min_span_tree, algorithm).
    """
    t0 = time.perf_counter()
    if mode not in ("fast", "verify", "full"):
        raise ValueError("mode must be 'fast', 'verify' or 'full'")
    if ties not in ("library", "check", "canonical"):
        raise ValueError("ties must be 'library', 'check' or 'canonical'")
    if refit not in ("backend", "chain"):
        raise ValueError("refit must be 'backend' or 'chain'")
    if refit == "chain" and ties != "canonical":
        raise ValueError("refit='chain' needs ties='canonical': with tied weights, another exact spanning tree can "
                         "change the library's labels, but not the canonical ones")
    if range_search not in (None, "binary", "linear"):
        raise ValueError("range_search must be None, 'binary' or 'linear'")
    X = _prepare(X, backend, kwargs)
    n, d = X.shape
    off = _OFFSET[backend]
    k0 = int(min_samples) + off
    if coupled is None:
        coupled = int(min_samples) == int(min_cluster_size)
    vals = [j for j in range(-steps, steps + 1) if j != 0 and int(min_samples) + j >= 1 and
            (not coupled or min_cluster_size + j >= 2)]
    kmax = k0 + max([j for j in vals if j > 0], default=0)
    if kmax > n:
        raise ValueError("min_samples + steps exceeds the number of points")
    mcs_of = lambda j: int(min_cluster_size) + (j if coupled else 0)
    extraction, tree, exact = _tree_info(backend, kwargs, d)
    canon = ties == "canonical"
    warnings = []
    if not exact:
        if ties != "library":
            raise ValueError("tie-merged canonical labels need an exact spanning tree; pass approx_min_span_tree=False")
        warnings.append(APPROX_WARNING)
    in_scope = (extraction == "eom" and not kwargs.get("cluster_selection_epsilon", 0)
                and not kwargs.get("allow_single_cluster", False) and kwargs.get("max_cluster_size") in (None, 0)
                and not kwargs.get("cluster_selection_persistence", 0))
    if not in_scope:
        if ties != "library":
            raise ValueError("tie-merged canonical labels cover the default EOM extraction only")
        warnings.append("settings outside the paper's scope (EOM extraction without epsilon, persistence, "
                        "single-cluster or max-size options); s_hat is reported, the certificate is not evaluated")
    if len(np.unique(X, axis=0)) < n:
        warnings.append("duplicate rows: tied distances can change the output (the paper removed duplicates "
                        "and added a 1e-9 jitter)")
    want_range = range_search is not None and not canon
    max_u = max(0, min(int(range_steps), n - k0)) if want_range else 0
    dist, nbr = _knn(X, max(kmax, k0, k0 + max_u), kwargs.get("algorithm", "auto"), kwargs.get("leaf_size", 40))
    backend_fit = _fitter(X, backend, kwargs)
    nfit = 0

    def fit(j):
        nonlocal nfit
        nfit += 1
        return backend_fit(k0 + j, mcs_of(j), dist[:, k0 + j - 1])

    R0 = fit(0)
    lab_lib = R0["lab"]
    tie_info = None
    C0 = None
    if ties != "library":
        tr0 = _tree_from(X, dist, nbr, k0, k0, R0["mst"])
        C0 = _canonical(n, *tr0, mcs_of(0))
        same = (C0["labels"].max() == lab_lib.max() and np.array_equal(C0["labels"] < 0, lab_lib < 0)
                and adjusted_rand_score(C0["labels"], lab_lib) == 1.0)
        tie_info = dict(mode=ties, tied_edges_share=_tied_share(tr0[2]), library_n_clusters=int(lab_lib.max() + 1),
                        canonical_n_clusters=int(C0["labels"].max() + 1), same_partition=bool(same),
                        ari_library_vs_canonical=float(adjusted_rand_score(lab_lib, C0["labels"])))
        if tie_info["ari_library_vs_canonical"] < 0.99 or tie_info["library_n_clusters"] != tie_info["canonical_n_clusters"]:
            warnings.append("tied edge weights: the library's labels differ from the tie-merged canonical labels at "
                            f"the chosen value (ARI {tie_info['ari_library_vs_canonical']:.3f}, "
                            f"{tie_info['library_n_clusters']} vs {tie_info['canonical_n_clusters']} clusters); another "
                            "exact spanning tree of the same data can give other labels")
    ups = [j for j in vals if j > 0]
    clusters = []
    root_pref = None
    sel = []
    T = None
    if canon:
        lab0 = C0["labels"]
        cl_info = [(c["members"], c["birth_lambda"]) for c in C0["clusters"]]
        root_pref = dict(flag=C0["root_preferred"], root_stability=C0["root_stability"],
                         children_stability=C0["children_stability"], note=EMPIRICAL_ROOT_NOTE)
    else:
        lab0 = lab_lib
        T = parse_condensed(R0["ct"], n)
        sel = selected_nodes(T, lab0) if lab0.max() >= 0 else []
        if sel is None:
            warnings.append("selected clusters could not be matched to the condensed tree; s_hat and the certificate "
                            "are not available")
            sel = []
        cl_info = [(np.flatnonzero(lab0 == j), T["birth"][c]) for j, c in enumerate(sel)]
        if in_scope:
            root_pref = _root_pref_tree(T)
    for j, (mem, lb) in enumerate(cl_info):
        eb = 1.0 / lb if lb > 0 else INF
        sh = [int((dist[mem, k0 + u - 1] <= eb).sum()) for u in ups]
        nx = [mcs_of(u) for u in ups]
        clusters.append(dict(label=j, size=int(len(mem)), margin=int(len(mem)) - int(min_cluster_size),
                             birth_radius=float(eb), s_hat=sh, next_mcs=nx, at_risk=[s < t for s, t in zip(sh, nx)]))

    Kd = None
    stepinfo = {}
    for u in ups:
        st = dict(certified=None, cert_reason=None, U_size=None, U_share=None, U=None)
        if not exact or not in_scope:
            st["cert_reason"] = "approximate tree" if not exact else "outside scope"
        elif canon:
            st["cert_reason"] = "canonical labels (the certificate covers the library's binary tree)"
        elif not sel:
            st["cert_reason"] = "no cluster"
        else:
            try:
                if Kd is None:
                    Kd = Dendro(n, *R0["mst"])
                Fd = _lower_hierarchy(X, R0["mst"], dist, nbr, k0 + u)
                m_ = mcs_of(u)
                res = certify(T, sel, Kd, Fd, mc=m_, mp=m_, mb=m_, phantom=_phantom(X, dist[:, k0 + u - 1]))
            except ValueError as e:
                res = dict(cert=False, reason=f"could not build hierarchies: {e}")
            st["certified"] = bool(res["cert"]); st["cert_reason"] = None if res["cert"] else res["reason"]
            if res["cert"]:
                st.update(U_size=int(res["U"].sum()), U_share=float(res["U"].mean()),
                          U=np.flatnonzero(res["U"]).tolist(), _res=res)
        stepinfo[u] = st

    rng = None
    rng_res = None
    if range_search is not None:
        rng = dict(search=range_search, max_steps=max_u, steps_certified=0, top_min_samples=int(min_samples),
                   top_min_cluster_size=int(min_cluster_size), U_size=None, U_share=None, U=None, checks=0,
                   evaluated={}, reason=None,
                   edge_set=f"observed spanning tree + k-NN graph at min_samples + {max_u}")
        if canon:
            rng["reason"] = "canonical labels (the certificate covers the library's binary tree)"
        elif not exact or not in_scope:
            rng["reason"] = "approximate tree" if not exact else "outside scope"
        elif not sel:
            rng["reason"] = "no cluster"
        elif max_u < 1:
            rng["reason"] = "no room above min_samples"
        else:
            try:
                if Kd is None:
                    Kd = Dendro(n, *R0["mst"])
                rr = _range_core(X, T, sel, Kd, R0["mst"], dist, nbr, k0, mcs_of, max_u, range_search)
                u_star, memo, results = rr
                rng.update(steps_certified=u_star, top_min_samples=int(min_samples) + u_star,
                           top_min_cluster_size=mcs_of(u_star), checks=len(memo),
                           evaluated={int(min_samples) + u: v for u, v in sorted(memo.items())})
                if u_star:
                    rng_res = results[u_star]
                    rng.update(U_size=int(rng_res["U"].sum()), U_share=float(rng_res["U"].mean()),
                               U=np.flatnonzero(rng_res["U"]).tolist())
                fails = [u for u in sorted(memo) if not memo[u] and u > u_star]
                rng["reason"] = results[fails[0]]["reason"] if fails else (
                    "maximum step reached" if u_star == max_u else None)
            except ValueError as e:
                rng["reason"] = f"could not build hierarchies: {e}"

    labels = {int(min_samples): lab0}
    skipped = 0
    steps_out = []
    to_fit = {}
    for j in vals:
        st = stepinfo.get(j, dict(certified=None, U_size=None, U_share=None, U=None,
                                  cert_reason="decreases are not covered by the predictor or the certificate"))
        res = st.pop("_res", None)
        in_range = rng_res is not None and 0 < j <= rng["steps_certified"]
        skip_ok = (res is not None and st["U_share"] <= u_cap) or (in_range and rng["U_share"] <= u_cap)
        do_fit = mode == "full" or (mode == "verify" and not skip_ok)
        if mode == "verify" and not do_fit:
            skipped += 1
        to_fit[j] = (st, res, in_range, do_fit)
    fitted = _refits(X, dist, nbr, k0, R0, mcs_of, [j for j in vals if to_fit[j][3]], canon, refit, fit)
    nchain = sum(1 for v in fitted.values() if v[1])
    for j in vals:
        st, res, in_range, do_fit = to_fit[j]
        row = dict(min_samples=int(min_samples) + j, min_cluster_size=mcs_of(j), ari=None, ari_nn=None,
                   delta_k=None, n_clusters=None, n_at_risk=None, cert_holds_on_refit=None, **st)
        if range_search is not None:
            row["in_certified_range"] = bool(in_range)
            row["range_holds_on_refit"] = None
        if j > 0:
            row["n_at_risk"] = sum(c["at_risk"][ups.index(j)] for c in clusters)
        if do_fit:
            lab1 = fitted[j][0]; labels[int(min_samples) + j] = lab1
            both = (lab0 >= 0) & (lab1 >= 0)
            row.update(ari=float(adjusted_rand_score(lab0, lab1)),
                       ari_nn=float(adjusted_rand_score(lab0[both], lab1[both])) if both.sum() > 1 else None,
                       n_clusters=int(lab1.max() + 1), delta_k=int(lab1.max() - lab0.max()))
            for key, r_ in (("cert_holds_on_refit", res), ("range_holds_on_refit", rng_res if in_range else None)):
                if r_ is not None:
                    ok, nbad = verify(r_, sel, lab0, lab1)
                    row[key] = ok
                    if not ok:
                        what = "certificate" if key == "cert_holds_on_refit" else "range certificate"
                        warnings.append(f"{what} claim failed at min_samples={row['min_samples']} "
                                        f"({nbad} points outside U changed); please report this")
        steps_out.append(row)

    rep = MinPtsReport(backend=backend,
                       convention=("min_samples counts the point itself (sklearn m+1 = hdbscan m)" if off == 0 else
                                   "min_samples does not count the point itself (hdbscan m = sklearn m+1)"),
                       min_samples=int(min_samples), min_cluster_size=int(min_cluster_size), coupled=bool(coupled),
                       mode=mode, extraction=extraction, tree=tree, exact_tree=exact, n=n,
                       n_clusters=int(lab0.max() + 1), noise_share=float((lab0 < 0).mean()), clusters=clusters,
                       steps=steps_out, warnings=warnings, fits=nfit + nchain, fits_skipped=skipped,
                       seconds=time.perf_counter() - t0, labels=labels, ties=tie_info, certified_range=rng,
                       root_preference=root_pref, fits_chained=nchain)
    rep.verdict = _verdict(rep, ups)
    return rep


def _refits(X, dist, nbr, k0, R0, mcs_of, js, canon, refit, fit):
    """labels at every step j in js: {j: (labels, came from the chained update?)}."""
    n = len(X)
    out = {}

    def canon_at(j, tr):
        return _canonical(n, *tr, mcs_of(j))["labels"]
    if refit == "backend":
        for j in js:
            R = fit(j)
            out[j] = (canon_at(j, _tree_from(X, dist, nbr, k0 + j, k0 + j, R["mst"])) if canon else R["lab"], False)
        return out
    up = sorted([j for j in js if j > 0], reverse=True)
    if up:                                              # one exact fit at the largest value, then down the chain
        tr = _tree_from(X, dist, nbr, k0 + up[0], k0 + up[0], fit(up[0])["mst"])
        out[up[0]] = (canon_at(up[0], tr), False)
        ke = k0 + up[0]
        for j in up[1:]:
            tr = _tree_from(X, dist, nbr, k0 + j, ke, tr); ke = k0 + j
            out[j] = (canon_at(j, tr), True)
    dn = sorted([j for j in js if j < 0], reverse=True)
    if dn:                                              # from the tree at the chosen value
        tr = _tree_from(X, dist, nbr, k0, k0, R0["mst"]); ke = k0
        for j in dn:
            tr = _tree_from(X, dist, nbr, k0 + j, ke, tr); ke = k0 + j
            out[j] = (canon_at(j, tr), True)
    return out


def _range_core(X, T, sel, Kd, mst, dist, nbr, k0, mcs_of, max_u, search):
    """range certificate at the tops k0 + u, u = 1..max_u, on the fixed edge set; returns (u*, evaluated, results)."""
    FE = _FixedEdges(X, mst, nbr, k0 + max_u)
    ph = _phantom(X, dist[:, k0])                       # core distances at k0 + 1: a lower bound over the range
    m1 = mcs_of(1)
    results = {}

    def ok(u):
        r = certify(T, sel, Kd, FE.hierarchy(dist[:, k0 + u - 1]), mc=mcs_of(u), mp=m1, mb=m1, phantom=ph)
        results[u] = r
        return r["cert"]
    u_star, memo = _search_top(ok, max_u, search)
    return u_star, memo, results


def certified_range(X, min_samples, min_cluster_size, backend="sklearn", max_steps=10, coupled=None,
                    search="binary", **kwargs):
    """Largest t such that every min_samples in (min_samples, t] is certified to keep the labels outside one set U.

    One fit at ``min_samples``, one k-NN query at min_samples + max_steps, and O(log max_steps) certificate checks
    (``search="binary"``) or max_steps checks (``search="linear"``).  Returns a dict: ``top_min_samples`` (t, in the
    backend's convention; equal to min_samples when nothing is certified), ``steps_certified`` (t - min_samples),
    ``top_min_cluster_size``, ``U`` (point indices), ``U_size``, ``U_share``, ``checks``, ``evaluated`` ({value:
    certified?} for the tops tried), ``reason`` (why the next top failed), ``warnings``.  Same scope as the
    certificate: EOM, exact tree, euclidean metric, library labels.  The same search runs inside
    ``check_min_samples(..., range_search="binary")``.

    The check at top t is the range form of Theorem 1 of the paper (Coutinho 2026): the observed hierarchy at
    min_samples (a lower bound of the hierarchy at every larger value), the lower hierarchy F(t) of Corollary 2 (an
    upper bound at every value up to t), the phantom bound with the core distances at min_samples + 1, m_c = the
    min_cluster_size at t, and m_p = m_b = the min_cluster_size at min_samples + 1.  F(t) is the single-linkage
    hierarchy of a FIXED edge set E (observed spanning tree + k-NN graph at min_samples + max_steps) weighted by the
    mutual reachability at t.

    Why the search is valid.

    1. Soundness does not depend on the search.  A top is reported only if the certificate was evaluated there and
       held, and a certificate at top t covers every value in (min_samples, t]: for every k'' in that range,
       d_k'' <= d_t edge by edge, and F(t) is a single-linkage hierarchy of a subgraph, so its merge heights are at
       least those of the hierarchy at k''; the observed hierarchy is below it because d_k0 <= d_k''; the phantom
       bound uses core distances at k0 + 1 <= k''; the size thresholds are the extreme ones over the range.

    2. Maximality uses monotonicity in t: certified at t => certified at every t' < t.  With E fixed, every edge
       weight max(core_t(a), core_t(b), |a - b|) is non-decreasing in t, so every minimax (merge) height of F(t) is
       non-decreasing in t (raising edge weights cannot lower a minimax path distance), and every block of F(t) at a
       given radius shrinks as t grows.  For fixed anchors, every test of the certificate gets harder as F rises:
       (C) the anchors' F-blocks must keep >= m_c points, and blocks shrink while m_c can only grow; the phantom
       test bounds the part of an observed block that F-blocks do not cover, which grows; the certified trunk
       stability S_lo is an integral of F-block sizes, which falls, and the trunk can only end earlier, which
       raises the descendant bound D_hi (computed on the observed tree); in (D2) the separations between groups of
       anchors grow, so more groups are tested and each upper bound S_hi integrates over a longer interval with
       smaller subtracted F-blocks.  v0.1 built F from the k-NN graph at the step itself, so the edge set grew
       with t and the weights' monotonicity no longer implied that of the merge heights.
       The anchors are chosen from F(t) (in each cluster, the point whose smallest F-block of m_c points lies in the
       cluster and has the lowest height), and a different F can choose a different anchor.  This step is not
       covered by the argument above; it is covered by brute force instead: on 300 random small sets (1,829 runs,
       3,937 certified tops, exact refits) the fixed-edge-set certificate was never non-monotone and never wrong,
       while the growing edge set of v0.1 gave 4 non-monotone cases; the package tests compare binary and linear
       search on random seeds.  Were monotonicity to fail, binary search would still return a certified top (by
       point 1), possibly not the largest; ``search="linear"`` returns the largest certified top.
    """
    rep = check_min_samples(X, min_samples, min_cluster_size, backend=backend, steps=0, mode="fast", coupled=coupled,
                            range_search=search, range_steps=max_steps, **kwargs)
    out = dict(rep.certified_range)
    out["warnings"] = list(rep.warnings)
    return out


def sweep(X, min_samples_values, min_cluster_size=None, backend="sklearn", **kwargs):
    """Tie-merged canonical labels for several values of min_samples from one exact fit (chained spanning trees).

    One k-NN query and one exact backend fit at the largest value; then, going down the values, the spanning tree
    at each value is a minimum spanning tree of (tree at the previous, larger value) + (its k-NN graph), which
    contains an exact minimum spanning tree of the complete graph (Neto et al. 2022, Thm 1, applied value by value),
    so each step works on n * k edges instead of refitting.  The labels are the tie-merged canonical ones
    (``canonical_labels``), which do not depend on which exact spanning tree is used; library labels could.

    min_cluster_size : None (equal to each min_samples in the backend's convention, as the libraries do when
        min_samples is unset), an int (fixed), or a callable value -> min_cluster_size.
    Returns {min_samples: labels}.  Scope: EOM extraction (default settings), exact tree, euclidean metric.
    """
    X = _prepare(X, backend, kwargs)
    n, d = X.shape
    _, _, exact = _tree_info(backend, kwargs, d)
    if not exact:
        raise ValueError("canonical labels need an exact spanning tree; pass approx_min_span_tree=False")
    if (kwargs.get("cluster_selection_method", "eom") != "eom" or kwargs.get("cluster_selection_epsilon", 0)
            or kwargs.get("allow_single_cluster", False) or kwargs.get("max_cluster_size") not in (None, 0)):
        raise ValueError("canonical labels cover the default EOM extraction only")
    off = _OFFSET[backend]
    vals = sorted({int(v) for v in np.atleast_1d(min_samples_values)}, reverse=True)
    if callable(min_cluster_size):
        mcs_of = lambda v: int(min_cluster_size(v))
    elif min_cluster_size is None:
        mcs_of = lambda v: v
    else:
        mcs_of = lambda v: int(min_cluster_size)
    if vals[-1] + off < 1 or vals[0] + off > n or min(mcs_of(v) for v in vals) < 2:
        raise ValueError("min_samples out of range or min_cluster_size < 2")
    kmax = vals[0] + off
    dist, nbr = _knn(X, kmax, kwargs.get("algorithm", "auto"), kwargs.get("leaf_size", 40))
    R = _fitter(X, backend, kwargs)(kmax, mcs_of(vals[0]), dist[:, kmax - 1])
    tr = _tree_from(X, dist, nbr, kmax, kmax, R["mst"]); ke = kmax
    out = {vals[0]: _canonical(n, *tr, mcs_of(vals[0]))["labels"]}
    for v in vals[1:]:
        tr = _tree_from(X, dist, nbr, v + off, ke, tr); ke = v + off
        out[v] = _canonical(n, *tr, mcs_of(v))["labels"]
    return dict(sorted(out.items()))


def canonical_labels(X, min_samples, min_cluster_size, backend="sklearn", **kwargs):
    """HDBSCAN* labels with ties merged as in the definition (Campello et al. 2015, Sec. 3.2: edges of equal weight
    are removed simultaneously), for one value.  The labels are the same for every exact minimum spanning tree of
    the data and for every row order (up to renaming); with pairwise distinct edge weights they equal the library's.
    Ties are exact equalities of the computed mutual-reachability distances.  One exact backend fit."""
    return sweep(X, [min_samples], min_cluster_size, backend=backend, **kwargs)[int(min_samples)]


def _verdict(rep, ups):
    if rep.n_clusters == 0:
        return "no cluster at the chosen value"
    parts = []
    risk1 = [c["label"] for c in rep.clusters if c["at_risk"] and c["at_risk"][0]]
    fitted = [s for s in rep.steps if s["ari"] is not None]
    frag = [s["min_samples"] for s in fitted if s["ari"] < 0.8]
    up = [s for s in rep.steps if s["min_samples"] > rep.min_samples]
    cert = [f"{s['min_samples']} (|U| {100 * s['U_share']:.1f}%)" for s in up if s["certified"]]
    if risk1:
        parts.append(f"AT RISK: cluster(s) {risk1} predicted to shrink below the next min_cluster_size")
    if frag:
        parts.append(f"FRAGILE: ARI < 0.8 at min_samples {frag}")
    if cert:
        parts.append("certified unchanged outside U at min_samples " + ", ".join(cert))
    r = rep.certified_range
    if r is not None and r.get("steps_certified"):
        parts.append(f"certified range up to min_samples {r['top_min_samples']} (|U| {100 * r['U_share']:.1f}%)")
    if rep.ties is not None and rep.ties["ari_library_vs_canonical"] < 0.8:
        parts.append(f"TIE-SENSITIVE: library and canonical labels differ (ARI {rep.ties['ari_library_vs_canonical']:.3f})")
    if not parts:
        parts.append("no cluster flagged, no step certified" + ("" if fitted else " (refit with mode='verify')"))
    return "; ".join(parts)
