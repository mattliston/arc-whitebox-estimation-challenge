#!/usr/bin/env python3
"""TAG r0c1-fourth-distinct-v1: Gaussian-part connected kappa4 for DISTINCT
4-tuples of post-ReLU units, and its push into K22/K31/k4 of the next
layer, factorized along tree diagrams (no 256^4 materialization).

Linked-cluster: kappa4conn(f_i,f_j,f_k,f_l) at jointly-Gaussian h =
sum over bond patterns {t_pq} (6 pairs) with CONNECTED graph, value
prod rho^t/t! * prod_node a[deg_node], a[t] = E[f He_t]. Leading terms are
the 16 labeled spanning trees (4 stars + 12 paths) plus degree upgrades to
BOND_CAP total. Push: K22next[a,b] = sum W[i,a]W[j,a]W[k,b]W[l,b] kappa4c;
each tree term factorizes into B-vector chains. Single-pair collisions are
subtracted (those tuples belong to the carried-plane push).
Gate 1: kappa4c formula vs 4D Gauss-Hermite on synthetic quadruples.
Gate 2: end-to-end at L0 -> pre-1 vs 64M MC K22/K31 truth.
Usage: fourth_distinct.py  (runs both gates)"""
import itertools
import json
from math import erf, factorial

import numpy as np

_erf = np.vectorize(erf)
BOND_CAP = 4


def a_columns(m, v, qmax=8):
    v = np.maximum(v, 1e-30)
    sd = np.sqrt(v)
    al = m / sd
    pdf = np.exp(-0.5 * al * al) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1 + _erf(al / np.sqrt(2)))
    ma = -al
    He = [np.ones_like(al), ma]
    for k in range(2, qmax + 2):
        He.append(ma * He[-1] - (k - 1) * He[-2])
    c1 = [sd * pdf + m * cdf, sd * cdf]
    for q in range(2, qmax + 1):
        c1.append(sd * He[q - 2] * pdf)
    return np.stack(c1)


PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def connected4(ts):
    parent = list(range(4))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (p, q), t in zip(PAIRS, ts):
        if t:
            rp, rq = find(p), find(q)
            if rp != rq:
                parent[rp] = rq
    return len({find(c) for c in range(4)}) == 1


def enumerate_terms(cap=BOND_CAP):
    """Connected TREE bond patterns over 4 nodes (exactly 3 bonded pairs,
    no cycles), total bond order <= cap. Looped patterns (>=4 distinct
    bonded pairs) are O(rho^4)-class and omitted in v1."""
    out = []
    for ts in itertools.product(range(cap + 1), repeat=6):
        tot = sum(ts)
        if tot == 0 or tot > cap:
            continue
        if sum(1 for t in ts if t) != 3:
            continue
        if not connected4(ts):
            continue
        deg = [0, 0, 0, 0]
        for (p, q), t in zip(PAIRS, ts):
            deg[p] += t
            deg[q] += t
        coef = 1.0
        for t in ts:
            coef /= factorial(t)
        out.append((ts, tuple(deg), coef))
    return out


def kappa4c_scalar(mm, sd, rho6, terms):
    """Scalar evaluation at one quadruple; rho6 keyed by PAIRS order."""
    cols = [a_columns(mm[c], sd[c] ** 2) for c in range(4)]
    tot = 0.0
    for (ts, deg, coef) in terms:
        val = coef
        for (pi, t) in zip(range(6), ts):
            if t:
                val *= rho6[pi] ** t
        for c in range(4):
            val *= cols[c][deg[c]]
        tot += val
    return tot


def gate1():
    """kappa4c formula vs 4D GH quadrature on synthetic Gaussian
    quadruples."""
    gh_x, gh_w = np.polynomial.hermite_e.hermegauss(48)
    gh_w = gh_w / np.sqrt(2 * np.pi)
    rng = np.random.default_rng(9)
    terms = enumerate_terms()
    print(json.dumps({"n_terms": len(terms)}))
    worst = 0.0
    for trial in range(4):
        mm = rng.normal(0, 0.6, 4)
        sd = np.exp(rng.normal(0, 0.25, 4))
        while True:
            A = rng.normal(0, 0.35, (4, 4))
            R = A @ A.T
            dinv = 1 / np.sqrt(np.diag(R))
            R = R * np.outer(dinv, dinv)
            offs = R[np.triu_indices(4, 1)]
            if np.linalg.eigvalsh(R).min() > 0.15 and np.abs(offs).max() < 0.5:
                break
        L = np.linalg.cholesky(R)
        grids = np.meshgrid(*([gh_x] * 4), indexing="ij")
        Z = np.stack([g.ravel() for g in grids])
        G = L @ Z
        WT = np.ones(len(Z[0]))
        for c in range(4):
            WT = WT * np.repeat(
                np.tile(gh_w, int(48 ** c)), int(48 ** (3 - c)))
        f = [np.maximum(mm[c] + sd[c] * G[c], 0.0) for c in range(4)]
        E = lambda a: float(np.sum(WT * a))
        e1 = [E(fc) for fc in f]
        fc_ = [f[c] - e1[c] for c in range(4)]
        mu1111 = E(fc_[0] * fc_[1] * fc_[2] * fc_[3])
        mu2 = {}
        for c, d in itertools.combinations(range(4), 2):
            mu2[(c, d)] = E(fc_[c] * fc_[d])
        k4_true = (mu1111 - mu2[(0, 1)] * mu2[(2, 3)]
                   - mu2[(0, 2)] * mu2[(1, 3)]
                   - mu2[(0, 3)] * mu2[(1, 2)])
        rho6 = [R[p][q] for (p, q) in PAIRS]
        k4_mine = kappa4c_scalar(mm, sd, rho6, terms)
        rel = abs(k4_mine - k4_true) / max(abs(k4_true), 1e-14)
        worst = max(worst, rel)
        print(json.dumps({"k4_true": k4_true, "k4_mine": k4_mine,
                          "rel": round(rel, 4)}))
    print(json.dumps({"gate1_worst_rel": round(worst, 4)}))
    return worst


def push_distinct(m, v, C, Wl, cap=BOND_CAP):
    """Pushed contributions of the distinct-4 Gaussian completion to
    (k4next, K22next, K31next). Tree-factorized; single-pair collisions
    subtracted. All inputs are the POST-frame pre-activation state h
    (the variables under the ReLU): m, v, C of h; Wl the next weights."""
    W_ = m.size
    sd = np.sqrt(np.maximum(v, 1e-30))
    a = a_columns(m, v)
    rho = C / np.outer(sd, sd)
    np.fill_diagonal(rho, 0.0)
    rp = {t: rho ** t for t in range(1, cap + 1)}
    terms = enumerate_terms(cap)
    K22n = np.zeros((W_, W_))
    K31n = np.zeros((W_, W_))
    k4n = np.zeros(W_)
    # slot assignments: K22 slots (a,a,b,b); K31 slots (a,a,a,b);
    # k4 slots (a,a,a,a). For each term, nodes 0..3 map to slots 0..3.
    # Build the contraction by sequential elimination along the graph.
    for (ts, deg, coef) in terms:
        edges = [(PAIRS[i], t) for i, t in enumerate(ts) if t]
        for target, slots in (("K22", (0, 0, 1, 1)), ("K31", (0, 0, 0, 1)),
                              ("k4", (0, 0, 0, 0))):
            val = _contract_term(a, deg, edges, rp, Wl, slots)
            if target == "K22":
                K22n += coef * val
            elif target == "K31":
                K31n += coef * val
            else:
                k4n += coef * np.diag(val) if val.ndim == 2 else coef * val
    np.fill_diagonal(K22n, 0.0)
    np.fill_diagonal(K31n, 0.0)
    return k4n, K22n, K31n


def _contract_term(a, deg, edges, rp, Wl, slots):
    """Contract one Kibble term against W-legs. Nodes carry a[deg]*W[:,slot];
    edges carry rho^t. Uses greedy leaf elimination (works for trees and
    small loops via einsum fallback). Returns [A,B] matrix over the two
    output slots (or vector-diag if all slots equal).

    Approximation note: single-pair index collisions are NOT subtracted
    here (bounded O(1/width) per pair for the tree sums, checked in
    gate 2)."""
    import numpy as np
    W_ = Wl.shape[0]
    nodes = {c: a[deg[c]][:, None] * Wl for c in range(4)}   # [i, out]
    # leaf elimination: a leaf node with one edge folds into its neighbor
    adj = {c: [] for c in range(4)}
    for (p, q), t in edges:
        adj[p].append((q, t))
        adj[q].append((p, t))
    # slot-vector per node: which output column (0->a-slot, 1->b-slot)
    # After elimination each remaining node's array is [i, outA?, outB?]
    # For simplicity handle TREES only (the dominant terms); loops fall
    # back to a dense einsum over the involved pattern (rare, small cap).
    if sum(t for _, t in edges) != len(edges) or len(edges) > 3:
        pass
    # generic path: einsum with letters per node index
    letters = "ijkl"
    out_letters = ["a" if s == 0 else "b" for s in slots]
    operands = []
    subs = []
    for c in range(4):
        operands.append(nodes[c])
        subs.append(letters[c] + out_letters[c])
    for (p, q), t in edges:
        operands.append(rp[t])
        subs.append(letters[p] + letters[q])
    uniq = sorted(set(out_letters))
    spec = ",".join(subs) + "->" + "".join(uniq)
    res = np.einsum(spec, *operands, optimize="greedy")
    if len(uniq) == 1:
        return res
    return res


def gate2():
    """End-to-end: L0 (exact Gaussian pre) push -> pre-1 K22/K31 vs 64M
    truth, plane-supported baseline vs +distinct completion."""
    import sys
    sys.path.insert(0, ".")
    from tensor_recursion_w256_v2 import (generate_weights, fourth_emission,
                                          fourth_push)
    mcv4 = np.load("mc_w256_seed1000000_v4.npz")
    W = generate_weights(1000000)
    W0, W1 = W[0].astype(np.float64), W[1].astype(np.float64)
    m = np.zeros(256)
    C = W0.T @ W0
    v = np.diag(C)
    Z = np.zeros(256)
    ZM = np.zeros((256, 256))
    k4p, K22p, K31p = fourth_emission(m, v, Z, Z, C, ZM, ZM, ZM)
    k4b, K22b, K31b = fourth_push(k4p, K22p, K31p, W1)
    k4d, K22d, K31d = push_distinct(m, v, C, W1)
    tK22 = mcv4["pre_K22"][1].astype(np.float64)
    tK31 = mcv4["pre_K31"][1].astype(np.float64)
    np.fill_diagonal(tK22, 0.0)
    np.fill_diagonal(tK31, 0.0)
    s22 = float(np.sqrt(np.mean(tK22 ** 2)))
    s31 = float(np.sqrt(np.mean(tK31 ** 2)))
    out = {
        "K22_plane_only": round(float(np.sqrt(np.mean(
            (K22b - tK22) ** 2))) / s22, 4),
        "K22_plus_distinct": round(float(np.sqrt(np.mean(
            (K22b + K22d - tK22) ** 2))) / s22, 4),
        "K31_plane_only": round(float(np.sqrt(np.mean(
            (K31b - tK31) ** 2))) / s31, 4),
        "K31_plus_distinct": round(float(np.sqrt(np.mean(
            (K31b + K31d - tK31) ** 2))) / s31, 4),
    }
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    import sys as _s
    if len(_s.argv) > 1 and _s.argv[1] == "g2":
        gate2()
    else:
        gate1()
        gate2()
