#!/usr/bin/env python3
"""4-point certificate: connected kappa4(x_j,x_k,x_l,x_h) of a Gaussian ReLU quadruple via 4D Gauss-Hermite
quadrature vs the 4-point Hermite (Mehler) expansion over connected bond words (6 edges, total order <= NMAX),
and the STAR-word share (hub with three single bonds; coefficient c_hub[3] * prod c_leaf[1] * prod rho)."""
import json, sys, itertools
from math import factorial
import numpy as np
sys.path.insert(0, ".")
from k3gen import hermite_cols
def kappa4_gh(m, C, nq=48):
    x, w = np.polynomial.hermite_e.hermegauss(nq); w = w / w.sum()
    L = np.linalg.cholesky(C); grids = np.meshgrid(x, x, x, x, indexing="ij")
    U = np.stack([g.ravel() for g in grids], 1) @ L.T + m; R = np.maximum(U, 0.0)
    Wt = (w[:, None, None, None] * w[None, :, None, None] * w[None, None, :, None] * w[None, None, None, :]).ravel()
    E = lambda idx: float((Wt * np.prod(R[:, list(idx)], axis=1)).sum())
    E1 = [E([i]) for i in range(4)]; E2 = {(i, j): E([i, j]) for i in range(4) for j in range(i + 1, 4)}; E3 = {t: E(list(t)) for t in itertools.combinations(range(4), 3)}
    E4 = E([0, 1, 2, 3])
    def e2(i, j): return E2[(min(i, j), max(i, j))]
    # connected 4th cumulant from raw moments (all variables distinct)
    k = E4
    for i in range(4):
        rest = [q for q in range(4) if q != i]; k -= E1[i] * E3[tuple(rest)]
    for (i, j) in E2:
        rest = [q for q in range(4) if q not in (i, j)]; k -= e2(i, j) * e2(rest[0], rest[1]) if i < j and (i, j) < tuple(rest) else 0.0
    # careful: subtract each unordered pair-pair partition once: (01|23),(02|13),(03|12)
    k = E4 - sum(E1[i] * E3[tuple(q for q in range(4) if q != i)] for i in range(4)) \
        - (e2(0, 1) * e2(2, 3) + e2(0, 2) * e2(1, 3) + e2(0, 3) * e2(1, 2)) \
        + 2 * sum(E1[i] * E1[j] * e2(kk, l) for (i, j, kk, l) in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2), (1, 2, 0, 3), (1, 3, 0, 2), (2, 3, 0, 1)]) \
        - 6 * np.prod(E1)
    return k
def series4(m, C, nmax=8):
    v = np.diag(C); c1, _, _, sd, _, _ = hermite_cols(m, v); rho = C / np.outer(sd, sd)
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    tot = star = 0.0; global shares; shares = {}
    for a in itertools.product(range(nmax + 1), repeat=6):
        if sum(a) == 0 or sum(a) > nmax: continue
        deg = [0] * 4
        for (e, (i, j)) in zip(a, edges): deg[i] += e; deg[j] += e
        # connectedness of the bond graph
        adj = {i: set() for i in range(4)}
        for (e, (i, j)) in zip(a, edges):
            if e: adj[i].add(j); adj[j].add(i)
        seen = {0}; stack = [0]
        while stack:
            u = stack.pop()
            for w_ in adj[u]:
                if w_ not in seen: seen.add(w_); stack.append(w_)
        if len(seen) < 4: continue
        wv = np.prod([c1[deg[i]][i] for i in range(4)]) * np.prod([rho[i, j] ** e / factorial(e) for (e, (i, j)) in zip(a, edges)])
        tot += wv
        if sum(a) == 3 and max(deg) == 3: star += wv
        if sum(a) == 3 and max(deg) == 2: shares["path3"] = shares.get("path3", 0.0) + wv
        if sum(a) == 4: shares["order4"] = shares.get("order4", 0.0) + wv
        if sum(a) == 5: shares["order5"] = shares.get("order5", 0.0) + wv
        if sum(a) >= 6: shares["order6plus"] = shares.get("order6plus", 0.0) + wv
        if sum(a) == 4 and sorted(a) == [0, 0, 1, 1, 1, 1] and max(deg) == 2: shares["cycle4"] = shares.get("cycle4", 0.0) + wv
    return tot, star
out = {}
for lab, r in [("rho~0.3", 0.3), ("rho~0.5", 0.5)]:
    rng = np.random.default_rng(3)
    m = np.array([0.2, -0.1, 0.3, 0.0]); v = np.array([1.0, 0.8, 1.2, 0.9])
    Rm = np.full((4, 4), r); np.fill_diagonal(Rm, 1.0); Rm[0, 3] = Rm[3, 0] = r * 0.7; Rm[1, 2] = Rm[2, 1] = r * 1.2
    C = Rm * np.sqrt(np.outer(v, v))
    ex = kappa4_gh(m, C); tot, star = series4(m, C)
    out[lab] = {"gh_exact": ex, "series_all_le8": tot, "rel_err": abs(tot - ex) / abs(ex), "star_share": star / tot, **{k + "_share": v / tot for k, v in shares.items()}}
    print(lab, json.dumps(out[lab]), flush=True)
