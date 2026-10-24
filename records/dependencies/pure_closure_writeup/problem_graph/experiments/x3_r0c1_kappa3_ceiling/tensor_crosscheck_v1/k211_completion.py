#!/usr/bin/env python3
"""TAG r0c1-k211-completion-v1: the semi-distinct connected kappa4[i,i,j,k]
(distinct i, j, k) - the named dominant missing content of the fourth-order
push - and its pushed contributions to K22/K31/k4 of the next layer.

Method: linked-cluster over FOUR nodes (two copies of unit i, one j, one k)
with the copy-copy bond at rho_ii = 1 resummed explicitly (t_ii up to
TII_CAP, 1/t! convergent), external bonds capped at EXT_CAP total. A term
is kept iff its bond graph connects all four nodes. Node columns are c1
(a[t] = E[f He_t]) throughout - the repeated unit appears as two c1 nodes,
whose rho=1 resummation reproduces the exact E[f^2 ...] content.

Gate A (formula): connected kappa4(f_i,f_i,f_j,f_k) vs 3D Gauss-Hermite
quadrature on synthetic Gaussian triples.
Gate B (end-to-end): pushed contributions at L0 -> pre-1 K22/K31 vs 64M
truth, on top of the plane-supported push + tree completion.
Usage: k211_completion.py [gateA|gateB|both]"""
import itertools
import json
import sys
from math import erf, factorial

import numpy as np

_erf = np.vectorize(erf)
TII_CAP = 10
EXT_CAP = 4

# nodes: 0,1 = the two copies of unit i; 2 = j; 3 = k
# bond slots: (0,1)=ii, (0,2),(1,2)=i-j, (0,3),(1,3)=i-k, (2,3)=j-k
BONDS = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]


def a_columns(m, v, qmax=18):
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


def connected(ts):
    parent = list(range(4))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (p, q), t in zip(BONDS, ts):
        if t:
            rp, rq = find(p), find(q)
            if rp != rq:
                parent[rp] = rq
    return len({find(c) for c in range(4)}) == 1


def enumerate_terms():
    """(t_ii, t_02, t_12, t_03, t_13, t_23) with external total <= EXT_CAP,
    t_ii <= TII_CAP, graph connected; every node degree >= 1."""
    out = []
    for tii in range(0, TII_CAP + 1):
        for ext in itertools.product(range(EXT_CAP + 1), repeat=5):
            if sum(ext) > EXT_CAP:
                continue
            ts = (tii,) + ext
            deg = [0, 0, 0, 0]
            for (p, q), t in zip(BONDS, ts):
                deg[p] += t
                deg[q] += t
            if min(deg) == 0:
                continue
            if not connected(ts):
                continue
            coef = 1.0
            for t in ts:
                coef /= factorial(t)
            out.append((ts, tuple(deg), coef))
    return out


def kappa4_211_scalar(mi, vi, mj, vj, mk, vk, rij, rik, rjk, terms):
    ci = a_columns(np.array([mi]), np.array([vi]))[:, 0]
    cj = a_columns(np.array([mj]), np.array([vj]))[:, 0]
    ck = a_columns(np.array([mk]), np.array([vk]))[:, 0]
    tot = 0.0
    for (ts, deg, coef) in terms:
        tii, t02, t12, t03, t13, t23 = ts
        val = (coef * rij ** (t02 + t12) * rik ** (t03 + t13)
               * rjk ** t23
               * ci[deg[0]] * ci[deg[1]] * cj[deg[2]] * ck[deg[3]])
        tot += val
    return tot


def gateA():
    gh_x, gh_w = np.polynomial.hermite_e.hermegauss(90)
    gh_w = gh_w / np.sqrt(2 * np.pi)
    rng = np.random.default_rng(21)
    terms = enumerate_terms()
    print(json.dumps({"n_terms": len(terms)}), flush=True)
    worst = 0.0
    for trial in range(5):
        mm = rng.normal(0, 0.6, 3)
        sd0 = np.exp(rng.normal(0, 0.25, 3))
        while True:
            r = rng.uniform(-0.4, 0.4, 3)
            R = np.array([[1, r[0], r[1]], [r[0], 1, r[2]],
                          [r[1], r[2], 1]])
            if np.linalg.eigvalsh(R).min() > 0.1:
                break
        L = np.linalg.cholesky(R)
        X, Y, Z = np.meshgrid(gh_x, gh_x, gh_x, indexing="ij")
        G = L @ np.stack([X.ravel(), Y.ravel(), Z.ravel()])
        WT = (gh_w[:, None, None] * gh_w[None, :, None]
              * gh_w[None, None, :]).ravel()
        f = [np.maximum(mm[c] + sd0[c] * G[c], 0.0) for c in range(3)]
        E = lambda a: float(np.sum(WT * a))
        e1 = [E(fc) for fc in f]
        fc_ = [f[c] - e1[c] for c in range(3)]
        mu211 = E(fc_[0] ** 2 * fc_[1] * fc_[2])
        v0 = E(fc_[0] ** 2)
        c01 = E(fc_[0] * fc_[1])
        c02 = E(fc_[0] * fc_[2])
        c12 = E(fc_[1] * fc_[2])
        k4_true = mu211 - v0 * c12 - 2 * c01 * c02
        k4_mine = kappa4_211_scalar(mm[0], sd0[0] ** 2 * 1.0,
                                    mm[1], sd0[1] ** 2,
                                    mm[2], sd0[2] ** 2,
                                    R[0, 1], R[0, 2], R[1, 2], terms)
        # NOTE: variances passed are the LATENT ones; f-vars via columns
        rel = abs(k4_mine - k4_true) / max(abs(k4_true), 1e-14)
        worst = max(worst, rel)
        print(json.dumps({"k4_true": k4_true, "k4_mine": k4_mine,
                          "rel": round(rel, 4)}), flush=True)
    print(json.dumps({"gateA_worst_rel": round(worst, 4)}), flush=True)
    return worst


def push_k211(m, v, C, Wl):
    """Pushed contributions of the semi-distinct completion to
    (k4next, K22next, K31next). Per Kibble term the (i,j,k)-field
    factorizes into node vectors and pair rho-power matrices; each
    position-pattern is one 3-node contraction. Multiplicities:
      K22 (aabb): 1x[W2a; j->b,k->b] + 1x[W2b; j->a,k->a]
                  + 4x[WaWb; j->a,k->b]
      K31 (aaab): 3x[W2a; j->a,k->b] + 3x[WaWb; j->a,k->a]
      k4  (aaaa): 6x[W2a; j->a,k->a]
    v1 approximations (measured in gate B): triangle terms with all three
    of (i-j, i-k, j-k) bonded are skipped; index collisions j=i / k=i in
    u=0 or w=0 patterns are not subtracted (rho diag is zeroed, which
    kills bonded collisions)."""
    W_ = m.size
    sd = np.sqrt(np.maximum(v, 1e-30))
    a = a_columns(m, v)
    rho = C / np.outer(sd, sd)
    np.fill_diagonal(rho, 0.0)
    rmax = 2 * EXT_CAP
    rp = {t: rho ** t for t in range(1, rmax + 1)}
    # group terms by external bond pattern, collapsing the tii resummation
    # into one effective fi vector per pattern (11x fewer contractions)
    grouped = {}
    for (ts, deg, coef) in enumerate_terms():
        tii, t02, t12, t03, t13, t23 = ts
        key = (t02, t12, t03, t13, t23)
        vec = coef * a[deg[0]] * a[deg[1]]
        if key in grouped:
            grouped[key] = grouped[key] + vec
        else:
            grouped[key] = vec.copy()
    terms = [((0,) + key,
              (0, 0, key[0] + key[1] + key[4], key[2] + key[3] + key[4]),
              1.0, fi_eff) for key, fi_eff in grouped.items()]
    W2 = Wl * Wl
    K22n = np.zeros((W_, W_))
    K31n = np.zeros((W_, W_))
    k4n = np.zeros(W_)

    def contract(fi, u, fj, w, fk, t23, ilegs, jslot, kslot):
        """Hand-staged contractions; every step is <=256^3."""
        Fj = fj[:, None] * Wl
        Fk = fk[:, None] * Wl
        if t23 == 0:
            if u == 0 or w == 0:
                return None
            Bj = rp[u] @ Fj
            Bk = rp[w] @ Fk
            # j=k collision: those tuples are kappa4[i,i,j,j] (plane class,
            # already in the plane-supported push) - subtract the diagonal
            # slice from the product Bj*Bk wherever both legs share the
            # collision unit.
            Rc = rp[u] * rp[w] if (u + w) <= rmax else rho ** (u + w)
            Cjk = Rc @ ((fj * fk)[:, None] * (Wl * Wl))     # shared-slot
            CjkM = None
        elif u and not w:
            X = Fj * (rp[t23] @ Fk)            # [j, s] both legs on j
            Bj, Bk = None, None
            BjX = rp[u] @ (fj[:, None] * Wl)   # unused placeholder
            # j carries its own leg and k's chained leg with SAME slot only
            # when jslot == kslot; for mixed slots build both:
            Bsame = rp[u] @ X                  # [i, s] with s = shared slot
            Bj_only = rp[u] @ Fj
            Bk_chain = rp[u] @ (fj[:, None] * (rp[t23] @ Fk))
        elif w and not u:
            Bsame = rp[w] @ (Fk * (rp[t23] @ Fj))
            Bj_only = rp[w] @ Fk
            Bk_chain = rp[w] @ (fk[:, None] * (rp[t23] @ Fj))
        else:
            return None
        W2l = W2

        def out_ab(vi_a, vi_b):
            return (vi_a.T @ vi_b)             # [a, b]

        if t23 == 0:
            if ilegs == "aa" and jslot == "b" and kslot == "b":
                X = Bj * Bk - Cjk
                return (fi[:, None] * W2l).T @ X
            if ilegs == "bb":
                X = Bj * Bk - Cjk
                return ((fi[:, None] * W2l).T @ X).T
            if ilegs == "ab" and jslot == "a" and kslot == "b":
                # collision j=k puts both legs on DIFFERENT slots of the
                # same unit: subtract sum_j rho^{u+w} fj fk W[j,a]W[j,b]
                # -> a rank-style correction per i then contracted:
                A = Wl * Bj
                B = Wl * Bk
                out = (fi[:, None] * A).T @ B
                colM = Rc @ ((fj * fk)[:, None] * Wl)       # [i, s]
                # subtract sum_i fi W[i,a]W[i,b] * [collision over shared]
                # exact: sum_{i,j} fi W[i,a]W[i,b] rho^{u+w} fj fk W[j,a]W[j,b]
                g = fj * fk
                Zc = (g[:, None, None] * Wl[:, :, None]
                      * Wl[:, None, :]).reshape(W_, -1)
                Yc = (Rc @ Zc).reshape(W_, Wl.shape[1], Wl.shape[1])
                out -= np.einsum("i,ia,ib,iab->ab", fi, Wl, Wl, Yc,
                                 optimize="greedy")
                return out
            if ilegs == "aa" and jslot == "a" and kslot == "b":
                A = W2l * Bj
                out = (fi[:, None] * A).T @ Bk
                g = fj * fk
                Zc = (g[:, None, None] * Wl[:, :, None]
                      * Wl[:, None, :]).reshape(W_, -1)
                Yc = (Rc @ Zc).reshape(W_, Wl.shape[1], Wl.shape[1])
                out -= np.einsum("i,ia,iab->ab", fi, W2l, Yc,
                                 optimize="greedy")
                return out
            if ilegs == "ab" and jslot == "a" and kslot == "a":
                X = Bj * Bk - Cjk
                return (fi[:, None] * (Wl * X)).T @ Wl
            if ilegs == "aa" and jslot == "a" and kslot == "a":
                return np.einsum("i,ia->a", fi, W2l * (Bj * Bk - Cjk))
            raise ValueError(ilegs + jslot + kslot)
        # chained cases: j (or k) hub carries the t23 chain
        if u and not w:
            hubW = Fj
            chain = rp[t23] @ Fk
        else:
            hubW = Fk
            chain = rp[t23] @ Fj
        # hub vector for (hub slot s1, chain slot s2):
        #   H[s1,s2][h] = hubW[h,s1]*chain[h,s2] -> Bmix[i,s1,s2] via rho^r
        r_ = u if u else w
        if ilegs == "aa" and jslot == "b" and kslot == "b":
            X = hubW * chain                   # both on b: [h, b]
            Bx = rp[r_] @ X
            return (fi[:, None] * W2l).T @ Bx
        if ilegs == "bb":
            X = hubW * chain
            Bx = rp[r_] @ X
            return ((fi[:, None] * W2l).T @ Bx).T
        if ilegs == "aa" and ((jslot == "a" and kslot == "b")
                              or (jslot == "b" and kslot == "a")):
            # mixed: need Bmix[i,a,b] = sum_h rho^r hubW[h,a]chain[h,b]
            # (or swapped roles). Stage as per-a matmul: [i,a,b] too big?
            # 256^3 f64 = 134MB - acceptable once per call:
            hub_a = hubW if (u and jslot == "a") or (w and kslot == "a")                 else chain
            hub_b = chain if hub_a is hubW else hubW
            Bmix = np.einsum("ih,ha,hb->iab", rp[r_], hub_a, hub_b,
                             optimize=True)
            return np.einsum("i,ia,iab->ab", fi, W2l, Bmix, optimize=True)
        if ilegs == "ab" and jslot == "a" and kslot == "a":
            X = hubW * chain
            Bx = rp[r_] @ X                     # [i, a] both hub legs on a
            return (fi[:, None] * (Wl * Bx)).T @ Wl
        if ilegs == "ab" and jslot == "a" and kslot == "b":
            hub_a = hubW if u else chain
            hub_b = chain if u else hubW
            Bmix = np.einsum("ih,ha,hb->iab", rp[r_], hub_a, hub_b,
                             optimize=True)
            return np.einsum("i,ia,ib,iab->ab", fi, Wl, Wl, Bmix,
                             optimize=True)
        if ilegs == "aa" and jslot == "a" and kslot == "a":
            X = hubW * chain
            Bx = rp[r_] @ X
            return np.einsum("i,ia->a", fi, W2l * Bx)
        raise ValueError(ilegs + jslot + kslot)

    for (ts, deg, coef, fi_eff) in terms:
        tii, t02, t12, t03, t13, t23 = ts
        u = t02 + t12
        w = t03 + t13
        if u > rmax or w > rmax:
            continue
        fi = fi_eff
        fj = a[deg[2]]
        fk = a[deg[3]]
        import os as _os
        _skip_ab = _os.environ.get("K211_NO_AB") == "1"
        specs = [
            ("K22", 1.0, "aa", "b", "b"), ("K22", 1.0, "bb", "a", "a"),
        ] + ([] if _skip_ab else [("K22", 4.0, "ab", "a", "b")]) + [
            ("K31", 3.0, "aa", "a", "b"), ("K31", 3.0, "ab", "a", "a"),
            ("k4", 6.0, "aa", "a", "a"),
        ]
        for (target, mult, ilegs, js, ks) in specs:
            val = contract(fi, u, fj, w, fk, t23, ilegs, js, ks)
            if val is None:
                continue
            if target == "K22":
                K22n += mult * val
            elif target == "K31":
                K31n += mult * val
            else:
                k4n += mult * val
    np.fill_diagonal(K22n, 0.0)
    np.fill_diagonal(K31n, 0.0)
    return k4n, K22n, K31n


def gateB():
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
    k4d, K22d, K31d = push_k211(m, v, C, W1)
    tK22 = mcv4["pre_K22"][1].astype(np.float64)
    tK31 = mcv4["pre_K31"][1].astype(np.float64)
    np.fill_diagonal(tK22, 0.0)
    np.fill_diagonal(tK31, 0.0)
    s22 = float(np.sqrt(np.mean(tK22 ** 2)))
    s31 = float(np.sqrt(np.mean(tK31 ** 2)))
    out = {
        "K22_plane_only": round(float(np.sqrt(np.mean(
            (K22b - tK22) ** 2))) / s22, 4),
        "K22_plus_k211": round(float(np.sqrt(np.mean(
            (K22b + K22d - tK22) ** 2))) / s22, 4),
        "K31_plane_only": round(float(np.sqrt(np.mean(
            (K31b - tK31) ** 2))) / s31, 4),
        "K31_plus_k211": round(float(np.sqrt(np.mean(
            (K31b + K31d - tK31) ** 2))) / s31, 4),
    }
    print(json.dumps(out, indent=1), flush=True)
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode in ("gateA", "both"):
        gateA()
    if mode in ("gateB", "both"):
        gateB()
