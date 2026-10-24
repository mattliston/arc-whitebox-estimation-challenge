#!/usr/bin/env python
"""
TAG: public_trivariate_derivation_v1/verify.py
Verifies DERIVATION.md exactly, term by term.

  kappa211  = kappa(x_i,x_i,x_j,x_k),  kappa1111 = kappa(x_i,x_j,x_k,x_l),
  x_u = relu(mu_u + sd_u z_u), z jointly standard Gaussian with correlations rho.

Instruments (all numpy + math only):
  c-tables   : exact upper-incomplete-moment recursion (authoritative),
               Gauss-Hermite hermgauss(200) (task-specified), Stein closed forms.
  formula    : all-orders master series (DERIVATION.md (3.2)/(4.3)), truncated at T.
  truth #1   : deterministic kink-split nested Gauss-Legendre quadrature (~1e-9).
  truth #2   : batched MC, exact-mean centering, batch-spread SE.

Outputs: FORMULA.json, VERIFICATION.md, stdout log.
Knobs: --quick (N/64, coarser quad) for smoke runs.  Seed base 20260824.
"""
import json
import math
import sys
import time
from fractions import Fraction
from math import factorial

import numpy as np

HERE = "/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_trivariate_derivation_v1"
QMAX = 14          # max Hermite index carried in every c-table
TFULL = 14         # "all-orders" reference truncation
SEED_BASE = 20260824

_erf = np.frompyfunc(math.erf, 1, 1)


def Phi(x):
    if np.isscalar(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    return 0.5 * (1.0 + _erf(np.asarray(x) / math.sqrt(2.0)).astype(np.float64))


def phi(x):
    return np.exp(-0.5 * np.square(x)) / math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------- Hermite (He, probabilists')
def he_coeff_list(qmax):
    """Monomial coefficient arrays (low->high) for He_0..He_qmax."""
    out = [np.array([1.0]), np.array([0.0, 1.0])]
    for q in range(1, qmax):
        a = np.zeros(q + 2)
        a[1:] += out[q]
        b = q * out[q - 1]
        a[: len(b)] -= b
        out.append(a)
    return out


HE = he_coeff_list(QMAX + 4)


def he_val(q, x):
    from numpy.polynomial import hermite_e
    c = np.zeros(q + 1)
    c[q] = 1.0
    return hermite_e.hermeval(x, c)


# ---------------------------------------------------------------- c-tables
def ctable_exact(mu, sd, qmax=QMAX):
    """c[p][q] = E[relu(mu+sd z)^p He_q(z)], p=1,2, via I_n = int_w^inf z^n phi dz."""
    w = -mu / sd
    nmax = qmax + 3
    I = np.zeros(nmax + 1)
    I[0] = Phi(-w)
    pw = phi(w)
    if nmax >= 1:
        I[1] = pw
    for n in range(2, nmax + 1):
        I[n] = w ** (n - 1) * pw + (n - 1) * I[n - 2]
    lin = np.array([mu, sd])                       # (mu + sd z)
    pows = {1: lin, 2: np.convolve(lin, lin)}
    c = {1: np.zeros(qmax + 1), 2: np.zeros(qmax + 1)}
    for p in (1, 2):
        for q in range(qmax + 1):
            poly = np.convolve(pows[p], HE[q])
            c[p][q] = float(np.dot(poly, I[: len(poly)]))
    return c


def ctable_hermgauss(mu, sd, qmax=QMAX, nodes=200):
    xg, wg = np.polynomial.hermite.hermgauss(nodes)
    z = math.sqrt(2.0) * xg
    wnorm = wg / math.sqrt(math.pi)
    x = np.maximum(mu + sd * z, 0.0)
    c = {1: np.zeros(qmax + 1), 2: np.zeros(qmax + 1)}
    for q in range(qmax + 1):
        hq = he_val(q, z)
        c[1][q] = float(np.sum(wnorm * x * hq))
        c[2][q] = float(np.sum(wnorm * x * x * hq))
    return c


def ctable_stein(mu, sd, qmax=QMAX):
    """Independent closed forms (integration by parts); matches estimator _q38_coefficients."""
    w = -mu / sd
    pw = phi(w)
    Pm = Phi(-w)
    c1 = np.zeros(qmax + 1)
    c2 = np.zeros(qmax + 1)
    c1[0] = mu * Pm + sd * pw
    c1[1] = sd * Pm
    for q in range(2, qmax + 1):
        c1[q] = sd * he_val(q - 2, w) * pw
    c2[0] = (mu * mu + sd * sd) * Pm + mu * sd * pw
    c2[1] = 2.0 * sd * c1[0]
    c2[2] = 2.0 * sd * sd * Pm
    for q in range(3, qmax + 1):
        c2[q] = 2.0 * sd * sd * he_val(q - 3, w) * pw
    return {1: c1, 2: c2}


def unit_AB(mu, sd, source="exact"):
    """(m, A[0..QMAX], B[0..QMAX]) with A_0=0, B_0=Var."""
    c = {"exact": ctable_exact, "hermgauss": ctable_hermgauss, "stein": ctable_stein}[source](mu, sd)
    m = c[1][0]
    A = c[1].copy()
    A[0] = 0.0
    B = np.empty(QMAX + 1)
    B[0] = c[2][0] - m * m
    B[1:] = c[2][1:] - 2.0 * m * c[1][1:]
    return m, A, B


# ---------------------------------------------------------------- master series
def kappa211_series(Ai, Bi, J, K, rij, rik, rjk, tmax=TFULL):
    """DERIVATION (3.2). Returns {total_order: contribution}."""
    per = {}
    for a in range(tmax + 1):
        for b in range(tmax + 1 - a):
            if a + b == 0:
                continue
            for t in range(tmax + 1 - a - b):
                ju, ku = a + t, b + t
                if ju == 0 or ku == 0 or a + b > QMAX or ju > QMAX or ku > QMAX:
                    continue
                kern = Bi[a + b] - (2.0 * Ai[a] * Ai[b] if t == 0 else 0.0)
                val = (rij ** a) * (rik ** b) * (rjk ** t) \
                    / (factorial(a) * factorial(b) * factorial(t)) * kern * J[ju] * K[ku]
                per[a + b + t] = per.get(a + b + t, 0.0) + val
    return per


EDGES4 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def _connected_covering(ts):
    sup = [e for e, t in zip(EDGES4, ts) if t > 0]
    seen = set()
    for e in sup:
        seen.update(e)
    if seen != {0, 1, 2, 3}:
        return False
    parent = list(range(4))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in sup:
        parent[find(u)] = find(v)
    return len({find(u) for u in range(4)}) == 1


def _tuples_sum_le(k, total):
    if k == 1:
        for t in range(total + 1):
            yield (t,)
        return
    for t in range(total + 1):
        for rest in _tuples_sum_le(k - 1, total - t):
            yield (t,) + rest


def kappa1111_series(alphas, R, tmax=10):
    """DERIVATION (4.3). alphas: list of 4 arrays A_q (A_0=0). R: 4x4 corr matrix."""
    per = {}
    for ts in _tuples_sum_le(6, tmax):
        if not _connected_covering(ts):
            continue
        s = [0, 0, 0, 0]
        w = 1.0
        for (u, v), t in zip(EDGES4, ts):
            s[u] += t
            s[v] += t
            if t:
                w *= (R[u, v] ** t) / factorial(t)
        if max(s) > QMAX:
            continue
        for u in range(4):
            w *= alphas[u][s[u]]
        per[sum(ts)] = per.get(sum(ts), 0.0) + w
    return per


def truncate(per, T):
    return sum(v for o, v in per.items() if o <= T)


# ---------------------------------------------------------------- FORMULA.json term lists
def kappa211_terms(orders=(2, 3, 4)):
    terms = []
    for a in range(max(orders) + 1):
        for b in range(max(orders) + 1 - a):
            if a + b == 0:
                continue
            for t in range(max(orders) + 1 - a - b):
                o = a + b + t
                if o not in orders or a + t == 0 or b + t == 0:
                    continue
                coeff = Fraction(1, factorial(a) * factorial(b) * factorial(t))
                if t == 0:
                    ifun = f"B[{a + b}] - 2*A[{a}]*A[{b}]"
                else:
                    ifun = f"B[{a + b}]"
                terms.append({
                    "order": o,
                    "rho_monomial": f"r_ij^{a} * r_ik^{b} * r_jk^{t}",
                    "exponents": {"r_ij": a, "r_ik": b, "r_jk": t},
                    "coefficient": str(coeff),
                    "i_functional": ifun,
                    "j_functional": f"J[{a + t}]",
                    "k_functional": f"K[{b + t}]",
                })
    return terms


def kappa1111_terms(orders=(3, 4)):
    names = "ijkl"
    terms = []
    for ts in _tuples_sum_le(6, max(orders)):
        o = sum(ts)
        if o not in orders or not _connected_covering(ts):
            continue
        s = [0, 0, 0, 0]
        den = 1
        expo = {}
        for (u, v), t in zip(EDGES4, ts):
            s[u] += t
            s[v] += t
            den *= factorial(t)
            expo[f"r_{names[u]}{names[v]}"] = t
        terms.append({
            "order": o,
            "exponents": expo,
            "rho_monomial": " * ".join(f"{e}^{t}" for e, t in expo.items() if t),
            "coefficient": str(Fraction(1, den)),
            "vertex_functionals": {names[u]: f"alpha[{s[u]}]" for u in range(4)},
        })
    return terms


# ---------------------------------------------------------------- deterministic quadrature truth
ZCUT = 9.0


def _gl_cache(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def _split_intervals(c):
    """Two intervals [-Z,c],[c,Z] with c clipped; empty -> zero width."""
    c = np.clip(c, -ZCUT, ZCUT)
    return (-ZCUT, c), (c, ZCUT)


def _map_nodes(lo, hi, xg, wg):
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    nodes = mid[..., None] + half[..., None] * xg
    weights = half[..., None] * wg
    return nodes, weights


def g_relu(A, B):
    """E[(A + B*eta)_+] over eta~N(0,1), B>0 (closed form)."""
    r = A / B
    return A * Phi(r) + B * phi(r)


def quad_cov(mu1, sd1, mu2, sd2, r, n=200):
    """Cov(x1,x2) by outer split-GL over z1 + closed-form inner for x2|z1."""
    m1, A1_, B1_ = unit_AB(mu1, sd1)
    m2 = unit_AB(mu2, sd2)[0]
    xg, wg = _gl_cache(n)
    tot = 0.0
    for lo, hi in _split_intervals(np.array(-mu1 / sd1)):
        z, w = _map_nodes(lo, hi, xg, wg)
        x1 = np.maximum(mu1 + sd1 * z, 0.0) - m1
        A = mu2 + sd2 * r * z
        Bc = sd2 * math.sqrt(max(1.0 - r * r, 1e-300))
        x2 = g_relu(A, Bc) - m2
        tot += float(np.sum(w * phi(z) * x1 * x2))
    return tot


def quad_kappa211(mus, sds, R, n=80):
    """Deterministic kappa211: 2-level kink-split GL + closed-form innermost (k)."""
    (mi, mj, mk), (si, sj, sk) = mus, sds
    L = np.linalg.cholesky(R)
    m_i = unit_AB(mi, si)[0]
    m_j = unit_AB(mj, sj)[0]
    m_k = unit_AB(mk, sk)[0]
    xg, wg = _gl_cache(n)
    p211 = 0.0
    for lo1, hi1 in _split_intervals(np.array(-mi / si)):
        e1, w1 = _map_nodes(lo1, hi1, xg, wg)          # (n,)
        e1, w1 = e1.ravel(), w1.ravel()
        xi2 = np.square(np.maximum(mi + si * e1, 0.0) - m_i) * w1 * phi(e1)
        c2 = (-mj / sj - L[1, 0] * e1) / L[1, 1]       # inner split point per outer node
        inner = 0.0
        for piece in range(2):
            lo2 = np.full_like(e1, -ZCUT) if piece == 0 else np.clip(c2, -ZCUT, ZCUT)
            hi2 = np.clip(c2, -ZCUT, ZCUT) if piece == 0 else np.full_like(e1, ZCUT)
            e2, w2 = _map_nodes(lo2, hi2, xg, wg)      # (n, n)
            zj = L[1, 0] * e1[:, None] + L[1, 1] * e2
            xj = np.maximum(mj + sj * zj, 0.0) - m_j
            Ak = mk + sk * (L[2, 0] * e1[:, None] + L[2, 1] * e2)
            Bk = sk * L[2, 2]
            xk = g_relu(Ak, Bk) - m_k
            inner = inner + np.sum(w2 * phi(e2) * xj * xk, axis=1)
        p211 += float(np.sum(xi2 * inner))
    v_i = unit_AB(mi, si)[2][0]
    cjk = quad_cov(mj, sj, mk, sk, R[1, 2])
    cij = quad_cov(mi, si, mj, sj, R[0, 1])
    cik = quad_cov(mi, si, mk, sk, R[0, 2])
    return p211 - v_i * cjk - 2.0 * cij * cik


def quad_kappa1111(mus, sds, R, n=48):
    """Deterministic kappa1111: 3-level kink-split GL + closed-form innermost (l)."""
    m = [unit_AB(mu, sd)[0] for mu, sd in zip(mus, sds)]
    L = np.linalg.cholesky(R)
    xg, wg = _gl_cache(n)
    p1111 = 0.0
    for lo1, hi1 in _split_intervals(np.array(-mus[0] / sds[0])):
        e1, w1 = _map_nodes(lo1, hi1, xg, wg)
        e1, w1 = e1.ravel(), w1.ravel()
        f1 = (np.maximum(mus[0] + sds[0] * e1, 0.0) - m[0]) * w1 * phi(e1)   # (n1,)
        c2 = (-mus[1] / sds[1] - L[1, 0] * e1) / L[1, 1]
        acc1 = np.zeros_like(e1)
        for piece2 in range(2):
            lo2 = np.full_like(e1, -ZCUT) if piece2 == 0 else np.clip(c2, -ZCUT, ZCUT)
            hi2 = np.clip(c2, -ZCUT, ZCUT) if piece2 == 0 else np.full_like(e1, ZCUT)
            e2, w2 = _map_nodes(lo2, hi2, xg, wg)                            # (n1,n2)
            zj = L[1, 0] * e1[:, None] + L[1, 1] * e2
            f2 = (np.maximum(mus[1] + sds[1] * zj, 0.0) - m[1]) * w2 * phi(e2)
            c3 = (-mus[2] / sds[2] - L[2, 0] * e1[:, None] - L[2, 1] * e2) / L[2, 2]
            acc2 = np.zeros_like(e2)
            for piece3 in range(2):
                lo3 = np.full_like(e2, -ZCUT) if piece3 == 0 else np.clip(c3, -ZCUT, ZCUT)
                hi3 = np.clip(c3, -ZCUT, ZCUT) if piece3 == 0 else np.full_like(e2, ZCUT)
                e3, w3 = _map_nodes(lo3, hi3, xg, wg)                        # (n1,n2,n3)
                zk = L[2, 0] * e1[:, None, None] + L[2, 1] * e2[..., None] + L[2, 2] * e3
                f3 = (np.maximum(mus[2] + sds[2] * zk, 0.0) - m[2]) * w3 * phi(e3)
                Al = mus[3] + sds[3] * (L[3, 0] * e1[:, None, None]
                                        + L[3, 1] * e2[..., None] + L[3, 2] * e3)
                Bl = sds[3] * L[3, 3]
                f4 = g_relu(Al, Bl) - m[3]
                acc2 = acc2 + np.sum(f3 * f4, axis=-1)
            acc1 = acc1 + np.sum(f2 * acc2, axis=-1)
        p1111 += float(np.sum(f1 * acc1))
    c = {}
    for (u, v) in EDGES4:
        c[(u, v)] = quad_cov(mus[u], sds[u], mus[v], sds[v], R[u, v])
    return p1111 - c[(0, 1)] * c[(2, 3)] - c[(0, 2)] * c[(1, 3)] - c[(0, 3)] * c[(1, 2)]


# ---------------------------------------------------------------- MC
def mc_kappa211(mus, sds, R, N, seed, batch=5_000_000):
    L = np.linalg.cholesky(R)
    m = np.array([unit_AB(mu, sd)[0] for mu, sd in zip(mus, sds)])
    mu = np.array(mus)
    sd = np.array(sds)
    nb = max(int(round(N / batch)), 8)
    rng = np.random.default_rng([SEED_BASE, seed])
    rows = np.empty((nb, 5))
    for ib in range(nb):
        z = rng.standard_normal((batch, 3)) @ L.T
        x = np.maximum(mu + sd * z, 0.0) - m
        xi, xj, xk = x[:, 0], x[:, 1], x[:, 2]
        xi2 = xi * xi
        rows[ib] = (np.mean(xi2 * xj * xk), np.mean(xi2), np.mean(xj * xk),
                    np.mean(xi * xj), np.mean(xi * xk))
    pooled = rows.mean(axis=0)
    kap = pooled[0] - pooled[1] * pooled[2] - 2.0 * pooled[3] * pooled[4]
    per_batch = rows[:, 0] - rows[:, 1] * rows[:, 2] - 2.0 * rows[:, 3] * rows[:, 4]
    se = float(np.std(per_batch, ddof=1) / math.sqrt(nb))
    return float(kap), se, nb * batch


def mc_kappa1111(mus, sds, R, N, seed, batch=5_000_000):
    L = np.linalg.cholesky(R)
    m = np.array([unit_AB(mu, sd)[0] for mu, sd in zip(mus, sds)])
    mu = np.array(mus)
    sd = np.array(sds)
    nb = max(int(round(N / batch)), 8)
    rng = np.random.default_rng([SEED_BASE, seed])
    rows = np.empty((nb, 7))
    for ib in range(nb):
        z = rng.standard_normal((batch, 4)) @ L.T
        x = np.maximum(mu + sd * z, 0.0) - m
        a, b, c_, d = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        rows[ib] = (np.mean(a * b * c_ * d),
                    np.mean(a * b), np.mean(c_ * d),
                    np.mean(a * c_), np.mean(b * d),
                    np.mean(a * d), np.mean(b * c_))
    p = rows.mean(axis=0)
    kap = p[0] - p[1] * p[2] - p[3] * p[4] - p[5] * p[6]
    per_batch = rows[:, 0] - rows[:, 1] * rows[:, 2] - rows[:, 3] * rows[:, 4] - rows[:, 5] * rows[:, 6]
    se = float(np.std(per_batch, ddof=1) / math.sqrt(nb))
    return float(kap), se, nb * batch


# ---------------------------------------------------------------- unit tests
def R3(rij, rik, rjk):
    return np.array([[1.0, rij, rik], [rij, 1.0, rjk], [rik, rjk, 1.0]])


def R4(rij, rik, ril, rjk, rjl, rkl):
    return np.array([[1.0, rij, rik, ril],
                     [rij, 1.0, rjk, rjl],
                     [rik, rjk, 1.0, rkl],
                     [ril, rjl, rkl, 1.0]])


def hermite_identity_value(svec, R):
    """(1.1) evaluated by brute-force enumeration of bond matrices."""
    n = len(svec)
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]

    def rec(idx, rows, acc):
        if idx == len(pairs):
            return acc if all(r == 0 for r in rows) else 0.0
        a, b = pairs[idx]
        tot = 0.0
        for t in range(0, min(rows[a], rows[b]) + 1):
            rows2 = list(rows)
            rows2[a] -= t
            rows2[b] -= t
            tot += rec(idx + 1, rows2, acc * (R[a, b] ** t) / factorial(t))
        return tot
    base = 1.0
    for s in svec:
        base *= factorial(s)
    return base * rec(0, list(svec), 1.0)


def unit_tests(log):
    ok = True
    # U1: Hermite product identity vs tensor Gauss-Hermite (polynomials -> GH exact)
    rng = np.random.default_rng(7)
    for trial in range(3):
        A = rng.standard_normal((3, 3))
        C = A @ A.T
        d = np.sqrt(np.diag(C))
        R = C / np.outer(d, d)
        Lc = np.linalg.cholesky(R)
        xg, wg = np.polynomial.hermite.hermgauss(16)
        z1 = math.sqrt(2.0) * xg
        wn = wg / math.sqrt(math.pi)
        G = np.stack(np.meshgrid(z1, z1, z1, indexing="ij"), axis=-1) @ Lc.T
        W = wn[:, None, None] * wn[None, :, None] * wn[None, None, :]
        worst = 0.0
        for s in [(1, 1, 0), (2, 1, 1), (3, 2, 1), (2, 2, 2), (4, 3, 3), (5, 4, 3)]:
            lhs = float(np.sum(W * he_val(s[0], G[..., 0]) * he_val(s[1], G[..., 1])
                               * he_val(s[2], G[..., 2])))
            rhs = hermite_identity_value(s, R)
            worst = max(worst, abs(lhs - rhs) / max(1.0, abs(rhs)))
        ok &= worst < 1e-9
        log(f"U1 trial {trial}: Hermite product identity max rel dev {worst:.2e}")
    # U1b: 4-variable
    A = rng.standard_normal((4, 4))
    C = A @ A.T
    d = np.sqrt(np.diag(C))
    R = C / np.outer(d, d)
    Lc = np.linalg.cholesky(R)
    xg, wg = np.polynomial.hermite.hermgauss(10)
    z1 = math.sqrt(2.0) * xg
    wn = wg / math.sqrt(math.pi)
    G = np.stack(np.meshgrid(z1, z1, z1, z1, indexing="ij"), axis=-1) @ Lc.T
    W = (wn[:, None, None, None] * wn[None, :, None, None]
         * wn[None, None, :, None] * wn[None, None, None, :])
    worst = 0.0
    for s in [(1, 1, 1, 1), (2, 1, 2, 1), (3, 1, 1, 1), (2, 2, 2, 2)]:
        lhs = float(np.sum(W * he_val(s[0], G[..., 0]) * he_val(s[1], G[..., 1])
                           * he_val(s[2], G[..., 2]) * he_val(s[3], G[..., 3])))
        rhs = hermite_identity_value(s, R)
        worst = max(worst, abs(lhs - rhs) / max(1.0, abs(rhs)))
    ok &= worst < 1e-9
    log(f"U1b 4-var identity max rel dev {worst:.2e}")

    # U2: c-table three routes
    worst_stein, worst_gh = 0.0, 0.0
    for (mu, sd) in [(0.0, 1.0), (0.5, 1.0), (-0.4, 0.8), (0.9, 1.3), (-0.8, 1.0), (1.2, 0.6)]:
        ce = ctable_exact(mu, sd)
        cs = ctable_stein(mu, sd)
        cg = ctable_hermgauss(mu, sd)
        for p in (1, 2):
            scale = np.maximum(np.abs(ce[p]), 1e-6)
            worst_stein = max(worst_stein, float(np.max(np.abs(ce[p] - cs[p]) / scale)))
            worst_gh = max(worst_gh, float(np.max(np.abs(ce[p] - cg[p]) / scale)))
    ok &= worst_stein < 1e-10
    log(f"U2 c-table: exact-vs-Stein max rel dev {worst_stein:.2e} (assert<1e-10); "
        f"exact-vs-hermgauss200 {worst_gh:.2e} (kink -> algebraic; report only)")

    # U3: symbolic positive controls through custom A/B arrays
    Ai = np.zeros(QMAX + 1)
    Ai[2] = 2.0                                   # x_i = z^2 (centered He_2)
    Bi = np.zeros(QMAX + 1)
    Bi[0], Bi[2], Bi[4] = 2.0, 8.0, 24.0
    Jj = np.zeros(QMAX + 1)
    Jj[1] = 1.0                                   # x_j = z
    per = kappa211_series(Ai, Bi, Jj, Jj, 0.3, -0.2, 0.0, tmax=TFULL)
    got = sum(per.values())
    want = 8.0 * 0.3 * (-0.2)
    ok &= abs(got - want) < 1e-14
    log(f"U3 z^2 control: series={got:+.12f} exact={want:+.12f} (all orders; dev {abs(got-want):.1e})")
    a = 0.37                                      # x = z + a z^2
    Ai2 = np.zeros(QMAX + 1)
    Ai2[1], Ai2[2] = 1.0, 2.0 * a
    Bi2 = np.zeros(QMAX + 1)
    Bi2[0], Bi2[1], Bi2[2], Bi2[3], Bi2[4] = 1 + 2 * a * a, 4 * a, 2 + 8 * a * a, 12 * a, 24 * a * a
    per2 = kappa211_series(Ai2, Bi2, Jj, Jj, 0.3, -0.2, 0.0, tmax=2)
    want2 = 8.0 * a * a * 0.3 * (-0.2)
    ok &= abs(truncate(per2, 2) - want2) < 1e-14
    log(f"U3b z+az^2 control (order-2 wedge kernel): dev {abs(truncate(per2,2)-want2):.1e}")

    # U4: linear/Gaussian limit (mu/sd = 8): every term ~ 0
    m8, A8, B8 = unit_AB(8.0, 1.0)
    per8 = kappa211_series(A8, B8, A8, A8, 0.3, 0.2, 0.1, tmax=TFULL)
    q8 = quad_kappa211([8.0, 8.0, 8.0], [1.0, 1.0, 1.0], R3(0.3, 0.2, 0.1))
    ok &= abs(sum(per8.values())) < 1e-10 and abs(q8) < 1e-8
    log(f"U4 Gaussian limit: series {sum(per8.values()):.2e}, quad {q8:.2e}")

    # U6: quad_cov vs bivariate Mehler
    worst = 0.0
    for (mu1, sd1, mu2, sd2, r) in [(0.0, 1.0, 0.5, 1.0, 0.35), (-0.4, 0.8, 0.9, 1.3, -0.27)]:
        _, Au, _ = unit_AB(mu1, sd1)
        _, Av, _ = unit_AB(mu2, sd2)
        ser = sum((r ** t) / factorial(t) * Au[t] * Av[t] for t in range(1, QMAX + 1))
        qd = quad_cov(mu1, sd1, mu2, sd2, r)
        worst = max(worst, abs(ser - qd))
    ok &= worst < 1e-9
    log(f"U6 quad_cov vs Mehler series: max abs dev {worst:.2e}")
    return ok


# ---------------------------------------------------------------- configs
M1 = [(0.0, 1.0)] * 3
M2 = [(0.5, 1.0), (-0.4, 0.8), (0.9, 1.3)]
M3 = [(-0.8, 1.0), (0.3, 0.7), (-0.2, 1.1)]
CFG211 = [
    ("t211-null-ij",     M2, (0.20, 0.00, 0.00), 4e8, "k independent -> exact 0"),
    ("t211-null-jk",     M2, (0.00, 0.00, 0.25), 4e8, "i independent -> exact 0"),
    ("t211-wedge-sym",   M1, (0.20, 0.20, 0.00), 8e8, "isolates wedge kernel B2-2A1^2"),
    ("t211-wedge-mixed", M2, (0.20, -0.15, 0.00), 8e8, "wedge, mixed sign"),
    ("t211-path-j",      M2, (0.20, 0.00, 0.20), 8e8, "isolates B1*J2*K1 path"),
    ("t211-path-k",      M2, (0.00, 0.20, 0.20), 8e8, "isolates B1*J1*K2 path"),
    ("t211-all-small",   M1, (0.10, 0.10, 0.10), 1.6e9, "all bonds, rho=0.1"),
    ("t211-all-mid",     M2, (0.20, -0.20, 0.15), 8e8, "all bonds, mixed signs"),
    ("t211-all-03",      M3, (0.30, 0.25, -0.20), 8e8, "larger rho, skewed marginals"),
    ("t211-stress-04",   M2, (0.40, 0.30, 0.20), 8e8, "stress |rho|=0.4"),
    ("t211-tiny-005",    M1, (0.05, 0.05, 0.05), 1.6e9, "scaling anchor"),
    ("t211-negwedge",    M3, (-0.20, -0.20, 0.00), 8e8, "wedge, both negative"),
]
M4 = [(0.0, 1.0), (0.5, 1.0), (-0.4, 0.8), (0.7, 1.2)]
M5 = [(0.0, 1.0)] * 4
CFG1111 = [
    ("t1111-null-matching", M4, (0.25, 0, 0, 0, 0, -0.20), 4e8, "pairs independent -> exact 0"),
    ("t1111-chain",         M4, (0.20, 0, 0, 0.20, 0, 0.20), 1.6e9, "path i-j-k-l isolated"),
    ("t1111-star-l",        M4, (0, 0, 0.20, 0, 0.20, 0.20), 1.6e9, "star at l isolated"),
    ("t1111-all-015",       M5, (0.15,) * 6, 8e8, "all six bonds"),
    ("t1111-chain-half",    M4, (0.10, 0, 0, 0.10, 0, 0.10), 1.6e9, "chain at rho/2 (cubic scaling)"),
    ("t1111-all-mixed",     M4, (0.20, -0.15, 0.10, 0.15, -0.10, 0.20), 8e8, "mixed signs"),
]


def run(quick=False):
    t0 = time.time()
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log(f"TAG public_trivariate_derivation_v1/verify.py | quick={quick} | seed base {SEED_BASE} "
        f"| QMAX={QMAX} TFULL={TFULL} | numpy {np.__version__}")
    assert unit_tests(log), "unit tests FAILED"
    log("unit tests PASS")

    nscale = 64 if quick else 1
    results = {"kappa211": [], "kappa1111": []}

    for idx, (name, marg, rho, N, note) in enumerate(CFG211):
        mus = [m for m, _ in marg]
        sds = [s for _, s in marg]
        R = R3(*rho)
        ev = float(np.linalg.eigvalsh(R).min())
        assert ev > 0.02, (name, ev)
        units = [unit_AB(mu, sd) for mu, sd in zip(mus, sds)]
        units_gh = [unit_AB(mu, sd, "hermgauss") for mu, sd in zip(mus, sds)]
        per = kappa211_series(units[0][1], units[0][2], units[1][1], units[2][1],
                              *rho, tmax=TFULL)
        per_gh = kappa211_series(units_gh[0][1], units_gh[0][2], units_gh[1][1],
                                 units_gh[2][1], *rho, tmax=TFULL)
        tr = {T: truncate(per, T) for T in (2, 3, 4, TFULL)}
        tr_gh = {T: truncate(per_gh, T) for T in (2, 3, 4)}
        q = quad_kappa211(mus, sds, R, n=(48 if quick else 80))
        q2 = quad_kappa211(mus, sds, R, n=(64 if quick else 120))
        kap, se, ndone = mc_kappa211(mus, sds, R, N / nscale, seed=idx,
                                     batch=1_000_000 if quick else 5_000_000)
        row = {"name": name, "note": note, "mus": mus, "sds": sds,
               "rho": {"r_ij": rho[0], "r_ik": rho[1], "r_jk": rho[2]},
               "mc": {"kappa": kap, "se": se, "N": ndone},
               "quad": {"kappa": q2, "refinement_delta": q2 - q},
               "series": {"T2": tr[2], "T3": tr[3], "T4": tr[4], f"T{TFULL}": tr[TFULL]},
               "series_hermgauss": {"T2": tr_gh[2], "T3": tr_gh[3], "T4": tr_gh[4]}}
        results["kappa211"].append(row)
        log(f"[{name}] quad={q2:+.6e} (ref d={q2 - q:+.1e})  MC={kap:+.6e} +/- {se:.1e} "
            f"(N={ndone:.1e}, pull={(kap - q2) / se if se > 0 else 0.0:+.2f})  "
            f"T2={tr[2]:+.6e} T3={tr[3]:+.6e} T4={tr[4]:+.6e} T{TFULL}={tr[TFULL]:+.6e}  "
            f"[{time.time() - t0:.0f}s]")

    for idx, (name, marg, rho6, N, note) in enumerate(CFG1111):
        mus = [m for m, _ in marg]
        sds = [s for _, s in marg]
        R = R4(*rho6)
        ev = float(np.linalg.eigvalsh(R).min())
        assert ev > 0.02, (name, ev)
        al = [unit_AB(mu, sd)[1] for mu, sd in zip(mus, sds)]
        al_gh = [unit_AB(mu, sd, "hermgauss")[1] for mu, sd in zip(mus, sds)]
        per = kappa1111_series(al, R, tmax=10)
        per_gh = kappa1111_series(al_gh, R, tmax=4)
        tr = {T: truncate(per, T) for T in (2, 3, 4, 10)}
        q = quad_kappa1111(mus, sds, R, n=(32 if quick else 48))
        q2 = quad_kappa1111(mus, sds, R, n=(40 if quick else 64))
        kap, se, ndone = mc_kappa1111(mus, sds, R, N / nscale, seed=100 + idx,
                                      batch=1_000_000 if quick else 5_000_000)
        row = {"name": name, "note": note, "mus": mus, "sds": sds,
               "rho": dict(zip(["r_ij", "r_ik", "r_il", "r_jk", "r_jl", "r_kl"], rho6)),
               "mc": {"kappa": kap, "se": se, "N": ndone},
               "quad": {"kappa": q2, "refinement_delta": q2 - q},
               "series": {"T2": tr[2], "T3": tr[3], "T4": tr[4], "T10": tr[10]},
               "series_hermgauss": {"T3": truncate(per_gh, 3), "T4": truncate(per_gh, 4)}}
        results["kappa1111"].append(row)
        log(f"[{name}] quad={q2:+.6e} (ref d={q2 - q:+.1e})  MC={kap:+.6e} +/- {se:.1e} "
            f"(N={ndone:.1e}, pull={(kap - q2) / se if se > 0 else 0.0:+.2f})  "
            f"T2={tr[2]:+.6e} T3={tr[3]:+.6e} T4={tr[4]:+.6e} T10={tr[10]:+.6e}  "
            f"[{time.time() - t0:.0f}s]")

    # ---------------- artifacts
    formula = {
        "tag": "public_trivariate_derivation_v1",
        "date": "2026-08-25",
        "model": "x_u = relu(mu_u + sd_u * z_u); (z) standard multivariate normal, corr rho",
        "primitives": {
            "m": "E[x] = c1[0]",
            "c1[q]": "E[x He_q(z)]  (He probabilists', He_{q+1}=z He_q - q He_{q-1})",
            "c2[q]": "E[x^2 He_q(z)]",
            "A[q]": "c1[q] for q>=1; A[0]=0",
            "B[q]": "c2[q] - 2*m*c1[q] for q>=1; B[0]=c2[0]-m^2 (=Var x)",
            "J[q]/K[q]/alpha[q]": "the A[q] of units j/k/(each unit)",
        },
        "kappa211": {
            "definition": "kappa(x_i,x_i,x_j,x_k) joint cumulant (x_i doubled slot)",
            "term_value": "coefficient * rho_monomial * i_functional * j_functional * k_functional",
            "exact_all_orders": "sum_{a,b,t>=0, a+b>=1} r_ij^a r_ik^b r_jk^t/(a! b! t!) * "
                                "(B[a+b] - 2*A[a]*A[b]*[t==0]) * J[a+t] * K[b+t]",
            "order_le_1": "identically zero (disconnected diagrams cancel)",
            "terms": kappa211_terms((2, 3, 4)),
        },
        "kappa1111": {
            "definition": "kappa(x_i,x_j,x_k,x_l), all slots distinct",
            "term_value": "coefficient * rho_monomial * prod_u vertex_functionals[u]",
            "exact_all_orders": "sum over bond multisets on the 6 edges whose support is a "
                                "connected graph covering all four vertices: "
                                "prod_e r_e^{t_e}/t_e! * prod_u alpha_{s_u}^{(u)}, s_u = incident bond order",
            "order_le_2": "identically zero (THEOREM: two bonds cannot connect four vertices; "
                          "the three perfect matchings cancel against the pair-product subtractions)",
            "terms": kappa1111_terms((3, 4)),
        },
        "verification": results,
        "notes": [
            "quad = deterministic kink-split nested Gauss-Legendre truth (refinement_delta shows its error)",
            "series TN = master formula truncated at total bond order N (T14/T10 ~ all orders)",
            "MC centered with exact means; se from batch spread; pull=(MC-quad)/se",
        ],
    }
    with open(f"{HERE}/FORMULA.json", "w") as f:
        json.dump(formula, f, indent=1)
    write_verification_md(results, quick)
    log(f"artifacts written: FORMULA.json, VERIFICATION.md  [{time.time() - t0:.0f}s total]")
    return results


def write_verification_md(results, quick):
    def relerr(x, truth):
        return abs(x - truth) / abs(truth) if abs(truth) > 1e-8 else float("nan")

    L = ["# VERIFICATION — public_trivariate_derivation_v1",
         f"quick={quick}; truth = kink-split deterministic quadrature (column `quad`, its own error",
         "= `refd`); MC (exact-mean centered, batch-spread SE) concordance shown as pull=(MC-quad)/SE.",
         "`T2/T3/T4` = master-series truncations at total bond order 2/3/4; `Tfull` = order 14 (211) / 10 (1111).",
         "Relative errors vs quad; `nan` where truth is an exact zero (see `MC` and `Tfull` columns instead).",
         ""]
    for pat in ("kappa211", "kappa1111"):
        L.append(f"## {pat}")
        L.append("| config | quad | refd | MC | SE | pull | T2 relerr | T3 relerr | T4 relerr | Tfull relerr | note |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in results[pat]:
            q = r["quad"]["kappa"]
            s = r["series"]
            tf = s.get("T14", s.get("T10"))
            L.append("| {} | {:+.4e} | {:+.0e} | {:+.4e} | {:.0e} | {:+.1f} | {} | {} | {} | {} | {} |".format(
                r["name"], q, r["quad"]["refinement_delta"], r["mc"]["kappa"], r["mc"]["se"],
                (r["mc"]["kappa"] - q) / r["mc"]["se"] if r["mc"]["se"] else 0.0,
                _fmt(relerr(s["T2"], q)), _fmt(relerr(s["T3"], q)), _fmt(relerr(s["T4"], q)),
                _fmt(relerr(tf, q)), r["note"]))
        L.append("")
    with open(f"{HERE}/VERIFICATION.md", "w") as f:
        f.write("\n".join(L) + "\n")


def _fmt(x):
    return "nan" if (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.2f}%"


if __name__ == "__main__":
    run(quick="--quick" in sys.argv)
