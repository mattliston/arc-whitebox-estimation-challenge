#!/usr/bin/env python3
"""TAG r0c1-secondorder-triple-v1: second-order-in-kappa3 response of the
CONNECTED post-ReLU third cumulant at a distinct triple (i,j,k).

Derivation: density = exp(B) phi_C with
  B = sum_c (k3_c/6) d_c^3  +  sum_{a!=b} (K21[a,b]/2) d_a^2 d_b
      + T[ijk] d_i d_j d_k          (all within coordinates {i,j,k})
Second-order response of E[f_i f_j f_k] = E[(B^2/2) applied], i.e. for each
unordered pattern pair (P1,P2): coef(P1,P2) * E[prod_c f_c^(n_c)] with
n_c = derivative count of c in P1+P2. The triple expectation expands in the
Kibble bond series with per-coordinate column families:
  n=0: a[t] = E[f He_t],  n=1: b[t] = E[f' He_t] = He_{t-1}(-al) pdf,
  n>=2: He_{(n-2)+t}(-al) pdf / sigma^(n-1).
CONNECTED filter: keep a (P1,P2,bond) term iff the graph on {i,j,k} with
edges {pairs co-appearing in P1} + {pairs in P2} + {bond pairs with t>0}
is connected (disconnected parts cancel against the assembly subtractions;
the first-order feed cancellation is the special case).
Validation: exact 3D Gauss-Hermite quadrature on constructed non-Gaussian
triples (coordinatewise cubic polynomials of correlated Gaussians, exact
cumulants via small-sample Isserlis-free numerical moments of the
polynomials). Usage: second_order_triple.py  (runs the validation gate)"""
import itertools
import json
from math import erf, factorial

import numpy as np

_erf = np.vectorize(erf)


def col_family(n, m, sd, qmax=14):
    """Column family for f^(n) vs He_t at N(m, sd^2): returns array [t]."""
    al = m / sd
    pdf = np.exp(-0.5 * al * al) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1 + erf(al / np.sqrt(2)))
    ma = -al
    He = [1.0, ma]
    for k in range(2, qmax + n + 2):
        He.append(ma * He[-1] - (k - 1) * He[-2])
    if n == 0:
        c = [sd * pdf + m * cdf, sd * cdf]
        for q in range(2, qmax + 1):
            c.append(sd * He[q - 2] * pdf)
        return np.array(c)
    if n == 1:
        return np.array([cdf] + [He[t - 1] * pdf for t in range(1, qmax + 1)])
    return np.array([He[(n - 2) + t] * pdf / sd ** (n - 1)
                     for t in range(0, qmax + 1)])


# insertion patterns over coordinates (0,1,2): (name, counts, kappa-key)
PATTERNS = [
    ("D0", (3, 0, 0), "k3_0"), ("D1", (0, 3, 0), "k3_1"),
    ("D2", (0, 0, 3), "k3_2"),
    ("P01", (2, 1, 0), "K21_01"), ("P10", (1, 2, 0), "K21_10"),
    ("P02", (2, 0, 1), "K21_02"), ("P20", (1, 0, 2), "K21_20"),
    ("P12", (0, 2, 1), "K21_12"), ("P21", (0, 1, 2), "K21_21"),
    ("TR", (1, 1, 1), "T_012"),
]
PATTERN_COEF = {"D": 1.0 / 6, "P": 0.5, "T": 1.0}


def pattern_edges(counts):
    sup = [c for c in range(3) if counts[c] > 0]
    return {frozenset(e) for e in itertools.combinations(sup, 2)} | (
        {frozenset(sup)} if len(sup) == 3 else set())


def connected(edge_sets):
    parent = list(range(3))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for es in edge_sets:
        for e in es:
            e = list(e)
            for a, b in itertools.combinations(e, 2):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    return len({find(c) for c in range(3)}) == 1


def second_order_terms(bond_cap=3):
    """Enumerate (P1, P2, bonds, coefficient) with the connected filter.
    Returns list of (key1, key2, (t01,t02,t12), (n0,n1,n2), coef)."""
    out = []
    for a in range(len(PATTERNS)):
        for b in range(a, len(PATTERNS)):
            n1, c1_, k1 = PATTERNS[a]
            n2, c2_, k2 = PATTERNS[b]
            base = PATTERN_COEF[n1[0]] * PATTERN_COEF[n2[0]]
            coef0 = base * (0.5 if a == b else 1.0)
            counts = tuple(c1_[i] + c2_[i] for i in range(3))
            e1, e2 = pattern_edges(c1_), pattern_edges(c2_)
            for t01 in range(0, bond_cap + 1):
                for t02 in range(0, bond_cap + 1 - t01):
                    for t12 in range(0, bond_cap + 1 - t01 - t02):
                        eb = set()
                        if t01:
                            eb.add(frozenset((0, 1)))
                        if t02:
                            eb.add(frozenset((0, 2)))
                        if t12:
                            eb.add(frozenset((1, 2)))
                        if not connected([e1, e2, eb]):
                            continue
                        bc = 1.0 / (factorial(t01) * factorial(t02)
                                    * factorial(t12))
                        out.append((k1, k2, (t01, t02, t12), counts,
                                    coef0 * bc))
    return out


def eval_second_order(kap, m, sd, rho, terms):
    """Scalar evaluation at one triple. kap: dict of kappa values keyed as
    in PATTERNS. m, sd: 3-vectors; rho: dict {(0,1):r01,...}."""
    cols = {}

    def cf(n, c):
        if (n, c) not in cols:
            cols[(n, c)] = col_family(n, m[c], sd[c])
        return cols[(n, c)]

    total = 0.0
    for (k1, k2, (t01, t02, t12), (n0, n1, n2), coef) in terms:
        deg = (t01 + t02, t01 + t12, t02 + t12)
        val = (coef * kap[k1] * kap[k2]
               * rho[(0, 1)] ** t01 * rho[(0, 2)] ** t02
               * rho[(1, 2)] ** t12
               * cf(n0, 0)[deg[0]] * cf(n1, 1)[deg[1]]
               * cf(n2, 2)[deg[2]])
        total += val
    return total


def validate():
    """Gate: constructed triples h_c = m_c + sd_c*(g_c + a_c*He2(g_c)) with
    correlated latents; exact joint moments by 3D GH quadrature; compare
    [connected cum3 from quadrature] - [first-order prediction] against the
    second-order formula. Passes if the second-order term explains most of
    the first-order residual."""
    gh_x, gh_w = np.polynomial.hermite_e.hermegauss(90)
    gh_w = gh_w / np.sqrt(2 * np.pi)
    rng = np.random.default_rng(5)
    results = []
    for trial in range(6):
        m = rng.normal(0, 0.6, 3)
        sd0 = np.exp(rng.normal(0, 0.25, 3))
        aa = rng.uniform(-0.12, 0.12, 3)
        while True:
            r = rng.uniform(-0.35, 0.35, 3)
            R = np.array([[1, r[0], r[1]], [r[0], 1, r[2]],
                          [r[1], r[2], 1]])
            if np.linalg.eigvalsh(R).min() > 0.1:
                break
        L = np.linalg.cholesky(R)
        X, Y, Z = np.meshgrid(gh_x, gh_x, gh_x, indexing="ij")
        G = L @ np.stack([X.ravel(), Y.ravel(), Z.ravel()])
        WT = (gh_w[:, None, None] * gh_w[None, :, None]
              * gh_w[None, None, :]).ravel()

        def hvar(c, g):
            he2 = g * g - 1
            return m[c] + sd0[c] * (g + aa[c] * he2)

        h = [hvar(c, G[c]) for c in range(3)]
        f = [np.maximum(hc, 0.0) for hc in h]
        # exact joint cumulants of h (polynomial in latents)
        E = lambda arr: float(np.sum(WT * arr))
        mh = [E(h[c]) for c in range(3)]
        hc = [h[c] - mh[c] for c in range(3)]
        cov = {}
        for c, d in itertools.combinations_with_replacement(range(3), 2):
            cov[(c, d)] = cov[(d, c)] = E(hc[c] * hc[d])
        kap = {}
        for c in range(3):
            kap[f"k3_{c}"] = E(hc[c] ** 3)
        for (c, d) in itertools.permutations(range(3), 2):
            kap[f"K21_{c}{d}"] = E(hc[c] ** 2 * hc[d])
        kap["T_012"] = E(hc[0] * hc[1] * hc[2])
        covd = {(c, d): E(hc[c] * hc[d]) for c in range(3) for d in range(3)}
        for c in range(3):
            kap[f"k4_{c}"] = E(hc[c] ** 4) - 3 * covd[(c, c)] ** 2
        for (c, d) in itertools.permutations(range(3), 2):
            kap[f"K31_{c}{d}"] = (E(hc[c] ** 3 * hc[d])
                                  - 3 * covd[(c, c)] * covd[(c, d)])
        for (c, d) in itertools.combinations(range(3), 2):
            kap[f"K22_{c}{d}"] = (E(hc[c] ** 2 * hc[d] ** 2)
                                  - covd[(c, c)] * covd[(d, d)]
                                  - 2 * covd[(c, d)] ** 2)
        rest = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
        for c in range(3):
            d, e = rest[c]
            kap[f"K4T_{c}"] = (E(hc[c] ** 2 * hc[d] * hc[e])
                               - covd[(c, c)] * covd[(d, e)]
                               - 2 * covd[(c, d)] * covd[(c, e)])
        sd = np.array([np.sqrt(cov[(c, c)]) for c in range(3)])
        mm = np.array(mh)
        rho = {(0, 1): cov[(0, 1)] / (sd[0] * sd[1]),
               (0, 2): cov[(0, 2)] / (sd[0] * sd[2]),
               (1, 2): cov[(1, 2)] / (sd[1] * sd[2])}
        # connected cum3 of f (truth by quadrature)
        Ef = [E(fc) for fc in f]
        Eff = {}
        for c, d in itertools.combinations(range(3), 2):
            Eff[(c, d)] = E(f[c] * f[d])
        Efff = E(f[0] * f[1] * f[2])
        cum_true = (Efff - Ef[0] * Eff[(1, 2)] - Ef[1] * Eff[(0, 2)]
                    - Ef[2] * Eff[(0, 1)] + 2 * Ef[0] * Ef[1] * Ef[2])
        # first-order prediction (connected): Gaussian part + linear terms
        first_terms = []
        for (name, counts, key) in PATTERNS:
            e1 = pattern_edges(counts)
            base = PATTERN_COEF[name[0]]
            for t01 in range(0, 5):
                for t02 in range(0, 5 - t01):
                    for t12 in range(0, 5 - t01 - t02):
                        eb = set()
                        if t01:
                            eb.add(frozenset((0, 1)))
                        if t02:
                            eb.add(frozenset((0, 2)))
                        if t12:
                            eb.add(frozenset((1, 2)))
                        if not connected([e1, eb]):
                            continue
                        bc = 1.0 / (factorial(t01) * factorial(t02)
                                    * factorial(t12))
                        first_terms.append((key, (t01, t02, t12), counts,
                                            base * bc))
        # gaussian connected part: bonds only, all three connected
        gauss = 0.0
        colsA = [col_family(0, mm[c], sd[c]) for c in range(3)]
        for t01 in range(0, 9):
            for t02 in range(0, 9 - t01):
                for t12 in range(0, 9 - t01 - t02):
                    eb = set()
                    if t01:
                        eb.add(frozenset((0, 1)))
                    if t02:
                        eb.add(frozenset((0, 2)))
                    if t12:
                        eb.add(frozenset((1, 2)))
                    if not connected([eb]):
                        continue
                    bc = 1.0 / (factorial(t01) * factorial(t02)
                                * factorial(t12))
                    deg = (t01 + t02, t01 + t12, t02 + t12)
                    gauss += (bc * rho[(0, 1)] ** t01 * rho[(0, 2)] ** t02
                              * rho[(1, 2)] ** t12 * colsA[0][deg[0]]
                              * colsA[1][deg[1]] * colsA[2][deg[2]])
        first = 0.0
        cols = {}

        def cf(n, c):
            if (n, c) not in cols:
                cols[(n, c)] = col_family(n, mm[c], sd[c])
            return cols[(n, c)]

        for (key, (t01, t02, t12), counts, coef) in first_terms:
            deg = (t01 + t02, t01 + t12, t02 + t12)
            first += (coef * kap[key]
                      * rho[(0, 1)] ** t01 * rho[(0, 2)] ** t02
                      * rho[(1, 2)] ** t12
                      * cf(counts[0], 0)[deg[0]] * cf(counts[1], 1)[deg[1]]
                      * cf(counts[2], 2)[deg[2]])
        terms2 = second_order_terms()
        second = eval_second_order(kap, mm, sd, rho, terms2)
        resid1 = cum_true - gauss - first
        resid2 = resid1 - second
        results.append({
            "cum_true": cum_true,
            "resid_first_order": resid1,
            "second_order_term": second,
            "resid_after_second": resid2,
            "improvement": round(abs(resid1 / resid2), 2)
            if resid2 != 0 else None})
        print(json.dumps(results[-1]))
    return results





# kappa4 patterns over (0,1,2): (name, counts, key, coefficient)
K4_PATTERNS = [
    ("Q400", (4, 0, 0), "k4_0", 1.0 / 24), ("Q040", (0, 4, 0), "k4_1", 1.0 / 24),
    ("Q004", (0, 0, 4), "k4_2", 1.0 / 24),
    ("Q310", (3, 1, 0), "K31_01", 1.0 / 6), ("Q130", (1, 3, 0), "K31_10", 1.0 / 6),
    ("Q301", (3, 0, 1), "K31_02", 1.0 / 6), ("Q103", (1, 0, 3), "K31_20", 1.0 / 6),
    ("Q031", (0, 3, 1), "K31_12", 1.0 / 6), ("Q013", (0, 1, 3), "K31_21", 1.0 / 6),
    ("Q220", (2, 2, 0), "K22_01", 1.0 / 4), ("Q202", (2, 0, 2), "K22_02", 1.0 / 4),
    ("Q022", (0, 2, 2), "K22_12", 1.0 / 4),
    ("Q211", (2, 1, 1), "K4T_0", 1.0 / 2), ("Q121", (1, 2, 1), "K4T_1", 1.0 / 2),
    ("Q112", (1, 1, 2), "K4T_2", 1.0 / 2),
]


def k4_first_terms(bond_cap=4):
    out = []
    for (name, counts, key, base) in K4_PATTERNS:
        for t01 in range(0, bond_cap + 1):
            for t02 in range(0, bond_cap + 1 - t01):
                for t12 in range(0, bond_cap + 1 - t01 - t02):
                    bc = 1.0 / (factorial(t01) * factorial(t02)
                                * factorial(t12))
                    out.append((key, (t01, t02, t12), counts, base * bc))
    return out


def k4_pair_response(kap, mm, sd, rho, c, d, tmax=8):
    """Raw E[f_c f_d] first-order response to kappa4 patterns on {c,d}."""
    pats = [(counts, key, base) for (name, counts, key, base) in K4_PATTERNS
            if all(x in (c, d) for x in range(3) if counts[x] > 0)]
    cols = {}

    def cf(n, x):
        if (n, x) not in cols:
            cols[(n, x)] = col_family(n, mm[x], sd[x], qmax=tmax + 8)
        return cols[(n, x)]

    r = rho[(min(c, d), max(c, d))]
    tot = 0.0
    for (counts, key, base) in pats:
        for t in range(0, tmax + 1):
            tot += (base * kap[key] * r ** t / factorial(t)
                    * cf(counts[c], c)[t] * cf(counts[d], d)[t])
    return tot


def k4_uni_response(kap, mm, sd, c):
    return (kap[f"k4_{c}"] / 24.0) * col_family(4, mm[c], sd[c])[0]


def second_order_terms_raw(bond_cap=3):
    """Unfiltered pattern-pair enumeration for RAW E[f_i f_j f_k]."""
    out = []
    for a in range(len(PATTERNS)):
        for b in range(a, len(PATTERNS)):
            n1, c1_, k1 = PATTERNS[a]
            n2, c2_, k2 = PATTERNS[b]
            coef0 = (PATTERN_COEF[n1[0]] * PATTERN_COEF[n2[0]]
                     * (0.5 if a == b else 1.0))
            counts = tuple(c1_[i] + c2_[i] for i in range(3))
            for t01 in range(0, bond_cap + 1):
                for t02 in range(0, bond_cap + 1 - t01):
                    for t12 in range(0, bond_cap + 1 - t01 - t02):
                        bc = 1.0 / (factorial(t01) * factorial(t02)
                                    * factorial(t12))
                        out.append((k1, k2, (t01, t02, t12), counts,
                                    coef0 * bc))
    return out


def raw_first_terms(bond_cap=4):
    out = []
    for (name, counts, key) in PATTERNS:
        base = PATTERN_COEF[name[0]]
        for t01 in range(0, bond_cap + 1):
            for t02 in range(0, bond_cap + 1 - t01):
                for t12 in range(0, bond_cap + 1 - t01 - t02):
                    bc = 1.0 / (factorial(t01) * factorial(t02)
                                * factorial(t12))
                    out.append((key, (t01, t02, t12), counts, base * bc))
    return out


def pair_raw_responses(kap, mm, sd, rho, c, d, order, tmax=8):
    """Raw E[f_c f_d] response at the given order (1 or 2) in the
    insertions SUPPORTED ON {c,d} (patterns within the pair)."""
    pats = []
    for (name, counts, key) in PATTERNS:
        sup = [x for x in range(3) if counts[x] > 0]
        if all(x in (c, d) for x in sup):
            pats.append((counts, key, PATTERN_COEF[name[0]]))
    cols = {}

    def cf(n, x):
        if (n, x) not in cols:
            cols[(n, x)] = col_family(n, mm[x], sd[x], qmax=tmax + 8)
        return cols[(n, x)]

    r = rho[(min(c, d), max(c, d))]
    if order == 1:
        tot = 0.0
        for (counts, key, base) in pats:
            for t in range(0, tmax + 1):
                tot += (base * kap[key] * r ** t / factorial(t)
                        * cf(counts[c], c)[t + 0] * cf(counts[d], d)[t])
        return tot
    tot = 0.0
    for a in range(len(pats)):
        for b in range(a, len(pats)):
            ca, ka, ba = pats[a]
            cb, kb, bb = pats[b]
            coef0 = ba * bb * (0.5 if a == b else 1.0)
            nc, nd = ca[c] + cb[c], ca[d] + cb[d]
            for t in range(0, tmax + 1):
                tot += (coef0 * kap[ka] * kap[kb] * r ** t / factorial(t)
                        * cf(nc, c)[t] * cf(nd, d)[t])
    return tot


def uni_responses(kap, mm, sd, c):
    k3 = kap[f"k3_{c}"]
    d3 = col_family(3, mm[c], sd[c])[0]     # E[d^3 f] = He1(-al) pdf / sd^2
    d6 = col_family(6, mm[c], sd[c])[0]     # E[d^6 f]
    return (k3 / 6.0) * d3, (k3 * k3 / 72.0) * d6


def validate_assembled():
    """Gate v2: raw responses + numerical product-rule assembly."""
    gh_x, gh_w = np.polynomial.hermite_e.hermegauss(90)
    gh_w = gh_w / np.sqrt(2 * np.pi)
    rng = np.random.default_rng(5)
    for trial in range(6):
        m = rng.normal(0, 0.6, 3)
        sd0 = np.exp(rng.normal(0, 0.25, 3))
        aa = rng.uniform(-0.22, 0.22, 3)
        while True:
            r = rng.uniform(-0.35, 0.35, 3)
            R = np.array([[1, r[0], r[1]], [r[0], 1, r[2]],
                          [r[1], r[2], 1]])
            if np.linalg.eigvalsh(R).min() > 0.1:
                break
        L = np.linalg.cholesky(R)
        X, Y, Z = np.meshgrid(gh_x, gh_x, gh_x, indexing="ij")
        G = L @ np.stack([X.ravel(), Y.ravel(), Z.ravel()])
        WT = (gh_w[:, None, None] * gh_w[None, :, None]
              * gh_w[None, None, :]).ravel()
        h = [m[c] + sd0[c] * (G[c] + aa[c] * (G[c] * G[c] - 1))
             for c in range(3)]
        f = [np.maximum(hc, 0.0) for hc in h]
        E = lambda arr: float(np.sum(WT * arr))
        mh = [E(h[c]) for c in range(3)]
        hc = [h[c] - mh[c] for c in range(3)]
        cov = {}
        for c, d in itertools.combinations_with_replacement(range(3), 2):
            cov[(c, d)] = cov[(d, c)] = E(hc[c] * hc[d])
        kap = {}
        for c in range(3):
            kap[f"k3_{c}"] = E(hc[c] ** 3)
        for (c, d) in itertools.permutations(range(3), 2):
            kap[f"K21_{c}{d}"] = E(hc[c] ** 2 * hc[d])
        kap["T_012"] = E(hc[0] * hc[1] * hc[2])
        covd = {(c, d): E(hc[c] * hc[d]) for c in range(3) for d in range(3)}
        for c in range(3):
            kap[f"k4_{c}"] = E(hc[c] ** 4) - 3 * covd[(c, c)] ** 2
        for (c, d) in itertools.permutations(range(3), 2):
            kap[f"K31_{c}{d}"] = (E(hc[c] ** 3 * hc[d])
                                  - 3 * covd[(c, c)] * covd[(c, d)])
        for (c, d) in itertools.combinations(range(3), 2):
            kap[f"K22_{c}{d}"] = (E(hc[c] ** 2 * hc[d] ** 2)
                                  - covd[(c, c)] * covd[(d, d)]
                                  - 2 * covd[(c, d)] ** 2)
        rest = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
        for c in range(3):
            d, e = rest[c]
            kap[f"K4T_{c}"] = (E(hc[c] ** 2 * hc[d] * hc[e])
                               - covd[(c, c)] * covd[(d, e)]
                               - 2 * covd[(c, d)] * covd[(c, e)])
        sd = np.array([np.sqrt(cov[(c, c)]) for c in range(3)])
        mm = np.array(mh)
        rho = {(0, 1): cov[(0, 1)] / (sd[0] * sd[1]),
               (0, 2): cov[(0, 2)] / (sd[0] * sd[2]),
               (1, 2): cov[(1, 2)] / (sd[1] * sd[2])}
        Ef = [E(fc) for fc in f]
        Eff = {(c, d): E(f[c] * f[d])
               for c, d in itertools.combinations(range(3), 2)}
        Efff = E(f[0] * f[1] * f[2])
        cum_true = (Efff - Ef[0] * Eff[(1, 2)] - Ef[1] * Eff[(0, 2)]
                    - Ef[2] * Eff[(0, 1)] + 2 * Ef[0] * Ef[1] * Ef[2])
        cols = {}

        def cf(n, c):
            if (n, c) not in cols:
                cols[(n, c)] = col_family(n, mm[c], sd[c], qmax=18)
            return cols[(n, c)]

        def bond3(counts, tmax=8):
            tot = 0.0
            for t01 in range(0, tmax + 1):
                for t02 in range(0, tmax + 1 - t01):
                    for t12 in range(0, tmax + 1 - t01 - t02):
                        bc = 1.0 / (factorial(t01) * factorial(t02)
                                    * factorial(t12))
                        deg = (t01 + t02, t01 + t12, t02 + t12)
                        tot += (bc * rho[(0, 1)] ** t01
                                * rho[(0, 2)] ** t02 * rho[(1, 2)] ** t12
                                * cf(counts[0], 0)[deg[0]]
                                * cf(counts[1], 1)[deg[1]]
                                * cf(counts[2], 2)[deg[2]])
            return tot

        E3_0 = bond3((0, 0, 0))
        E3_1 = 0.0
        for (key, (t01, t02, t12), counts, coef) in raw_first_terms():
            deg = (t01 + t02, t01 + t12, t02 + t12)
            E3_1 += (coef * kap[key] * rho[(0, 1)] ** t01
                     * rho[(0, 2)] ** t02 * rho[(1, 2)] ** t12
                     * cf(counts[0], 0)[deg[0]] * cf(counts[1], 1)[deg[1]]
                     * cf(counts[2], 2)[deg[2]])
        for (key, (t01, t02, t12), counts, coef) in k4_first_terms():
            deg = (t01 + t02, t01 + t12, t02 + t12)
            E3_1 += (coef * kap[key] * rho[(0, 1)] ** t01
                     * rho[(0, 2)] ** t02 * rho[(1, 2)] ** t12
                     * cf(counts[0], 0)[deg[0]] * cf(counts[1], 1)[deg[1]]
                     * cf(counts[2], 2)[deg[2]])
        E3_2 = 0.0
        for (k1, k2, (t01, t02, t12), counts, coef) in second_order_terms_raw():
            deg = (t01 + t02, t01 + t12, t02 + t12)
            E3_2 += (coef * kap[k1] * kap[k2] * rho[(0, 1)] ** t01
                     * rho[(0, 2)] ** t02 * rho[(1, 2)] ** t12
                     * cf(counts[0], 0)[deg[0]] * cf(counts[1], 1)[deg[1]]
                     * cf(counts[2], 2)[deg[2]])
        # pair and univariate responses (raw)
        def pair0(c, d, tmax=10):
            r_ = rho[(min(c, d), max(c, d))]
            return sum(r_ ** t / factorial(t) * cf(0, c)[t] * cf(0, d)[t]
                       for t in range(tmax + 1))

        E1_0 = [cf(0, c)[0] for c in range(3)]
        E1_1, E1_2 = zip(*[uni_responses(kap, mm, sd, c) for c in range(3)])
        E1_1 = tuple(E1_1[c] + k4_uni_response(kap, mm, sd, c)
                     for c in range(3))
        pairs = [(1, 2), (0, 2), (0, 1)]
        E2_0 = {pr: pair0(*pr) for pr in pairs}
        E2_1 = {pr: pair_raw_responses(kap, mm, sd, rho, *pr, order=1)
                + k4_pair_response(kap, mm, sd, rho, *pr)
                for pr in pairs}
        E2_2 = {pr: pair_raw_responses(kap, mm, sd, rho, *pr, order=2)
                for pr in pairs}
        # assemble second-order of connected cum3 by product rule
        # cum3 = E3 - sum_c E1_c * E2_{other} + 2 E1E1E1
        d2 = E3_2
        for c, pr in zip(range(3), pairs):
            d2 -= (E1_2[c] * E2_0[pr] + E1_1[c] * E2_1[pr]
                   + E1_0[c] * E2_2[pr])
        d2 += 2 * (E1_2[0] * E1_0[1] * E1_0[2]
                   + E1_0[0] * E1_2[1] * E1_0[2]
                   + E1_0[0] * E1_0[1] * E1_2[2]
                   + E1_1[0] * E1_1[1] * E1_0[2]
                   + E1_1[0] * E1_0[1] * E1_1[2]
                   + E1_0[0] * E1_1[1] * E1_1[2])
        # first-order connected (same assembly at order 1)
        d1 = E3_1
        for c, pr in zip(range(3), pairs):
            d1 -= (E1_1[c] * E2_0[pr] + E1_0[c] * E2_1[pr])
        d1 += 2 * (E1_1[0] * E1_0[1] * E1_0[2]
                   + E1_0[0] * E1_1[1] * E1_0[2]
                   + E1_0[0] * E1_0[1] * E1_1[2])
        d0 = E3_0
        for c, pr in zip(range(3), pairs):
            d0 -= E1_0[c] * E2_0[pr]
        d0 += 2 * E1_0[0] * E1_0[1] * E1_0[2]
        resid1 = cum_true - d0 - d1
        resid2 = resid1 - d2
        print(json.dumps({"cum_true": cum_true,
                          "resid_first": resid1,
                          "second_pred": d2,
                          "resid_after": resid2,
                          "ratio": round(abs(resid1) / max(abs(resid2), 1e-300), 1)}))


if __name__ == "__main__":
    validate_assembled()

