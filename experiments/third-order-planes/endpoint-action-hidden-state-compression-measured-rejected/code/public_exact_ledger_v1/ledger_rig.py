#!/usr/bin/env python3
"""TAG public_exact_ledger_v1/ledger_rig.py: small-n closed-loop ledger machine.

The n=1024 incumbent's structure, reimplemented exactly at small n (plain numpy,
float64) so every map is checkable:
  state  = m, Sigma, k3 diag + K21 plane, k4 diag + K31 + K22 planes,
           k5 diag + K41 + K32 planes (level-gated)
  push   = exact multilinear diag+plane chains (brute-force-verified: chain_check)
  reset  = univariate c-table columns (level-gated tilt order) + pair moments by
           Edgeworth-product-over-Mehler with level-gated channel sets
           (base; L3; L4 + L3^2/2; Lambda5 + L3*L4 + L3^3/6); post cumulants by
           connected (Leonov-Shiryaev) assembly on emitted raw moments.
Levels: A kappa<=2, B kappa<=3, C kappa<=4 (= incumbent class),
        D kappa<=5 exact-carry + complete pair tilt-3, E = D + kappa6-free
        univariate tilt-4 columns.
Plane storage: K21[i,j] = cum(x_i,x_i,x_j), K31[i,j] = cum(x_i,x_i,x_i,x_j),
K22[i,j] = cum(x_i,x_i,x_j,x_j) (symmetric), K41[i,j] = cum(x_i^4,x_j),
K32[i,j] = cum(x_i^3,x_j^2); zero diagonals (diagonal mass in k{r}d).
Push convention: h_a = sum_i x_i W[i,a].
"""
from math import comb, factorial

import numpy as np

INV_SQRT_2PI = 0.3989422804014327


def norm_cdf(x):
    from math import erf
    return 0.5 * (1.0 + np.vectorize(erf)(np.asarray(x, dtype=np.float64)
                                          / np.sqrt(2.0)))


def hermite_rows(x, qmax):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty((qmax + 1,) + x.shape, dtype=np.float64)
    out[0] = 1.0
    if qmax >= 1:
        out[1] = x
    for q in range(2, qmax + 1):
        out[q] = x * out[q - 1] - (q - 1) * out[q - 2]
    return out


# ------------------------------------------------- connected cumulant assembly
def _partitions(items):
    if len(items) == 1:
        yield [items]
        return
    first, rest = items[0], items[1:]
    for part in _partitions(rest):
        for i in range(len(part)):
            yield part[:i] + [[first] + part[i]] + part[i + 1:]
        yield [[first]] + part


def cumulant_terms(p, q):
    """kappa(x:p, y:q) = sum over set partitions with no singleton blocks of
    (-1)^{k-1} (k-1)! * prod central-moment(p_b, q_b). Returns
    [(coef, ((p1,q1),...)), ...] with like terms merged."""
    legs = [0] * p + [1] * q
    seen = {}
    for part in _partitions(list(range(p + q))):
        if any(len(b) == 1 for b in part):
            continue
        key = tuple(sorted((sum(1 for l in b if legs[l] == 0),
                            sum(1 for l in b if legs[l] == 1)) for b in part))
        coef = (-1) ** (len(part) - 1) * factorial(len(part) - 1)
        seen[key] = seen.get(key, 0) + coef
    return [(c, k) for k, c in sorted(seen.items()) if c != 0]


CUM_TERMS = {(p, q): cumulant_terms(p, q)
             for p in range(2, 7) for q in range(0, 4) if 2 <= p + q <= 6}


def connected(p, q, central):
    out = 0.0
    for coef, blocks in CUM_TERMS[(p, q)]:
        term = float(coef)
        for pk_qk in blocks:
            term = term * central[pk_qk]
        out = out + term
    return out


def central_pair_from_raw(raw, mx, my):
    """raw[(p,q)] = E[X^p Y^q] -> central moments about (mx, my)."""
    out = {}
    keys = list(raw.keys())
    raw = dict(raw)
    raw[(0, 0)] = 1.0 + (0.0 * mx)
    for p in range(5):
        if (p, 0) not in raw and p > 0:
            pass
    for (p, q) in keys:
        acc = 0.0
        ok = True
        for a in range(p + 1):
            for b in range(q + 1):
                m = raw.get((a, b))
                if m is None:
                    ok = False
                    break
                acc = acc + (comb(p, a) * comb(q, b) * ((-mx) ** (p - a))
                             * ((-my) ** (q - b)) * m)
            if not ok:
                break
        if ok:
            out[(p, q)] = acc
    return out


# ----------------------------------------------------------- univariate reset
def gaussian_pos_raw(mu, var, pmax=5):
    """E[X_+^p], X = relu(N(mu, var)), p=0..pmax, closed forms."""
    sd = np.sqrt(var)
    a = mu / sd
    pdf = np.exp(-0.5 * a * a) * INV_SQRT_2PI
    cdf = norm_cdf(a)
    m2, v = mu * mu, var
    raws = [cdf,
            sd * pdf + mu * cdf,
            (m2 + v) * cdf + mu * sd * pdf,
            mu * (m2 + 3 * v) * cdf + sd * (m2 + 2 * v) * pdf,
            (m2 * m2 + 6 * m2 * v + 3 * v * v) * cdf
            + mu * sd * (m2 + 5 * v) * pdf,
            mu * (m2 * m2 + 10 * m2 * v + 15 * v * v) * cdf
            + sd * (m2 * m2 + 9 * m2 * v + 8 * v * v) * pdf]
    return raws[:pmax + 1], sd, a, pdf, cdf


def c_table(mu, var, pmax=5, qmax=52):
    """c[p][q] = E[X_+^p He_q(Z)], Z the standardized pre-activation:
    q <= p: p!/(p-q)! sd^q raw_{p-q};  q > p: p! sd^p He_{q-p-1}(-a) pdf."""
    raws, sd, a, pdf, _ = gaussian_pos_raw(mu, var, pmax)
    he = hermite_rows(-a, qmax)
    table = []
    for p in range(pmax + 1):
        rows = np.empty((qmax + 1,) + np.shape(mu), dtype=np.float64)
        for q in range(qmax + 1):
            if q <= p:
                rows[q] = (factorial(p) / factorial(p - q)) * sd ** q * raws[p - q]
            else:
                rows[q] = factorial(p) * sd ** p * he[q - p - 1] * pdf
        table.append(rows)
    return table, sd, a, pdf


UNI_COLS = {
    "A": [], "B": ["g3"], "C": ["g3", "g4", "g3g3"],
    "D": ["g3", "g4", "g3g3", "g5", "g3g4", "g3g3g3"],
    "E": ["g3", "g4", "g3g3", "g5", "g3g4", "g3g3g3",
          "g3g5", "g4g4", "g3g3g4", "g3g3g3g3"],
}
UNI_Q = {"g3": 3, "g4": 4, "g3g3": 6, "g5": 5, "g3g4": 7, "g3g3g3": 9,
         "g3g5": 8, "g4g4": 8, "g3g3g4": 10, "g3g3g3g3": 12}
UNI_W = {
    "g3": lambda g: g["g3"] / 6.0, "g4": lambda g: g["g4"] / 24.0,
    "g3g3": lambda g: g["g3"] ** 2 / 72.0, "g5": lambda g: g["g5"] / 120.0,
    "g3g4": lambda g: g["g3"] * g["g4"] / 144.0,
    "g3g3g3": lambda g: g["g3"] ** 3 / 1296.0,
    "g3g5": lambda g: g["g3"] * g["g5"] / 720.0,
    "g4g4": lambda g: g["g4"] ** 2 / 1152.0,
    "g3g3g4": lambda g: g["g3"] ** 2 * g["g4"] / 1728.0,
    "g3g3g3g3": lambda g: g["g3"] ** 4 / 31104.0,
}


def univariate_reset(mu, var, g, level, pmax=5):
    """Post raw moments E[x^p], p=1..pmax, at the level's column set."""
    table, sd, a, pdf = c_table(mu, var, pmax)
    out = []
    for p in range(1, pmax + 1):
        m = table[p][0].copy()
        for name in UNI_COLS[level]:
            m = m + UNI_W[name](g) * table[p][UNI_Q[name]]
        out.append(m)
    return out


# ----------------------------------------------------------------- pair reset
BLOCK3 = {(3, 0): 1 / 6, (2, 1): 1 / 2, (1, 2): 1 / 2, (0, 3): 1 / 6}
BLOCK4 = {(4, 0): 1 / 24, (3, 1): 1 / 6, (2, 2): 1 / 4, (1, 3): 1 / 6,
          (0, 4): 1 / 24}
BLOCK5 = {(5, 0): 1 / 120, (4, 1): 1 / 24, (3, 2): 1 / 12, (2, 3): 1 / 12,
          (1, 4): 1 / 24, (0, 5): 1 / 120}


def pair_channels(level, st):
    """{(a,b): weight matrix} from the operator-exponential expansion at the
    level's tilt order. st fields are STANDARDIZED: g3, g4, g5 (n,) and
    G21, G31, G22, G41, G32 (n,n)."""
    n = st["g3"].shape[0]
    one = np.ones((n, n))
    k = {(3, 0): st["g3"][:, None] * one, (0, 3): st["g3"][None, :] * one,
         (2, 1): st["G21"], (1, 2): st["G21"].T}
    if level in ("C", "D", "E"):
        k.update({(4, 0): st["g4"][:, None] * one,
                  (0, 4): st["g4"][None, :] * one,
                  (3, 1): st["G31"], (1, 3): st["G31"].T, (2, 2): st["G22"]})
    if level in ("D", "E"):
        k.update({(5, 0): st["g5"][:, None] * one,
                  (0, 5): st["g5"][None, :] * one,
                  (4, 1): st["G41"], (1, 4): st["G41"].T, (3, 2): st["G32"],
                  (2, 3): st["G32"].T})

    def block(spec):
        return {ab: spec[ab] * k[ab] for ab in spec}

    def mul(t1, t2):
        out = {}
        for ab1, w1 in t1.items():
            for ab2, w2 in t2.items():
                key = (ab1[0] + ab2[0], ab1[1] + ab2[1])
                out[key] = out.get(key, 0.0) + w1 * w2
        return out

    def add(dst, table, scale=1.0):
        for ab, w in table.items():
            dst[ab] = dst.get(ab, 0.0) + scale * w

    chans = {}
    if level == "A":
        return chans
    L3 = block(BLOCK3)
    add(chans, L3)
    if level in ("C", "D", "E"):
        L4 = block(BLOCK4)
        add(chans, L4)
        add(chans, mul(L3, L3), 0.5)
    if level in ("D", "E"):
        L5 = block(BLOCK5)
        add(chans, L5)
        add(chans, mul(L3, L4), 1.0)
        add(chans, mul(mul(L3, L3), L3), 1.0 / 6.0)
    return chans


def pair_reset(mu, var, Sigma, st, level,
               moments=((1, 1), (2, 1), (3, 1), (2, 2), (4, 1), (3, 2)),
               tmax=40, pmax=5):
    """Post pair raw moments E[x_i^p x_j^q] via the Mehler-product kernels
    (diagonals meaningless; fix from the univariate row)."""
    qmax = 12 + tmax
    table, sd, a, pdf = c_table(mu, var, pmax, qmax=qmax)
    inv_sd = 1.0 / sd
    rho = np.clip(Sigma * inv_sd[:, None] * inv_sd[None, :],
                  -0.999999, 0.999999)
    np.fill_diagonal(rho, 0.0)
    n = mu.shape[0]
    rho_pow = np.empty((tmax + 1, n, n))
    rho_pow[0] = 1.0
    for t in range(1, tmax + 1):
        rho_pow[t] = rho_pow[t - 1] * rho
    inv_fact = np.array([1.0 / factorial(t) for t in range(tmax + 1)])
    kern = rho_pow * inv_fact[:, None, None]
    chans = {(0, 0): np.ones((n, n))}
    chans.update(pair_channels(level, st))
    out = {}
    for (p, q) in moments:
        acc = np.zeros((n, n))
        for (aa, bb), wmat in chans.items():
            fi = table[p][aa:aa + tmax + 1] * (inv_sd ** aa)[None, :]
            fj = table[q][bb:bb + tmax + 1] * (inv_sd ** bb)[None, :]
            acc = acc + wmat * np.einsum("ti,tj,tij->ij", fi, fj, kern,
                                         optimize=True)
        out[(p, q)] = acc
    return out


# ------------------------------------------------------------------ push maps
def push_state(W, s, level):
    """Exact multilinear diag+plane chains (see chain_check for the
    brute-force verification of every coefficient)."""
    out = {"m": s["m"] @ W, "Sigma": W.T @ s["Sigma"] @ W}
    if level == "A":
        return out
    W2, W3 = W * W, W * W * W
    W4, W5 = W2 * W2, W2 * W3
    E = np.einsum
    k3d, K21 = s["k3d"], s["K21"]
    out["k3d"] = W3.T @ k3d + 3.0 * E("ia,ja,ij->a", W2, W, K21, optimize=True)
    K21n = (W2 * k3d[:, None]).T @ W \
        + E("ia,jb,ij->ab", W2, W, K21, optimize=True) \
        + 2.0 * E("ia,ja,ib,ij->ab", W, W, W, K21, optimize=True)
    np.fill_diagonal(K21n, 0.0)
    out["K21"] = K21n
    if level == "B":
        return out
    k4d, K31, K22 = s["k4d"], s["K31"], s["K22"]
    out["k4d"] = W4.T @ k4d \
        + 4.0 * E("ia,ja,ij->a", W3, W, K31, optimize=True) \
        + 3.0 * E("ia,ja,ij->a", W2, W2, K22, optimize=True)
    K31n = (W3 * k4d[:, None]).T @ W \
        + E("ia,jb,ij->ab", W3, W, K31, optimize=True) \
        + 3.0 * E("ia,ja,ib,ij->ab", W2, W, W, K31, optimize=True) \
        + 3.0 * E("ia,ja,jb,ij->ab", W2, W, W, K22, optimize=True)
    K22n = (W2 * k4d[:, None]).T @ W2 \
        + 2.0 * E("ia,ja,ib,ij->ab", W, W, W2, K31, optimize=True) \
        + 2.0 * E("ia,ib,jb,ij->ab", W2, W, W, K31, optimize=True) \
        + E("ia,jb,ij->ab", W2, W2, K22, optimize=True) \
        + 2.0 * E("ia,ib,ja,jb,ij->ab", W, W, W, W, K22, optimize=True)
    np.fill_diagonal(K31n, 0.0)
    np.fill_diagonal(K22n, 0.0)
    out["K31"], out["K22"] = K31n, K22n
    if level == "C":
        return out
    k5d, K41, K32 = s["k5d"], s["K41"], s["K32"]
    out["k5d"] = W5.T @ k5d \
        + 5.0 * E("ia,ja,ij->a", W4, W, K41, optimize=True) \
        + 10.0 * E("ia,ja,ij->a", W3, W2, K32, optimize=True)
    K41n = (W4 * k5d[:, None]).T @ W \
        + E("ia,jb,ij->ab", W4, W, K41, optimize=True) \
        + 4.0 * E("ia,ja,ib,ij->ab", W3, W, W, K41, optimize=True) \
        + 4.0 * E("ia,ja,jb,ij->ab", W3, W, W, K32, optimize=True) \
        + 6.0 * E("ia,ja,ib,ij->ab", W2, W2, W, K32, optimize=True)
    K32n = (W3 * k5d[:, None]).T @ W2 \
        + 2.0 * E("ia,ib,jb,ij->ab", W3, W, W, K41, optimize=True) \
        + 3.0 * E("ia,ja,ib,ij->ab", W2, W, W2, K41, optimize=True) \
        + E("ia,jb,ij->ab", W3, W2, K32, optimize=True) \
        + 6.0 * E("ia,ja,ib,jb,ij->ab", W2, W, W, W, K32, optimize=True) \
        + 3.0 * E("ia,ja,ib,ij->ab", W, W2, W2, K32, optimize=True)
    np.fill_diagonal(K41n, 0.0)
    np.fill_diagonal(K32n, 0.0)
    out["K41"], out["K32"] = K41n, K32n
    return out


# ------------------------------------------------------- brute-force checkers
def _full_tensor(n, order, diag, planes):
    """Symmetric tensor from diag + plane content. planes: {(r1, 1): P} means
    P[i,j] on multiset {i^r1, j}; {(r1, r2): P} with r2=2 on {i^r1, j^2}."""
    shape = (n,) * order
    T = np.zeros(shape)
    idx = np.arange(n)
    T[tuple([idx] * order)] = diag
    from itertools import permutations
    for (r1, r2), P in planes.items():
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                base = (i,) * r1 + (j,) * r2
                for perm in set(permutations(base)):
                    T[perm] = P[i, j] if r2 == 1 or (r1, r2) != (2, 2) else P[i, j]
    return T


def _push_full(T, W):
    order = T.ndim
    for _ in range(order):
        T = np.tensordot(T, W, axes=([0], [0]))
    return T


def chain_check(seed=0, n=6):
    """Brute-force verification of every push-chain coefficient."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n, n))
    rep = {}

    def planes_ok(order, spec_builder, chain_fields):
        diag = rng.normal(size=n)
        planes = {}
        for key in spec_builder:
            P = rng.normal(size=(n, n))
            np.fill_diagonal(P, 0.0)
            if key == (2, 2):
                P = 0.5 * (P + P.T)
            planes[key] = P
        T = _full_tensor(n, order, diag, planes)
        Tp = _push_full(T, W)
        s = {"m": np.zeros(n), "Sigma": np.eye(n)}
        if order == 3:
            s.update({"k3d": diag, "K21": planes[(2, 1)]})
            lev = "B"
        elif order == 4:
            s.update({"k3d": np.zeros(n), "K21": np.zeros((n, n)),
                      "k4d": diag, "K31": planes[(3, 1)], "K22": planes[(2, 2)]})
            lev = "C"
        else:
            s.update({"k3d": np.zeros(n), "K21": np.zeros((n, n)),
                      "k4d": np.zeros(n), "K31": np.zeros((n, n)),
                      "K22": np.zeros((n, n)),
                      "k5d": diag, "K41": planes[(4, 1)], "K32": planes[(3, 2)]})
            lev = "D"
        p = push_state(W, s, lev)
        errs = {}
        idx = np.arange(n)
        diag_slice = Tp[tuple([idx] * order)]
        errs["diag"] = float(np.max(np.abs(
            p[chain_fields[0]] - diag_slice)) / np.max(np.abs(diag_slice)))
        for fld, sl in chain_fields[1]:
            ref = np.array([[Tp[sl(a, b)] for b in range(n)] for a in range(n)])
            np.fill_diagonal(ref, 0.0)
            errs[fld] = float(np.max(np.abs(p[fld] - ref))
                              / max(np.max(np.abs(ref)), 1e-300))
        return errs

    rep["order3"] = planes_ok(3, [(2, 1)], ("k3d", [("K21", lambda a, b:
                                                     (a, a, b))]))
    rep["order4"] = planes_ok(4, [(3, 1), (2, 2)],
                              ("k4d", [("K31", lambda a, b: (a, a, a, b)),
                                       ("K22", lambda a, b: (a, a, b, b))]))
    rep["order5"] = planes_ok(5, [(4, 1), (3, 2)],
                              ("k5d", [("K41", lambda a, b: (a, a, a, a, b)),
                                       ("K32", lambda a, b: (a, a, a, b, b))]))
    return rep


# -------------------------------------------------------------------- MC truth
PAIR_PQ = ((1, 1), (2, 1), (3, 1), (2, 2), (4, 1), (3, 2))


def mc_truth(Ws, n_samples, seed, batch=200_000):
    """Antithetic MC through the net. Returns per-layer dicts with:
    h_mom[k] (k=1..6, per unit), h_pair[(p,q)] (n,n), x_mom[k] (k=1..5),
    x_pair[(p,q)] (n,n)."""
    n = Ws[0].shape[0]
    depth = len(Ws)
    acc = [{"h": np.zeros((6, n)), "x": np.zeros((5, n)),
            "hp": {pq: np.zeros((n, n)) for pq in PAIR_PQ},
            "xp": {pq: np.zeros((n, n)) for pq in PAIR_PQ}}
           for _ in range(depth)]
    rng = np.random.default_rng(seed)
    Ws32 = [np.asarray(W, dtype=np.float32) for W in Ws]
    done = 0
    while done < n_samples:
        b = min(batch, n_samples - done)
        z = rng.standard_normal(size=(b, n)).astype(np.float32)
        for sgn in (np.float32(1.0), np.float32(-1.0)):
            x = sgn * z
            for li in range(depth):
                h = x @ Ws32[li]
                powsh = [h]
                for _ in range(5):
                    powsh.append(powsh[-1] * h)
                a = acc[li]
                for k in range(6):
                    a["h"][k] += powsh[k].sum(axis=0, dtype=np.float64)
                for (p, q) in PAIR_PQ:
                    a["hp"][(p, q)] += (powsh[p - 1].T @ powsh[q - 1]
                                        ).astype(np.float64)
                x = np.maximum(h, np.float32(0.0))
                powsx = [x]
                for _ in range(4):
                    powsx.append(powsx[-1] * x)
                for k in range(5):
                    a["x"][k] += powsx[k].sum(axis=0, dtype=np.float64)
                for (p, q) in PAIR_PQ:
                    a["xp"][(p, q)] += (powsx[p - 1].T @ powsx[q - 1]
                                        ).astype(np.float64)
        done += b
    total = 2.0 * done
    out = []
    for li in range(depth):
        a = acc[li]
        out.append({
            "h_mom": a["h"] / total, "x_mom": a["x"] / total,
            "h_pair": {pq: a["hp"][pq] / total for pq in PAIR_PQ},
            "x_pair": {pq: a["xp"][pq] / total for pq in PAIR_PQ},
        })
    return out


def univariate_cumulants_from_moments(mom):
    """mom[k-1] = E[X^k], k = 1..K. Returns [mean, var, k3, ..., kK]."""
    K = len(mom)
    m = [1.0 * np.ones_like(mom[0])] + list(mom)
    kap = [None] * (K + 1)
    for r in range(1, K + 1):
        acc = m[r]
        for j in range(1, r):
            acc = acc - comb(r - 1, j - 1) * kap[j] * m[r - j]
        kap[r] = acc
    return kap[1:]


def state_from_mc(layer, order5=True):
    """Ledger-state fields (RAW cumulants + Sigma) measured from MC tables for
    the PRE-activation h (teacher forcing) with diag+plane content."""
    h = layer["h_mom"]
    kaps = univariate_cumulants_from_moments([h[k] for k in range(6)])
    mean, var = kaps[0], kaps[1]
    st = {"m_pre": mean, "var_pre": var, "k3d": kaps[2], "k4d": kaps[3],
          "k5d": kaps[4], "k6d": kaps[5]}
    raw = dict(layer["h_pair"])
    n = mean.shape[0]
    raw[(1, 2)] = raw[(2, 1)].T.copy()
    raw[(1, 0)] = np.tile(mean[:, None], (1, n))
    raw[(2, 0)] = np.tile(h[1][:, None], (1, n))
    raw[(3, 0)] = np.tile(h[2][:, None], (1, n))
    raw[(4, 0)] = np.tile(h[3][:, None], (1, n))
    raw[(0, 1)] = np.tile(mean[None, :], (n, 1))
    raw[(0, 2)] = np.tile(h[1][None, :], (n, 1))
    mx = raw[(1, 0)]
    my = raw[(0, 1)]
    central = central_pair_from_raw(raw, mx, my)
    st["Sigma"] = central[(1, 1)].copy()
    np.fill_diagonal(st["Sigma"], var)
    st["K21"] = connected(2, 1, central)
    st["K31"] = connected(3, 1, central)
    st["K22"] = connected(2, 2, central)
    if order5:
        st["K41"] = connected(4, 1, central)
        st["K32"] = connected(3, 2, central)
    for key in ("K21", "K31", "K22", "K41", "K32"):
        if key in st:
            np.fill_diagonal(st[key], 0.0)
    return st


def post_truth_from_mc(layer):
    x = layer["x_mom"]
    kaps = univariate_cumulants_from_moments([x[k] for k in range(5)])
    cov = layer["x_pair"][(1, 1)] - np.outer(x[0], x[0])
    np.fill_diagonal(cov, kaps[1])
    return {"m": x[0], "var": kaps[1], "Sigma": cov,
            "k3d": kaps[2], "k4d": kaps[3]}


# ------------------------------------------------------------- ledger machine
def reset_state(pre, level, n):
    """One relu reset: consume pre-state {m, Sigma, k3d..K32}, emit post-state.
    Standardization + column consumption + pair kernels + connected rebirth."""
    mu = pre["m"]
    var = np.maximum(np.diag(pre["Sigma"]).copy(), 1e-12)
    sd = np.sqrt(var)
    g = {"g3": np.zeros(n), "g4": np.zeros(n), "g5": np.zeros(n)}
    st = {"g3": np.zeros(n), "g4": np.zeros(n), "g5": np.zeros(n),
          "G21": np.zeros((n, n)), "G31": np.zeros((n, n)),
          "G22": np.zeros((n, n)), "G41": np.zeros((n, n)),
          "G32": np.zeros((n, n))}
    if level != "A":
        g["g3"] = pre["k3d"] / sd ** 3
        st["g3"] = g["g3"]
        st["G21"] = pre["K21"] / (sd[:, None] ** 2 * sd[None, :])
    if level in ("C", "D", "E"):
        g["g4"] = pre["k4d"] / sd ** 4
        st["g4"] = g["g4"]
        st["G31"] = pre["K31"] / (sd[:, None] ** 3 * sd[None, :])
        st["G22"] = pre["K22"] / (sd[:, None] ** 2 * sd[None, :] ** 2)
    if level in ("D", "E"):
        g["g5"] = pre["k5d"] / sd ** 5
        st["g5"] = g["g5"]
        st["G41"] = pre["K41"] / (sd[:, None] ** 4 * sd[None, :])
        st["G32"] = pre["K32"] / (sd[:, None] ** 3 * sd[None, :] ** 2)

    uni = univariate_reset(mu, var, g, level, pmax=5)
    m1, m2, m3, m4, m5 = uni
    post = {"m": m1}
    kaps = univariate_cumulants_from_moments([m1, m2, m3, m4, m5])
    v_post = np.maximum(kaps[1], 1e-12)
    pair = pair_reset(mu, var, pre["Sigma"], st, level)
    n_ = n
    raw = dict(pair)
    raw[(1, 2)] = raw[(2, 1)].T.copy()
    raw[(1, 0)] = np.tile(m1[:, None], (1, n_))
    raw[(2, 0)] = np.tile(m2[:, None], (1, n_))
    raw[(3, 0)] = np.tile(m3[:, None], (1, n_))
    raw[(4, 0)] = np.tile(m4[:, None], (1, n_))
    raw[(0, 1)] = np.tile(m1[None, :], (n_, 1))
    raw[(0, 2)] = np.tile(m2[None, :], (n_, 1))
    central = central_pair_from_raw(raw, raw[(1, 0)], raw[(0, 1)])
    Sigma_post = central[(1, 1)].copy()
    np.fill_diagonal(Sigma_post, v_post)
    post["Sigma"] = Sigma_post
    if level != "A":
        post["k3d"] = kaps[2]
        post["K21"] = connected(2, 1, central)
        np.fill_diagonal(post["K21"], 0.0)
    if level in ("C", "D", "E"):
        post["k4d"] = kaps[3]
        post["K31"] = connected(3, 1, central)
        post["K22"] = connected(2, 2, central)
        np.fill_diagonal(post["K31"], 0.0)
        np.fill_diagonal(post["K22"], 0.0)
    if level in ("D", "E"):
        post["k5d"] = kaps[4]
        post["K41"] = connected(4, 1, central)
        post["K32"] = connected(3, 2, central)
        np.fill_diagonal(post["K41"], 0.0)
        np.fill_diagonal(post["K32"], 0.0)
    return post


def run_ledger(Ws, level):
    """Closed-loop recursion. Layer 1 = exact Gaussian reset."""
    n = Ws[0].shape[0]
    depth = len(Ws)
    zero = np.zeros(n)
    zmat = np.zeros((n, n))
    state = {"m": zero.copy(), "Sigma": np.eye(n), "k3d": zero.copy(),
             "K21": zmat.copy(), "k4d": zero.copy(), "K31": zmat.copy(),
             "K22": zmat.copy(), "k5d": zero.copy(), "K41": zmat.copy(),
             "K32": zmat.copy()}
    posts = []
    for li in range(depth):
        pre = push_state(Ws[li], state, "D" if level in ("D", "E") else level)
        for key in ("k3d", "k4d", "k5d"):
            pre.setdefault(key, zero.copy())
        for key in ("K21", "K31", "K22", "K41", "K32"):
            pre.setdefault(key, zmat.copy())
        lev_here = "A" if li == 0 else level
        post = reset_state(pre, lev_here, n)
        full = {"m": post["m"], "Sigma": post["Sigma"], "k3d": zero.copy(),
                "K21": zmat.copy(), "k4d": zero.copy(), "K31": zmat.copy(),
                "K22": zmat.copy(), "k5d": zero.copy(), "K41": zmat.copy(),
                "K32": zmat.copy()}
        for key in ("k3d", "K21", "k4d", "K31", "K22", "k5d", "K41", "K32"):
            if key in post:
                full[key] = post[key]
        state = full
        posts.append(post)
    return posts
