#!/usr/bin/env python3
"""TAG chaos-ceiling: truncation-order ladder for the fused-stream state.

Same tiny world as exact_recurrence_v3 (n=32, r=8, seed 9). Truth
chain: z' = W^T relu(z) on N samples of the shared latent g. Ladder
arms k=2,3,4: identical chain except each layer's relu output is
EMPIRICALLY PROJECTED (per unit) onto the Hermite chaos of g up to
order k before mixing — i.e. exact couplings at every carried order,
zero engine error. Measures the pure STATE-TRUNCATION ceiling that
any analytic engine at order k inherits. Shared draws across arms so
comparisons are noise-cancelled. Gate context: exact-engine (m,a,M)
arm = k=2 (cross-check); preregistered depth-8 gate C/K21 <= 15%.
Usage: run_chaos_ceiling.py [seed]
"""
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 9
N, n, r, DEPTH = 2_000_000, 32, 8, 8
CHUNK = 100_000
KMAX = 4
ARMS = [2, 3, 4]


def multi_indices(r, kmax):
    out = []
    for k in range(kmax + 1):
        for c in itertools.combinations_with_replacement(range(r), k):
            a = np.zeros(r, dtype=np.int64)
            for d in c:
                a[d] += 1
            out.append(a)
    return np.array(out)          # (F, r), graded order


def he_features(gc, mi, fact):
    """Normalized Hermite product features for one chunk (Nc, F)."""
    Nc = gc.shape[0]
    deg = int(mi.max())
    He = np.ones((deg + 1, Nc, r), dtype=np.float32)
    if deg >= 1:
        He[1] = gc
    for k in range(2, deg + 1):
        He[k] = gc * He[k - 1] - (k - 1) * He[k - 2]
    F = mi.shape[0]
    feat = np.ones((Nc, F), dtype=np.float32)
    for f in range(F):
        for d in range(r):
            o = mi[f, d]
            if o:
                feat[:, f] *= He[o, :, d]
    feat /= fact[None, :]
    return feat


def moments(Z):
    mu = Z.mean(0).astype(np.float64)
    zc = (Z - mu).astype(np.float32)
    C = (zc.T @ zc).astype(np.float64) / len(zc)
    K21 = ((zc * zc).T @ zc).astype(np.float64) / len(zc)
    return mu, C, K21


def rel(a, b):
    return round(float(np.linalg.norm(a - b)
                       / max(np.linalg.norm(b), 1e-30)), 4)


def main():
    import os
    rng = np.random.default_rng(SEED)
    A0 = rng.standard_normal((n, r)) * float(os.environ.get("SA", "0.3"))
    M0 = rng.standard_normal((n, r, r)) * float(os.environ.get("SM", "0.25"))
    M0 = 0.5 * (M0 + np.transpose(M0, (0, 2, 1)))
    m0 = rng.standard_normal(n) * float(os.environ.get("SMU", "0.1"))
    Ws = [rng.standard_normal((n, n)).astype(np.float32)
          * np.float32(np.sqrt(2.0 / n)) for _ in range(DEPTH)]
    g = rng.standard_normal((N, r)).astype(np.float32)

    mi = multi_indices(r, KMAX)
    fact = np.array([np.sqrt(np.prod([math.factorial(int(o))
                                      for o in a])) for a in mi],
                    dtype=np.float32)
    orders = mi.sum(1)
    Fcut = {k: int((orders <= k).sum()) for k in ARMS}
    F = mi.shape[0]
    print(f"features F={F} cuts={Fcut}", flush=True)

    # Gram of features (layer-independent), chunked
    G = np.zeros((F, F), dtype=np.float64)
    for s in range(0, N, CHUNK):
        ft = he_features(g[s:s + CHUNK], mi, fact)
        G += (ft.T @ ft).astype(np.float64)
    G /= N

    Z0 = (m0[None, :].astype(np.float32) + g @ A0.T.astype(np.float32)
          + 0.5 * (np.einsum("na,iab,nb->ni", g,
                             M0.astype(np.float32), g, optimize=True)
                   - np.einsum("iaa->i", M0)[None, :].astype(np.float32)))
    Zs = {"truth": Z0.copy()}
    for k in ARMS:
        Zs[k] = Z0.copy()
    out = {k: {} for k in ARMS}
    for L in range(1, DEPTH + 1):
        tm, tC, tK = moments(Zs["truth"])
        for k in ARMS:
            mm, mC, mK = moments(Zs[k])
            out[k][L] = {"mu_rel": rel(mm, tm), "C_rel": rel(mC, tC),
                         "K21_rel": rel(mK, tK)}
            print(f"k={k} L{L}", json.dumps(out[k][L]), flush=True)
        if L == DEPTH:
            break
        W = Ws[L - 1]
        Zs["truth"] = np.maximum(Zs["truth"], 0.0) @ W
        for k in ARMS:
            Fk = Fcut[k]
            ph = np.maximum(Zs[k], 0.0)
            b = np.zeros((Fk, n), dtype=np.float64)
            for s in range(0, N, CHUNK):
                ft = he_features(g[s:s + CHUNK], mi[:Fk], fact[:Fk])
                b += (ft.T @ ph[s:s + CHUNK]).astype(np.float64)
            b /= N
            coef = np.linalg.solve(G[:Fk, :Fk], b).astype(np.float32)
            if k == KMAX and "--save" in sys.argv:
                np.save(HERE / f"ceil_coef_seed{SEED}_L{L}.npy", coef)
            proj = np.empty_like(ph)
            for s in range(0, N, CHUNK):
                ft = he_features(g[s:s + CHUNK], mi[:Fk], fact[:Fk])
                proj[s:s + CHUNK] = ft @ coef
            Zs[k] = proj @ W
            del ph, proj
    (HERE / f"CHAOS_CEILING_seed{SEED}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
