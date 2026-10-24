#!/usr/bin/env python3
"""TAG shared-rank: rank-R shared-basis carry of H3/H4 in the recurrence.

Same world/machinery as run_chaos_ceiling (imports it). Arms: k=2,
k=4 exact, and k=4 with the order-3 and order-4 coefficient blocks
truncated per layer to a rank-R SHARED basis across units (SVD),
R in {4,8,16}. Shared-basis transport is exactly mixing-closed, so
this measures the ceiling of the deployable streaming-subspace carry
(birth-update-truncate). Gate: R~8-12 recovers >=90% of the k=4 arm's
gain over k=2 at L8 on the HARD world (seed 10).
Usage: run_shared_rank.py [seed]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_chaos_ceiling import (CHUNK, DEPTH, KMAX, N, he_features,
                               moments, multi_indices, n, r, rel)

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 10
ARMS = [("k2", 2, None), ("k4", 4, None), ("k4r4", 4, 4),
        ("k4r8", 4, 8), ("k4r16", 4, 16)]


def main():
    import math
    rng = np.random.default_rng(SEED)
    A0 = rng.standard_normal((n, r)) * 0.3
    M0 = rng.standard_normal((n, r, r)) * 0.25
    M0 = 0.5 * (M0 + np.transpose(M0, (0, 2, 1)))
    m0 = rng.standard_normal(n) * 0.1
    Ws = [rng.standard_normal((n, n)).astype(np.float32)
          * np.float32(np.sqrt(2.0 / n)) for _ in range(DEPTH)]
    g = rng.standard_normal((N, r)).astype(np.float32)

    mi = multi_indices(r, KMAX)
    fact = np.array([np.sqrt(np.prod([math.factorial(int(o))
                                      for o in a])) for a in mi],
                    dtype=np.float32)
    orders = mi.sum(1)
    idx3 = np.where(orders == 3)[0]
    idx4 = np.where(orders == 4)[0]
    Fcut = {2: int((orders <= 2).sum()), 4: int((orders <= 4).sum())}

    G = np.zeros((mi.shape[0], mi.shape[0]), dtype=np.float64)
    for s in range(0, N, CHUNK):
        ft = he_features(g[s:s + CHUNK], mi, fact)
        G += (ft.T @ ft).astype(np.float64)
    G /= N

    Z0 = (m0[None, :].astype(np.float32) + g @ A0.T.astype(np.float32)
          + 0.5 * (np.einsum("na,iab,nb->ni", g,
                             M0.astype(np.float32), g, optimize=True)
                   - np.einsum("iaa->i", M0)[None, :].astype(np.float32)))
    Zs = {"truth": Z0.copy()}
    for name, _, _ in ARMS:
        Zs[name] = Z0.copy()
    out = {name: {} for name, _, _ in ARMS}
    for L in range(1, DEPTH + 1):
        tm, tC, tK = moments(Zs["truth"])
        for name, _, _ in ARMS:
            mm, mC, mK = moments(Zs[name])
            out[name][L] = {"mu_rel": rel(mm, tm),
                            "C_rel": rel(mC, tC),
                            "K21_rel": rel(mK, tK)}
            print(f"{name} L{L}", json.dumps(out[name][L]), flush=True)
        if L == DEPTH:
            break
        W = Ws[L - 1]
        Zs["truth"] = np.maximum(Zs["truth"], 0.0) @ W
        for name, k, R in ARMS:
            Fk = Fcut[k]
            ph = np.maximum(Zs[name], 0.0)
            b = np.zeros((Fk, n), dtype=np.float64)
            for s in range(0, N, CHUNK):
                ft = he_features(g[s:s + CHUNK], mi[:Fk], fact[:Fk])
                b += (ft.T @ ph[s:s + CHUNK]).astype(np.float64)
            b /= N
            coef = np.linalg.solve(G[:Fk, :Fk], b)
            if R is not None:
                for idx in (idx3, idx4):
                    B = coef[idx, :]
                    U, s_, Vt = np.linalg.svd(B, full_matrices=False)
                    coef[idx, :] = (U[:, :R] * s_[:R]) @ Vt[:R]
            coef = coef.astype(np.float32)
            proj = np.empty_like(ph)
            for s in range(0, N, CHUNK):
                ft = he_features(g[s:s + CHUNK], mi[:Fk], fact[:Fk])
                proj[s:s + CHUNK] = ft @ coef
            Zs[name] = proj @ W
            del ph, proj
    (HERE / f"SHARED_RANK_seed{SEED}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
