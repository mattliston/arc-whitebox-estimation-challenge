#!/usr/bin/env python3
"""TAG quadnnls-v3: refit degree masses from the v2 centered curve.

Fixes the v2 NNLS blowup: degrees whose basis column is numerically ~0 on
|rho|<=0.99 (e.g. 8192) gave denormal Gram diagonals and exploding CD steps.
Basis pruned to identifiable degrees; diagonal floored; convergence checked.
Reads curve_v2.json, writes nnls_v3.json.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEGS = list(range(1, 81)) + [100, 101, 140, 141, 200, 201, 300, 301, 460, 461]


def nnls_cd(A, y, w, iters=20000):
    Aw = A * w[:, None]
    G = A.T @ Aw
    b = Aw.T @ y
    c = np.zeros(A.shape[1])
    d = np.maximum(np.diag(G).copy(), 1e-10 * np.diag(G).max())
    for it in range(iters):
        delta = 0.0
        for j in range(len(c)):
            r = b[j] - G[j] @ c + G[j, j] * c[j]
            new = max(0.0, r / d[j])
            delta = max(delta, abs(new - c[j]))
            c[j] = new
        if delta < 1e-12:
            break
    return c, it


def main():
    cv = json.loads((HERE / "curve_v2.json").read_text())
    rhos = sorted([float(r) for r in cv["chat_final"]])
    y = np.array([cv["chat_final"][repr(r) if repr(r) in cv["chat_final"] else str(r)] for r in rhos])
    se = np.array([cv["cse_final"][repr(r) if repr(r) in cv["cse_final"] else str(r)] for r in rhos])
    rho_arr = np.array(rhos)
    w = 1.0 / np.maximum(se, 1e-4) ** 2
    A = np.stack([np.sign(rho_arr) ** d * np.abs(rho_arr) ** d for d in DEGS], axis=1)
    keep = np.abs(A).max(axis=0) > 1e-8
    A = A[:, keep]
    degs = [d for d, k in zip(DEGS, keep) if k]
    c, iters = nnls_cd(A, y, w)
    resid = A @ c - y
    chi2 = float(np.sum((resid / np.maximum(se, 1e-4)) ** 2))
    mass = {d: float(v) for d, v in zip(degs, c) if v > 1e-5}
    cum_le = lambda p: float(sum(v for d, v in zip(degs, c) if d <= p))
    out = {
        "tag": "quadnnls-v3",
        "degrees_kept": degs,
        "iters": int(iters),
        "masses": {str(k): v for k, v in sorted(mass.items())},
        "total_fitted": float(c.sum()),
        "mass_le_1": cum_le(1), "mass_le_2": cum_le(2), "mass_le_4": cum_le(4),
        "mass_le_8": cum_le(8), "mass_le_16": cum_le(16), "mass_le_32": cum_le(32),
        "mass_le_64": cum_le(64),
        "odd_mass_le_80": float(sum(v for d, v in zip(degs, c) if d <= 80 and d % 2 == 1)),
        "fit_rms_resid": float(np.sqrt(np.mean(resid**2))),
        "fit_chi2_per_dof": chi2 / max(len(rhos) - int((c > 0).sum()), 1),
    }
    (HERE / "nnls_v3.json").write_text(json.dumps(out, indent=1))
    print("v3 nnls: le1=%.4f le2=%.4f le4=%.4f le8=%.4f le16=%.4f le64=%.4f odd=%.4f tot=%.4f rms=%.2e iters=%d"
          % (out["mass_le_1"], out["mass_le_2"], out["mass_le_4"], out["mass_le_8"],
             out["mass_le_16"], out["mass_le_64"], out["odd_mass_le_80"],
             out["total_fitted"], out["fit_rms_resid"], iters))
    print("masses:", {k: round(v, 4) for k, v in sorted(((int(k), v) for k, v in out["masses"].items()))})


if __name__ == "__main__":
    main()
