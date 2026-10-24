#!/usr/bin/env python3
"""Exact certificate of the Gaussian triple Hermite series (path + triangle words) via 3D
Gauss-Hermite quadrature (no MC noise). Also reports the triangle share at several correlation levels."""
import json, sys
from math import factorial
import numpy as np
sys.path.insert(0, ".")
from k3gen import hermite_cols
def kappa3_gh(m, v, C, nq=160):
    x, w = np.polynomial.hermite_e.hermegauss(nq); w = w / w.sum()
    L = np.linalg.cholesky(C)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij"); Wt = (w[:, None, None] * w[None, :, None] * w[None, None, :])
    U = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1) @ L.T + m
    R = np.maximum(U, 0.0); ww = Wt.ravel()
    E3 = (ww * R[:, 0] * R[:, 1] * R[:, 2]).sum(); E1 = (ww[:, None] * R).sum(0)
    E01 = (ww * R[:, 0] * R[:, 1]).sum(); E02 = (ww * R[:, 0] * R[:, 2]).sum(); E12 = (ww * R[:, 1] * R[:, 2]).sum()
    return E3 - E1[0] * E12 - E1[1] * E02 - E1[2] * E01 + 2 * np.prod(E1)
def series(m, v, C, nmax=14):
    c1, _, _, sd, _, _ = hermite_cols(m, v); rho = C / np.outer(sd, sd)
    tot = path = 0.0
    for a in range(nmax + 1):
        for b in range(nmax + 1 - a):
            for c in range(nmax + 1 - a - b):
                if (a == 0 and b == 0) or (a == 0 and c == 0) or (b == 0 and c == 0): continue
                wv = c1[a + c][0] * c1[a + b][1] * c1[b + c][2] * rho[0, 1] ** a * rho[1, 2] ** b * rho[0, 2] ** c / (factorial(a) * factorial(b) * factorial(c))
                tot += wv
                if min(a, b, c) == 0: path += wv
    return tot, path
out = {}
for lab, r in [("rho~0.45", 0.45), ("rho~0.2", 0.2), ("rho~0.6", 0.6)]:
    m = np.array([0.3, -0.2, 0.1]); v = np.array([1.0, 0.7, 1.3])
    Rm = np.array([[1, r, r * 0.8], [r, 1, r * 1.1], [r * 0.8, r * 1.1, 1]]); C = Rm * np.sqrt(np.outer(v, v))
    ex = kappa3_gh(m, v, C); tot, path = series(m, v, C)
    out[lab] = {"gh_exact": float(ex), "series_all": float(tot), "rel_err": float(abs(tot - ex) / abs(ex)), "path_share": float(path / tot)}
print(json.dumps(out, indent=1))
