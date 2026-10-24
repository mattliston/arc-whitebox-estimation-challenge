#!/usr/bin/env python3
"""G21 pair consumer: full Edgeworth-4 bivariate ReLU pair map (Mehler series,
first-order channels dM and second-order kappa3^2 channels dM2), transcribed
from the engine's emission_K21 (run_tensor_recursion_v4.py:59-120) with the
engine's `columns` exec'd VERBATIM. M11 = E[relu(x) relu(y)] for a pair with
means m, variances v, covariance C and cumulants k3 (diag), K21 (plane), k4,
K22, K31. Certified against emission_K21 by the K21post identity (cert_pairmap)."""
import os, sys
from math import erf, factorial
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
G19 = HERE.parent / "public_g19_attribution_v1"
sys.path.insert(0, str(G19))
_src = (G19 / "run_tensor_recursion_v4.py").read_text()
_a = _src.index("def columns("); _b = _src.index("def emission_K21(")
_ns = {"np": np, "factorial": factorial, "_erf": np.vectorize(erf)}
exec(compile(_src[_a:_b], "engine_columns_verbatim", "exec"), _ns)
columns = _ns["columns"]
from run_attribution import edg2_relu_mean


def series(colsA, colsB, a, b, rho, sd, tmax=20):
    acc = np.zeros_like(rho)
    rp = np.ones_like(rho)
    for t in range(tmax + 1):
        if t > 0:
            rp = rp * rho
        acc += rp / factorial(t) * np.outer(colsA[a + t] / sd ** a, colsB[b + t] / sd ** b)
    return acc


def dM(colsA, colsB, rho, sd, k3, k4, K21c, K22c, K31c, tmax=20):
    S = lambda a, b: series(colsA, colsB, a, b, rho, sd, tmax)
    return ((k3[:, None] / 6.0) * S(3, 0) + (k3[None, :] / 6.0) * S(0, 3)
            + (K21c / 2.0) * S(2, 1) + (K21c.T / 2.0) * S(1, 2)
            + (k4[:, None] / 24.0) * S(4, 0) + (k4[None, :] / 24.0) * S(0, 4)
            + (K22c / 4.0) * S(2, 2) + (K31c / 6.0) * S(3, 1) + (K31c.T / 6.0) * S(1, 3))


def dM2(colsA, colsB, rho, sd, k3, K21c, tmax=20):
    S = lambda a, b: series(colsA, colsB, a, b, rho, sd, tmax)
    k3r = k3[:, None]; k3c = k3[None, :]; Kt = K21c.T
    return ((k3r * k3r / 72.0) * S(6, 0) + (k3c * k3c / 72.0) * S(0, 6)
            + (k3r * k3c / 36.0) * S(3, 3)
            + (k3r * K21c / 12.0) * S(5, 1) + (k3r * Kt / 12.0) * S(4, 2)
            + (k3c * Kt / 12.0) * S(1, 5) + (k3c * K21c / 12.0) * S(2, 4)
            + (K21c * K21c / 8.0) * S(4, 2) + (Kt * Kt / 8.0) * S(2, 4)
            + (K21c * Kt / 4.0) * S(3, 3))


def pair_M11_M21(m, v, C, k3, k4, K21c, K22c, K31c, order=2, tmax=20):
    """Return (M11, M21, r1, r2, sd). order: 0 Gaussian, 1 first-order Edgeworth, 2 + kappa3^2."""
    c1, c2, sd, pdf, cdf = columns(m, v)
    rho = np.clip(C / np.outer(sd, sd), -0.9999, 0.9999)
    M11 = series(c1, c1, 0, 0, rho, sd, tmax)
    M21 = series(c2, c1, 0, 0, rho, sd, tmax)
    if order >= 1:
        M11 = M11 + dM(c1, c1, rho, sd, k3, k4, K21c, K22c, K31c, tmax)
        M21 = M21 + dM(c2, c1, rho, sd, k3, k4, K21c, K22c, K31c, tmax)
    if order >= 2:
        M11 = M11 + dM2(c1, c1, rho, sd, k3, K21c, tmax)
        M21 = M21 + dM2(c2, c1, rho, sd, k3, K21c, tmax)
    al = m / sd; ma = -al
    he1, he3 = ma, ma ** 3 - 3 * ma
    g3v, g4v = k3 / sd ** 3, k4 / sd ** 4
    r1 = edg2_relu_mean(m, v, k3, k4)
    r2 = ((m * m + v) * cdf + m * sd * pdf + (g3v / 6) * (2 * v * pdf)
          + (g4v / 24) * (2 * v * he1 * pdf) + (g3v * g3v / 72) * (2 * v * he3 * pdf))
    return M11, M21, r1, r2, sd


def post_cov(m, v, C, k3, k4, K21c, K22c, K31c, order=2, tmax=20):
    """Post-ReLU covariance from the Edgeworth-4 pair map; diagonal = r2 - r1^2."""
    M11, _M21, r1, r2, _sd = pair_M11_M21(m, v, C, k3, k4, K21c, K22c, K31c, order, tmax)
    Cp = M11 - np.outer(r1, r1)
    np.fill_diagonal(Cp, np.maximum(r2 - r1 * r1, 1e-30))
    return Cp


def cert_pairmap(rank=35, L=5):
    """K21post identity vs the engine's emission_K21 (SECOND_ORDER on and off)."""
    _e = {"np": np, "factorial": factorial, "_erf": np.vectorize(erf), "os": os, "sys": sys, "HERE": G19}
    b2 = _src.index("def fourth_emission_1024(")
    exec(compile(_src[_a:b2], "engine_emission_verbatim", "exec"), _e)
    em = _e["emission_K21"]
    st = np.load(G19 / f"states_rank{rank:04d}.npz"); h = np.load(G19 / f"host_cov_rank{rank:04d}_base.npz")
    m = st["host_m"][L - 1].astype(float); v = st["host_v"][L - 1].astype(float)
    k3 = st["host_k3"][L - 1].astype(float); k4 = st["host_k4"][L - 1].astype(float)
    C = h["cov"][L - 1].astype(float); K21 = h["k21"][L - 1].astype(float).copy(); np.fill_diagonal(K21, k3)
    K22 = h["k22"][L - 1].astype(float); K31 = h["k31"][L - 1].astype(float)
    out = {}
    for so, order in (("0", 1), ("1", 2)):
        os.environ["SECOND_ORDER"] = so
        ref = em(m, v, k3, k4, C, K21, K22, K31)
        M11, M21, r1, r2, sd = pair_M11_M21(m, v, C, k3, k4, K21, K22, K31, order)
        mine = M11 * 0 + (M21 - 2 * r1[:, None] * M11 - r2[:, None] * r1[None, :] + 2 * (r1 ** 2)[:, None] * r1[None, :])
        out[f"order{order}_rel"] = float(np.linalg.norm(mine - ref) / np.linalg.norm(ref))
    os.environ.pop("SECOND_ORDER", None)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(cert_pairmap(int(sys.argv[1]) if len(sys.argv) > 1 else 35, int(sys.argv[2]) if len(sys.argv) > 2 else 5)))
