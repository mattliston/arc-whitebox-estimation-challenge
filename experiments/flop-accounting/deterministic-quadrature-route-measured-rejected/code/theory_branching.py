#!/usr/bin/env python3
"""TAG quadtheory-v1: exact Hermite degree-mass law for deep ReLU nets.

Derivation implemented here (no data touched, pure math):
  Under the annealed (mean-field/NNGP) model the centered normalized
  correlation of unit activations across two inputs with correlation rho obeys
    rho_{l+1} = c(rho_l),
    c(rho) = [sqrt(1-rho^2) + rho(pi - arccos rho) - 1] / (pi - 1),
  the exact scalar relu correlation map (arc-cosine kernel, centered).
  Mehler's identity: for any L2 f, E[f(g)f(g')] with corr-rho Gaussians equals
  sum_k ||f_k||^2 rho^k over Hermite degrees k. So the power-series
  coefficients of c are the per-layer Hermite degree masses
    m_k = a_k^2 / (k! sigma^2),  a_k = E[relu(g) He_k(g)],
    a_1 = 1/2,  a_{2j} = (-1)^{j-1}(2j-3)!!/sqrt(2 pi) (j>=1),  odd k>=3: 0.
  c is a PGF (m_k >= 0, sum_{k>=1} m_k = 1, m_0 = 0), so D-fold composition
  is a Galton-Watson process: the Hermite degree of the depth-D function is
  distributed as generation-D total progeny; mean mu^D, mu = c'(1) = pi/(pi-1).
  Low-degree masses are EXACT by windowed convolution: degrees never decrease
  (offspring >= 1), so mass leaving the window never returns.
Outputs theory_branching.json: per-layer masses, exact low-degree cumulative
  table D=1..16, full-distribution quantiles via FFT of the composed PGF on
  the unit circle, analytic curves c_6/c_16 for overlay against measurement.
Preregistered predictions: M_1(16) = 0.733620^16 = 7.005e-3, M_1(6) = 0.15578,
  mean degree mu^16 = 460.4, mu^6 = 9.97.
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KBASE = 400          # per-layer degree cutoff (tail mass lumped at KBASE)
WINDOW = 64          # exact low-degree window
MGRID = 1 << 17      # FFT grid for full degree distribution
DEPTHS = list(range(1, 17))


def relu_layer_masses(K=KBASE):
    """Exact per-layer Hermite degree masses m_k of relu on N(0,1)."""
    var = 0.5 - 1.0 / (2.0 * math.pi)
    m = np.zeros(K + 1)
    m[1] = 0.25 / var
    for j in range(1, K // 2 + 1):
        k = 2 * j
        # (2j-3)!!, with (-1)!! = 1 at j=1
        log_dfact = sum(math.log(2 * i - 3) for i in range(2, j + 1)) if j >= 2 else 0.0
        log_mk = 2.0 * log_dfact - math.log(2.0 * math.pi) - math.lgamma(k + 1) - math.log(var)
        m[k] = math.exp(log_mk)
    deficit = 1.0 - m.sum()
    assert -1e-12 < deficit, deficit
    m[K] += max(deficit, 0.0)  # lump tail low: understates final degrees (conservative)
    return m, float(deficit)


def c_map(rho):
    rho = np.clip(rho, -1.0, 1.0)
    return (np.sqrt(1.0 - rho**2) + rho * (np.pi - np.arccos(rho)) - 1.0) / (np.pi - 1.0)


def window_compose(m, depth, W=WINDOW):
    """Exact degree distribution on 1..W after depth layers; bucket = mass > W.

    One layer: deg_new = sum of X iid copies of deg_old, X ~ m. The truncated
    convolution drops only mass with degree > W (monotone), so the window is
    exact and bucket = 1 - sum(window).
    """
    p = m[: W + 1].copy()  # p[0] == 0 always (m_0 = 0)
    for _ in range(depth - 1):
        q = np.zeros(W + 1)
        conv = np.zeros(W + 1)
        conv[0] = 1.0  # p^{*0}
        for x in range(1, W + 1):
            conv = np.convolve(conv, p[: W + 1])[: W + 1]
            if m[x] > 0.0:
                q += m[x] * conv
        p = q
    bucket = 1.0 - p.sum()
    return p, float(bucket)


def full_distribution(m, depth, M=MGRID):
    """Composed-PGF degree distribution via FFT on the unit circle."""
    w = np.exp(2j * np.pi * np.arange(M) / M)
    ks = np.nonzero(m)[0]
    coeffs = m[ks]

    def base_pgf(z):
        out = np.zeros(z.shape, dtype=complex)
        zp = np.ones_like(z)
        last = 0
        for k, mk in zip(ks, coeffs):
            step = int(k - last)
            if step == 1:
                zp = zp * z
            elif step == 2:
                zp = zp * z * z
            else:
                zp = zp * z**step
            last = int(k)
            out += mk * zp
        return out

    P = base_pgf(w)
    for _ in range(depth - 1):
        P = base_pgf(P)
    # P_j = sum_k p_k e^{+2 pi i jk/M}, so p = fft(P)/M (ifft would reverse the index)
    probs = np.clip(np.fft.fft(P).real / M, 0.0, None)
    probs /= probs.sum()
    cdf = np.cumsum(probs)
    qs = {str(q): int(np.searchsorted(cdf, q)) for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)}
    mean = float((probs * np.arange(M)).sum())
    return qs, mean


def main():
    print("TAG quadtheory-v1 | KBASE=%d WINDOW=%d MGRID=%d" % (KBASE, WINDOW, MGRID))
    m, deficit = relu_layer_masses()
    mu = float(np.dot(np.arange(len(m)), m))
    var = 0.5 - 1.0 / (2.0 * math.pi)
    out = {
        "tag": "quadtheory-v1",
        "relu_var": var,
        "per_layer_masses_head": {str(k): float(m[k]) for k in range(1, 13)},
        "per_layer_tail_deficit_lumped_at_K": deficit,
        "offspring_mean_mu": mu,
        "mu_analytic_pi_over_pi_minus_1": math.pi / (math.pi - 1.0),
    }
    table = {}
    for D in DEPTHS:
        p, bucket = window_compose(m, D)
        cum = np.cumsum(p)
        table[str(D)] = {
            "P_deg_eq_1": float(p[1]),
            "P_le_2": float(cum[2]),
            "P_le_4": float(cum[4]),
            "P_le_8": float(cum[8]),
            "P_le_16": float(cum[16]),
            "P_le_32": float(cum[32]),
            "P_le_64": float(cum[64]),
            "bucket_gt_64": bucket,
        }
        if D in (6, 16):
            qs, mean = full_distribution(m, D)
            table[str(D)]["quantiles"] = qs
            table[str(D)]["mean_degree_fft_aliased"] = mean
            table[str(D)]["mean_degree_analytic"] = (math.pi / (math.pi - 1.0)) ** D
    out["degree_table"] = table
    out["prereg_check"] = {
        "M1_window_16": table["16"]["P_deg_eq_1"],
        "M1_powlaw_16": float(m[1] ** 16),
        "M1_window_6": table["6"]["P_deg_eq_1"],
        "M1_powlaw_6": float(m[1] ** 6),
    }
    rho = np.round(np.arange(-1.0, 1.0001, 0.01), 4)
    cur = rho.copy()
    c6 = None
    c16 = None
    for d in range(1, 17):
        cur = c_map(cur)
        if d == 6:
            c6 = cur.copy()
        if d == 16:
            c16 = cur.copy()
    out["c_curves"] = {
        "rho": rho.tolist(),
        "c6": np.round(c6, 10).tolist(),
        "c16": np.round(c16, 10).tolist(),
    }
    (HERE / "theory_branching.json").write_text(json.dumps(out, indent=1))
    print("per-layer m1=%.6f m2=%.6f m4=%.6f mu=%.6f (pi/(pi-1)=%.6f)"
          % (m[1], m[2], m[4], mu, math.pi / (math.pi - 1)))
    for D in (6, 16):
        r = table[str(D)]
        print("D=%2d  P[deg=1]=%.5f  P<=2=%.5f  P<=8=%.5f  P<=64=%.5f  mean=%.1f  median=%s"
              % (D, r["P_deg_eq_1"], r["P_le_2"], r["P_le_8"], r["P_le_64"],
                 r["mean_degree_analytic"], r["quantiles"]["0.5"]))
    print("window-vs-powerlaw: P[deg=1](16)=%.6e vs m1^16=%.6e"
          % (table["16"]["P_deg_eq_1"], m[1] ** 16))
    print("DONE quadtheory-v1")


if __name__ == "__main__":
    main()
