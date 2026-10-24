#!/usr/bin/env python3
"""TAG quadcurve-v2: centered c-curve + per-layer profile + weight spectra.

Fixes quadmeasure-v1 P1: the uncentered pair-product estimator carries the
first-order noise term 2 m^T Sigma m (the rho=0 null read +0.147); v2 uses the
CENTERED estimator sum_j (f_j - mhat_j)(f'_j - mhat_j), whose small bias
E[(m-mhat)^2] = V/S0 is subtracted exactly. 384 pairs.
Adds: per-layer aggregate correlation at each rho (depth profile of the true
per-layer linear-mass retention), and singular-value summaries of each W_l
(mechanism for the real nets' high linear mass vs annealed 0.7335/layer).
Same net (rank 35), same self-written forward, one forward at a time.
Run: nice -n 10 python measure_curve_v2.py   (~2 min, RSS < 1 GB)
"""
import json
import math
import resource
import time
from pathlib import Path

import numpy as np

import measure_realnet as base

HERE = Path(__file__).resolve().parent
SEED = 31338601
S0 = 2048
PAIRS = 384
RHOS = base.RHOS
NNLS_DEGS = base.NNLS_DEGS


def main():
    t0 = time.time()
    print("TAG quadcurve-v2 | seed=%d S0=%d pairs=%d rhos=%d" % (SEED, S0, PAIRS, len(RHOS)), flush=True)
    rng = np.random.default_rng(SEED)
    W, WT, prov = base.load_weights()
    out = {"tag": "quadcurve-v2", "rank": base.RANK, "seed": SEED, "provenance": prov}

    # per-layer weight spectra
    spec = []
    for l in range(16):
        sv = np.linalg.svd(W[l].astype(np.float64), compute_uv=False)
        spec.append({"layer": l + 1, "smax": float(sv[0]), "s2": float(sv[1]),
                     "s8": float(sv[7]), "s32": float(sv[31]), "s128": float(sv[127]),
                     "median": float(np.median(sv)),
                     "rms_gain": float(np.sqrt((sv**2).mean()))})
    out["weight_spectra"] = spec
    print("Wspec smax: " + " ".join("%.2f" % s["smax"] for s in spec), flush=True)
    print("Wspec rmsg: " + " ".join("%.2f" % s["rms_gain"] for s in spec), flush=True)

    # stats pass: per-layer post-activation mean/var
    sum_x = np.zeros((16, 1024))
    sum_x2 = np.zeros((16, 1024))
    for s in range(S0):
        z = rng.standard_normal(1024, dtype=np.float32)
        x = z
        for l in range(16):
            x = np.maximum(x @ W[l], np.float32(0.0))
            xf = x.astype(np.float64)
            sum_x[l] += xf
            sum_x2[l] += xf * xf
    m_l = sum_x / S0
    v_l = np.maximum(sum_x2 / S0 - m_l**2, 0.0) * S0 / (S0 - 1)
    denom_l = v_l.sum(axis=1)
    out["per_layer_trvar"] = denom_l.tolist()
    print("trVar per layer: " + " ".join("%.1f" % d for d in denom_l), flush=True)

    # centered curve, all layers
    cross = np.zeros((len(RHOS), 16))
    cross2 = np.zeros((len(RHOS), 16))
    for p in range(PAIRS):
        z = rng.standard_normal(1024, dtype=np.float32)
        wv = rng.standard_normal(1024, dtype=np.float32)
        xs = np.empty((16, 1024))
        x = z
        for l in range(16):
            x = np.maximum(x @ W[l], np.float32(0.0))
            xs[l] = x
        for ri, r in enumerate(RHOS):
            zp = np.float32(r) * z + np.float32(math.sqrt(1.0 - r * r)) * wv
            xp = zp
            for l in range(16):
                xp = np.maximum(xp @ W[l], np.float32(0.0))
                t = float(((xs[l] - m_l[l]) * (xp.astype(np.float64) - m_l[l])).sum())
                cross[ri, l] += t
                cross2[ri, l] += t * t
    cross /= PAIRS
    cross2 /= PAIRS
    se = np.sqrt(np.maximum(cross2 - cross**2, 0.0) / (PAIRS - 1))
    # debias: E[(f-mhat)(f'-mhat)] = Cov + V/S0
    chat = (cross - (denom_l / S0)[None, :]) / denom_l[None, :]
    cse = se / denom_l[None, :]
    out["chat_final"] = {str(r): float(chat[ri, 15]) for ri, r in enumerate(RHOS)}
    out["cse_final"] = {str(r): float(cse[ri, 15]) for ri, r in enumerate(RHOS)}
    out["chat_layers"] = {str(l + 1): {str(r): float(chat[ri, l]) for ri, r in enumerate(RHOS)}
                          for l in range(16)}
    print("v2 chat final: " + " ".join("%.2f:%.4f" % (r, chat[ri, 15]) for ri, r in enumerate(RHOS)), flush=True)
    i01p = RHOS.index(0.1)
    i01m = RHOS.index(-0.1)
    i02p = RHOS.index(0.2)
    i02m = RHOS.index(-0.2)
    lin1 = (chat[i01p] - chat[i01m]) / 0.2
    lin2 = (chat[i02p] - chat[i02m]) / 0.4
    # Richardson: odd part at rho: M1 rho + M3 rho^3 -> (4*lin1 - lin2)/3 kills M3
    lin_rich = (4.0 * lin1 - lin2) / 3.0
    out["per_layer_linear_mass"] = {
        "slope_pm0.1": lin1.tolist(), "slope_pm0.2": lin2.tolist(),
        "richardson": lin_rich.tolist(),
        "retention_ratio_richardson": (lin_rich[1:] / np.maximum(lin_rich[:-1], 1e-9)).tolist(),
    }
    print("linear mass per layer (rich): " + " ".join("%.3f" % v for v in lin_rich), flush=True)

    # NNLS on the final-layer centered curve
    rho_arr = np.array(RHOS)
    y = chat[:, 15]
    wts = 1.0 / np.maximum(cse[:, 15], 1e-6) ** 2
    A = np.stack([rho_arr**d for d in NNLS_DEGS], axis=1)
    c_fit = base.nnls_cd(A, y, wts, iters=6000)
    mass = {int(d): float(c) for d, c in zip(NNLS_DEGS, c_fit) if c > 1e-6}
    idx = {d: i for i, d in enumerate(NNLS_DEGS)}
    cum_le = lambda p: float(sum(c_fit[idx[d]] for d in NNLS_DEGS if d <= p))
    out["nnls_v2"] = {
        "masses": {str(k): v for k, v in sorted(mass.items())},
        "total_fitted": float(c_fit.sum()),
        "mass_le_1": cum_le(1), "mass_le_2": cum_le(2), "mass_le_4": cum_le(4),
        "mass_le_8": cum_le(8), "mass_le_16": cum_le(16), "mass_le_64": cum_le(64),
        "odd_mass_le_80": float(sum(c_fit[idx[d]] for d in range(1, 81, 2))),
        "fit_rms_resid": float(np.sqrt(np.mean((A @ c_fit - y) ** 2))),
        "null_rho0": float(chat[RHOS.index(0.0), 15]),
        "null_rho0_se": float(cse[RHOS.index(0.0), 15]),
    }
    print("v2 nnls: le1=%.4f le2=%.4f le8=%.4f le64=%.4f tot=%.4f null0=%.4f+-%.4f"
          % (out["nnls_v2"]["mass_le_1"], out["nnls_v2"]["mass_le_2"],
             out["nnls_v2"]["mass_le_8"], out["nnls_v2"]["mass_le_64"],
             out["nnls_v2"]["total_fitted"], out["nnls_v2"]["null_rho0"],
             out["nnls_v2"]["null_rho0_se"]), flush=True)

    out["wall_s"] = time.time() - t0
    out["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 30)
    (HERE / "curve_v2.json").write_text(json.dumps(out, indent=1))
    print("DONE quadcurve-v2 wall=%.0fs rss=%.2fGB" % (out["wall_s"], out["peak_rss_gb"]), flush=True)


if __name__ == "__main__":
    main()
