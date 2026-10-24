#!/usr/bin/env python3
"""TAG quadrules-v1: estimator race on the REAL net (rank 35) vs lawful truth.

Rules, all scored as per-coord MSE of the m_16 estimate vs the sealed truth
row (matched forward counts; one forward at a time; self-written forward):
  MC       plain Monte Carlo, N in {1024, 4096, 16384}
  CV       Gaussian-equivalent affine control variate on the SAME draws:
           est = c_eff + mean(f - g), g(z) = z @ A_eff + c_eff (exact mean
           = c_eff; unbiased regardless of calibration quality)
  ANTI     antithetic pairs, N=4096  (prediction VR = 1 + chat(-1))
  GH2      tensor Gauss-Hermite (8 nodes)^2 on the top-2 active directions
           x MC on the complement, N=4096 (prediction: ~64x WORSE than MC:
           pays a 64-point grid to remove <~3% of variance)
  LAT      rank-1 Korobov lattice, N=4093 prime, 8 random shifts, PPF via
           Acklam inverse normal (prediction: <=1.4x gain: only the deg<=2
           mass ~43% is latticeable in principle at these N)
Also: cross-row robustness pass (ranks 36, 120, 234): vbar, affine capture,
Richardson linear mass. Writes realnet_rules.json.
Identity stated for the record: affine CV + antithetic == antithetic alone
(an affine g has pure-odd centered part; pair sums cancel it exactly).
Run: nice -n 10 python realnet_rules.py   (~8-10 min, RSS < 1 GB)
"""
import json
import math
import resource
import time
from pathlib import Path

import numpy as np

import measure_realnet as base

HERE = Path(__file__).resolve().parent
SEED = 47205513
CAL = 2048
SJ2 = 48
MC_NS = [1024, 4096, 16384]
MC_REPS = [12, 8, 5]
ANTI_N, ANTI_REPS = 4096, 8
GH_N, GH_REPS, GH_M = 4096, 6, 8
LAT_N, LAT_SHIFTS = 4093, 8
XRANKS = [36, 120, 234]


def acklam_ppf(u):
    """Inverse normal CDF, Acklam rational approximation (~1e-9 rel)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    u = np.asarray(u, dtype=np.float64)
    x = np.empty_like(u)
    plow, phigh = 0.02425, 1 - 0.02425
    lo = u < plow
    hi = u > phigh
    mid = ~(lo | hi)
    if mid.any():
        q = u[mid] - 0.5
        r = q * q
        x[mid] = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
                 (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    if lo.any():
        q = np.sqrt(-2 * np.log(u[lo]))
        x[lo] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if hi.any():
        q = np.sqrt(-2 * np.log(1 - u[hi]))
        x[hi] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    return x


def calibrate(W, rng, n=CAL):
    sum_h = np.zeros((16, 1024))
    sum_h2 = np.zeros((16, 1024))
    sum_f = np.zeros(1024)
    sum_f2 = np.zeros(1024)
    for _ in range(n):
        z = rng.standard_normal(1024, dtype=np.float32)
        x, hs = base.forward_with_h(W, z)
        sum_f += x
        sum_f2 += x.astype(np.float64) ** 2
        for l in range(16):
            h = hs[l].astype(np.float64)
            sum_h[l] += h
            sum_h2[l] += h * h
    mu = sum_h / n
    sd = np.sqrt(np.maximum(sum_h2 / n - mu**2, 1e-20))
    mhat = sum_f / n
    vhat = np.maximum(sum_f2 / n - mhat**2, 0.0) * n / (n - 1)
    return mu, sd, mhat, vhat


def build_affine(W, mu, sd):
    Aeff = np.eye(1024)
    ceff = np.zeros(1024)
    for l in range(16):
        al = base.Phi(mu[l] / sd[l])
        ml = mu[l] * al + sd[l] * base.phi(mu[l] / sd[l])
        Aeff = (Aeff @ W[l].astype(np.float64)) * al[None, :]
        ceff = (ceff @ W[l].astype(np.float64) - mu[l]) * al + ml
    return Aeff.astype(np.float32), ceff


def top_dirs(W, WT, rng, k=2, sj=SJ2):
    C = np.zeros((1024, 1024))
    for _ in range(sj):
        z = rng.standard_normal(1024, dtype=np.float32)
        x = z
        J = None
        for l in range(16):
            h = x @ W[l]
            mask = (h > 0).astype(np.float32)
            J = mask[:, None] * WT[0] if J is None else mask[:, None] * (WT[l] @ J)
            x = np.maximum(h, np.float32(0.0))
        C += (J.T @ J).astype(np.float64)
    C /= sj
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    cap = evals[order][:k].sum() / evals.sum()
    return evecs[:, order[:k]].astype(np.float32), float(cap)


def mse(est, truth):
    return float(np.mean((est - truth) ** 2))


def main():
    t0 = time.time()
    print("TAG quadrules-v1 | seed=%d cal=%d MC=%s reps=%s anti=%d gh=%d lat=%d shifts=%d"
          % (SEED, CAL, MC_NS, MC_REPS, ANTI_N, GH_N, LAT_N, LAT_SHIFTS), flush=True)
    rng = np.random.default_rng(SEED)
    W, WT, prov = base.load_weights()
    truth = base.load_truth()[15]
    mu, sd, mhat, vhat = calibrate(W, rng)
    vbar = float(vhat.mean())
    Aeff32, ceff = build_affine(W, mu, sd)
    U2, cap2 = top_dirs(W, WT, rng)
    out = {"tag": "quadrules-v1", "seed": SEED, "provenance": prov, "vbar": vbar,
           "cap_top2_gradient": cap2}
    print("calibrated vbar=%.4e cap2(grad)=%.4f t=%.0fs" % (vbar, cap2, time.time() - t0), flush=True)

    # --- MC and CV on shared draws ---
    res = {"MC": {}, "CV": {}}
    for N, R in zip(MC_NS, MC_REPS):
        mses_mc, mses_cv = [], []
        for r in range(R):
            acc_f = np.zeros(1024)
            acc_d = np.zeros(1024)
            for _ in range(N):
                z = rng.standard_normal(1024, dtype=np.float32)
                f = base.forward(W, z).astype(np.float64)
                g = (z @ Aeff32).astype(np.float64) + ceff
                acc_f += f
                acc_d += f - g
            mses_mc.append(mse(acc_f / N, truth))
            mses_cv.append(mse(ceff + acc_d / N, truth))
        for name, arr in (("MC", mses_mc), ("CV", mses_cv)):
            res[name][str(N)] = {"mse_mean": float(np.mean(arr)),
                                 "mse_se": float(np.std(arr, ddof=1) / math.sqrt(R)),
                                 "reps": R}
        print("N=%5d MC=%.3e CV=%.3e ratio=%.3f t=%.0fs"
              % (N, np.mean(mses_mc), np.mean(mses_cv),
                 np.mean(mses_cv) / np.mean(mses_mc), time.time() - t0), flush=True)

    # --- antithetic ---
    mses = []
    for r in range(ANTI_REPS):
        acc = np.zeros(1024)
        for _ in range(ANTI_N // 2):
            z = rng.standard_normal(1024, dtype=np.float32)
            acc += base.forward(W, z).astype(np.float64)
            acc += base.forward(W, -z).astype(np.float64)
        mses.append(mse(acc / ANTI_N, truth))
    res["ANTI"] = {str(ANTI_N): {"mse_mean": float(np.mean(mses)),
                                 "mse_se": float(np.std(mses, ddof=1) / math.sqrt(ANTI_REPS)),
                                 "reps": ANTI_REPS}}
    print("ANTI N=%d mse=%.3e t=%.0fs" % (ANTI_N, np.mean(mses), time.time() - t0), flush=True)

    # --- GH2 x MC ---
    nodes, wts = np.polynomial.hermite_e.hermegauss(GH_M)
    wts = wts / wts.sum()
    grid = np.array([[ya, yb] for ya in nodes for yb in nodes], dtype=np.float32)
    gwts = np.array([wa * wb for wa in wts for wb in wts])
    Mrep = GH_N // (GH_M * GH_M)
    mses = []
    for r in range(GH_REPS):
        acc = np.zeros(1024)
        for i in range(Mrep):
            wv = rng.standard_normal(1024, dtype=np.float32)
            wperp = wv - U2 @ (U2.T @ wv)
            for gpt, gw in zip(grid, gwts):
                zz = U2 @ gpt + wperp
                acc += gw * base.forward(W, zz).astype(np.float64)
        mses.append(mse(acc / Mrep, truth))
    res["GH2"] = {str(GH_N): {"mse_mean": float(np.mean(mses)),
                              "mse_se": float(np.std(mses, ddof=1) / math.sqrt(GH_REPS)),
                              "reps": GH_REPS}}
    print("GH2 N=%d mse=%.3e t=%.0fs" % (GH_N, np.mean(mses), time.time() - t0), flush=True)

    # --- lattice ---
    cand_rng = np.random.default_rng(SEED + 1)
    best_a, best_score = None, None
    i_arr = np.arange(1, LAT_N, dtype=np.float64)
    for a in cand_rng.integers(2, LAT_N - 1, size=48):
        g = np.empty(1024, dtype=np.int64)
        g[0] = 1
        for dd in range(1, 1024):
            g[dd] = (g[dd - 1] * int(a)) % LAT_N
        # P2-lite score on 64 random projections of dims to keep it cheap
        dims = cand_rng.choice(1024, size=64, replace=False)
        frac = np.outer(i_arr, g[dims]).astype(np.float64) / LAT_N % 1.0
        b2 = frac * frac - frac + 1.0 / 6.0
        score = float(np.exp(np.log1p(b2 * 3.0).sum(axis=1)).mean())
        if best_score is None or score < best_score:
            best_a, best_score = int(a), score
    g = np.empty(1024, dtype=np.int64)
    g[0] = 1
    for dd in range(1, 1024):
        g[dd] = (g[dd - 1] * best_a) % LAT_N
    mses = []
    for s in range(LAT_SHIFTS):
        shift = rng.random(1024)
        acc = np.zeros(1024)
        for i in range(LAT_N):
            u = (i * g / LAT_N + shift) % 1.0
            u = np.clip(u, 1e-12, 1 - 1e-12)
            z = acklam_ppf(u).astype(np.float32)
            acc += base.forward(W, z).astype(np.float64)
        mses.append(mse(acc / LAT_N, truth))
    res["LAT"] = {str(LAT_N): {"mse_mean": float(np.mean(mses)),
                               "mse_se": float(np.std(mses, ddof=1) / math.sqrt(LAT_SHIFTS)),
                               "reps": LAT_SHIFTS, "korobov_a": best_a}}
    print("LAT N=%d a=%d mse=%.3e t=%.0fs" % (LAT_N, best_a, np.mean(mses), time.time() - t0), flush=True)

    # exponent fits (MC, CV over the three Ns)
    for name in ("MC", "CV"):
        xs = np.log2(np.array(MC_NS, dtype=float))
        ys = np.log2(np.array([res[name][str(N)]["mse_mean"] for N in MC_NS]))
        slope = float(np.polyfit(xs, ys, 1)[0])
        res[name]["fitted_exponent"] = slope
    out["rules"] = res

    # --- cross-row robustness ---
    xr = {}
    manifest = json.loads((base.STORE / "manifest.json").read_text())
    entries = manifest["roles"]["observer_fit"]["row_manifests"]
    import hashlib
    for rank in XRANKS:
        entry = entries[rank]
        rmp = base.STORE / entry["path"]
        raw = np.load(rmp.parent / "weights.npy", mmap_mode="r")
        Wr = [np.ascontiguousarray(raw[l], dtype=np.float32) for l in range(16)]
        rr = np.random.default_rng(SEED + 100 + rank)
        mu_r = np.zeros((16, 1024))
        sd_r = np.zeros((16, 1024))
        sum_h = np.zeros((16, 1024))
        sum_h2 = np.zeros((16, 1024))
        sum_f = np.zeros(1024)
        sum_f2 = np.zeros(1024)
        n_cal = 1024
        for _ in range(n_cal):
            z = rr.standard_normal(1024, dtype=np.float32)
            x, hs = base.forward_with_h(Wr, z)
            sum_f += x
            sum_f2 += x.astype(np.float64) ** 2
            for l in range(16):
                h = hs[l].astype(np.float64)
                sum_h[l] += h
                sum_h2[l] += h * h
        mu_r = sum_h / n_cal
        sd_r = np.sqrt(np.maximum(sum_h2 / n_cal - mu_r**2, 1e-20))
        mh = sum_f / n_cal
        vh = np.maximum(sum_f2 / n_cal - mh**2, 0.0) * n_cal / (n_cal - 1)
        A_r, c_r = build_affine(Wr, mu_r, sd_r)
        sd2 = np.zeros(1024)
        sdd = np.zeros(1024)
        sf = np.zeros(1024)
        sf2 = np.zeros(1024)
        np2 = 256
        for _ in range(np2):
            z = rr.standard_normal(1024, dtype=np.float32)
            f = base.forward(Wr, z).astype(np.float64)
            gg = (z @ A_r).astype(np.float64) + c_r
            d = f - gg
            sdd += d
            sd2 += d * d
            sf += f
            sf2 += f * f
        var_d = np.maximum(sd2 / np2 - (sdd / np2) ** 2, 0) * np2 / (np2 - 1)
        var_f = np.maximum(sf2 / np2 - (sf / np2) ** 2, 0) * np2 / (np2 - 1)
        # Richardson linear mass
        prs = 96
        rho_set = [-0.2, -0.1, 0.1, 0.2]
        cr = dict((r, 0.0) for r in rho_set)
        for _ in range(prs):
            z = rr.standard_normal(1024, dtype=np.float32)
            wv = rr.standard_normal(1024, dtype=np.float32)
            fz = base.forward(Wr, z).astype(np.float64) - mh
            for r_ in rho_set:
                zp = np.float32(r_) * z + np.float32(math.sqrt(1 - r_ * r_)) * wv
                cr[r_] += float((fz * (base.forward(Wr, zp).astype(np.float64) - mh)).sum())
        for r_ in rho_set:
            cr[r_] = (cr[r_] / prs - vh.sum() / n_cal) / vh.sum()
        lin1 = (cr[0.1] - cr[-0.1]) / 0.2
        lin2 = (cr[0.2] - cr[-0.2]) / 0.4
        xr[str(rank)] = {"mlp_id": entry["mlp_id"], "vbar": float(vh.mean()),
                         "capture_affine": float(1.0 - var_d.sum() / var_f.sum()),
                         "linear_mass_richardson": float((4 * lin1 - lin2) / 3)}
        print("xrank %d: vbar=%.3e cap_affine=%.4f linmass=%.4f t=%.0fs"
              % (rank, xr[str(rank)]["vbar"], xr[str(rank)]["capture_affine"],
                 xr[str(rank)]["linear_mass_richardson"], time.time() - t0), flush=True)
    out["cross_rows"] = xr

    out["wall_s"] = time.time() - t0
    out["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 30)
    (HERE / "realnet_rules.json").write_text(json.dumps(out, indent=1))
    print("DONE quadrules-v1 wall=%.0fs rss=%.2fGB" % (out["wall_s"], out["peak_rss_gb"]), flush=True)


if __name__ == "__main__":
    main()
