#!/usr/bin/env python3
"""TAG quadverify-v1: n=64 depth=6 numeric verification (task-4 gate).

Two experiments, self-contained (synthetic He-init nets, x <- relu(x @ W),
W_ij ~ N(0, 2/n), no data deps; batched eval is fine at this scale):

E1 ENSEMBLE LAW: 24 random nets; per-layer linear (Hermite-1) mass via the
   centered rho-curve Richardson slope. Verifies the annealed branching
   prediction M_1(d) approx 0.733471^d for the ENSEMBLE MEAN, and measures the
   fixed-net scatter (the finite-net coherence that the real 1024-net shows).

E2 ESTIMATOR RACE on one fixed net vs a 2^24-antithetic-sample truth
   (second-seed truth bounds the truth floor):
     MC, ANTI, CV (Gaussian-equivalent affine, exact mean), GH2 (8^2 tensor
     Gauss-Hermite on top-2 gradient directions x MC complement), LAT
     (Korobov rank-1 lattice, random shifts).
   N grid 2^7..2^14 (rule-dependent), 24 reps; fitted log-log exponent and
   the RMSE ratio vs MC at matched N.  FAIL-FAST criterion: does ANY rule
   beat MC by >3x RMSE at matched N?  Predictions (from measured masses of
   THIS net, written before reading results): VR_CV = 1 - M1, VR_ANTI =
   1 + chat(-1), GH2 ~ (grid 64)x worse, LAT exponent ~ -1.
Writes quad_verify.json.  Run: nice -n 10 python quad_verify.py (~4 min)
"""
import json
import math
import resource
import time
from pathlib import Path

import numpy as np

from realnet_rules import acklam_ppf

HERE = Path(__file__).resolve().parent
N64, D6 = 64, 6
SEED = 90125
ENS_NETS = 24
ENS_PAIRS = 4096
TRUTH_N = 1 << 24
RACE_NS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
RACE_REPS = 24
GH_M = 8
LAT_PRIMES = [127, 251, 509, 1021, 2039, 4093, 8191, 16381]


def make_net(rng):
    return [rng.standard_normal((N64, N64)).astype(np.float32) * np.float32(math.sqrt(2.0 / N64))
            for _ in range(D6)]


def fwd_batch(W, Z):
    X = Z
    for w in W:
        X = np.maximum(X @ w, np.float32(0.0))
    return X


def fwd_batch_layers(W, Z):
    X = Z
    outs = []
    for w in W:
        X = np.maximum(X @ w, np.float32(0.0))
        outs.append(X)
    return outs


def layer_stats(W, rng, n=8192):
    Z = rng.standard_normal((n, N64), dtype=np.float32)
    X = Z
    mu, sd, mx, vx = [], [], [], []
    for w in W:
        H = X @ w
        mu.append(H.astype(np.float64).mean(axis=0))
        sd.append(H.astype(np.float64).std(axis=0, ddof=1))
        X = np.maximum(H, np.float32(0.0))
        mx.append(X.astype(np.float64).mean(axis=0))
        vx.append(X.astype(np.float64).var(axis=0, ddof=1))
    return mu, sd, mx, vx


def phi_np(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def Phi_np(x):
    return np.array([0.5 * (1 + math.erf(t / math.sqrt(2))) for t in np.asarray(x).ravel()]).reshape(np.shape(x))


def build_affine(W, mu, sd):
    A = np.eye(N64)
    c = np.zeros(N64)
    for l in range(D6):
        al = Phi_np(mu[l] / sd[l])
        ml = mu[l] * al + sd[l] * phi_np(mu[l] / sd[l])
        A = (A @ W[l].astype(np.float64)) * al[None, :]
        c = (c @ W[l].astype(np.float64) - mu[l]) * al + ml
    return A.astype(np.float32), c


def linear_mass_curve(W, rng, pairs=ENS_PAIRS):
    """Per-layer Richardson linear mass from centered +-0.1/+-0.2 slopes."""
    _, _, mx, vx = layer_stats(W, rng)
    mx = np.array(mx)
    vx = np.array(vx)
    denom = vx.sum(axis=1)
    rset = [-0.2, -0.1, 0.1, 0.2]
    Z = rng.standard_normal((pairs, N64), dtype=np.float32)
    Wv = rng.standard_normal((pairs, N64), dtype=np.float32)
    XL = fwd_batch_layers(W, Z)
    cr = np.zeros((len(rset), D6))
    for ri, r in enumerate(rset):
        Zp = np.float32(r) * Z + np.float32(math.sqrt(1 - r * r)) * Wv
        XPL = fwd_batch_layers(W, Zp)
        for l in range(D6):
            cr[ri, l] = float(((XL[l].astype(np.float64) - mx[l]) *
                               (XPL[l].astype(np.float64) - mx[l])).sum(axis=1).mean())
    chat = cr / denom[None, :]
    lin1 = (chat[2] - chat[1]) / 0.2   # rset order: [-0.2, -0.1, 0.1, 0.2]
    lin2 = (chat[3] - chat[0]) / 0.4
    return (4 * lin1 - lin2) / 3


def main():
    t0 = time.time()
    print("TAG quadverify-v1 | n=%d d=%d nets=%d truthN=%d raceN=%s reps=%d"
          % (N64, D6, ENS_NETS, TRUTH_N, RACE_NS, RACE_REPS), flush=True)
    out = {"tag": "quadverify-v1", "n": N64, "d": D6, "seed": SEED}

    # ---- E1: ensemble law ----
    ens = np.zeros((ENS_NETS, D6))
    for i in range(ENS_NETS):
        rng = np.random.default_rng(SEED + i)
        W = make_net(rng)
        ens[i] = linear_mass_curve(W, rng)
    annealed = 0.733471 ** np.arange(1, D6 + 1)
    out["e1"] = {
        "ensemble_mean_linear_mass": ens.mean(axis=0).tolist(),
        "ensemble_sd": ens.std(axis=0, ddof=1).tolist(),
        "ensemble_se": (ens.std(axis=0, ddof=1) / math.sqrt(ENS_NETS)).tolist(),
        "annealed_prediction": annealed.tolist(),
        "fixed_net_min": ens.min(axis=0).tolist(),
        "fixed_net_max": ens.max(axis=0).tolist(),
    }
    print("E1 ens mean: " + " ".join("%.3f" % v for v in ens.mean(axis=0)), flush=True)
    print("E1 annealed: " + " ".join("%.3f" % v for v in annealed), flush=True)
    print("E1 sd      : " + " ".join("%.3f" % v for v in ens.std(axis=0, ddof=1)), flush=True)

    # ---- E2: race on one fixed net ----
    rng = np.random.default_rng(SEED + 1000)
    W = make_net(rng)
    mu, sd, mx, vx = layer_stats(W, rng, n=16384)
    A32, ceff = build_affine(W, mu, sd)
    vbar = float(np.mean(vx[-1]))

    def truth_run(seed):
        r = np.random.default_rng(seed)
        acc = np.zeros(N64)
        B = 1 << 14
        for _ in range(TRUTH_N // (2 * B)):
            Z = r.standard_normal((B, N64), dtype=np.float32)
            acc += fwd_batch(W, Z).astype(np.float64).sum(axis=0)
            acc += fwd_batch(W, -Z).astype(np.float64).sum(axis=0)
        return acc / TRUTH_N

    T1 = truth_run(SEED + 2000)
    T2 = truth_run(SEED + 3000)
    truth_floor = float(np.mean((T1 - T2) ** 2))
    truth = 0.5 * (T1 + T2)
    out["e2_truth_floor_mse"] = truth_floor
    out["e2_vbar"] = vbar
    print("E2 vbar=%.4f truth floor=%.3e t=%.0fs" % (vbar, truth_floor, time.time() - t0), flush=True)

    # measured chat(-1) and M1 for predictions
    Z = rng.standard_normal((8192, N64), dtype=np.float32)
    F = fwd_batch(W, Z).astype(np.float64)
    Fm = fwd_batch(W, -Z).astype(np.float64)
    mxf = np.array(mx[-1])
    vxf = np.array(vx[-1])
    chat_m1 = float((((F - mxf) * (Fm - mxf)).sum(axis=1)).mean() / vxf.sum())
    G = (Z @ A32).astype(np.float64) + ceff
    var_d = (F - G).var(axis=0, ddof=1)
    vr_cv_direct = float(var_d.sum() / F.var(axis=0, ddof=1).sum())
    out["e2_predictions"] = {
        "vr_cv_direct_paired": vr_cv_direct,
        "vr_anti_pred_1_plus_chat_m1": 1.0 + chat_m1,
        "gh2_pred_factor_worse": float(GH_M**2 * (1 - 0.03)),
        "note": "written before reading race results",
    }
    print("E2 pred: VR_CV=%.3f VR_ANTI=%.3f" % (vr_cv_direct, 1 + chat_m1), flush=True)

    mse = lambda est: float(np.mean((est - truth) ** 2))
    race = {}

    def record(name, N, arr):
        race.setdefault(name, {})[str(N)] = {
            "mse_mean": float(np.mean(arr)),
            "mse_se": float(np.std(arr, ddof=1) / math.sqrt(len(arr))),
            "reps": len(arr)}

    for N in RACE_NS:
        mc, cv, an = [], [], []
        for rep in range(RACE_REPS):
            Z = rng.standard_normal((N, N64), dtype=np.float32)
            F = fwd_batch(W, Z).astype(np.float64)
            G = (Z @ A32).astype(np.float64) + ceff
            mc.append(mse(F.mean(axis=0)))
            cv.append(mse(ceff + (F - G).mean(axis=0)))
            Zh = Z[: N // 2]
            Fh = F[: N // 2]
            Fmh = fwd_batch(W, -Zh).astype(np.float64)
            an.append(mse(0.5 * (Fh.mean(axis=0) + Fmh.mean(axis=0))))
        record("MC", N, mc)
        record("CV", N, cv)
        record("ANTI", N, an)
        print("N=%5d MC=%.3e CV=%.3e ANTI=%.3e t=%.0fs"
              % (N, np.mean(mc), np.mean(cv), np.mean(an), time.time() - t0), flush=True)

    # GH2 on top-2 gradient dirs
    C = np.zeros((N64, N64))
    for _ in range(2048):
        z = rng.standard_normal(N64, dtype=np.float32)
        x = z
        J = None
        for l in range(D6):
            h = x @ W[l]
            mask = (h > 0).astype(np.float32)
            J = mask[:, None] * W[0].T if J is None else mask[:, None] * (W[l].T @ J)
            x = np.maximum(h, np.float32(0.0))
        C += (J.T @ J).astype(np.float64)
    evals, evecs = np.linalg.eigh(C / 2048)
    order = np.argsort(evals)[::-1]
    U2 = evecs[:, order[:2]].astype(np.float32)
    out["e2_cap2_gradient"] = float(evals[order][:2].sum() / evals.sum())
    nodes, wts = np.polynomial.hermite_e.hermegauss(GH_M)
    wts = wts / wts.sum()
    grid = np.array([[a, b] for a in nodes for b in nodes], dtype=np.float32) @ U2.T  # (64, N64)
    gwts = np.array([wa * wb for wa in wts for wb in wts])
    for N in (1024, 4096, 16384):
        Mrep = N // (GH_M * GH_M)
        arr = []
        for rep in range(RACE_REPS // 2):
            Wv = rng.standard_normal((Mrep, N64), dtype=np.float32)
            Wperp = Wv - (Wv @ U2) @ U2.T
            acc = np.zeros(N64)
            for i in range(Mrep):
                pts = grid + Wperp[i][None, :]
                acc += gwts @ fwd_batch(W, pts).astype(np.float64)
            arr.append(mse(acc / Mrep))
        record("GH2", N, arr)
        print("GH2 N=%d mse=%.3e" % (N, np.mean(arr)), flush=True)

    # lattice
    for N in LAT_PRIMES:
        g = np.empty(N64, dtype=np.int64)
        g[0] = 1
        best_a, best_s = None, None
        i_arr = np.arange(1, N, dtype=np.float64)
        crng = np.random.default_rng(SEED + N)
        for cand in crng.integers(2, N - 1, size=32):
            gg = np.empty(N64, dtype=np.int64)
            gg[0] = 1
            for dd in range(1, N64):
                gg[dd] = (gg[dd - 1] * int(cand)) % N
            frac = np.outer(i_arr, gg).astype(np.float64) / N % 1.0
            b2 = frac * frac - frac + 1.0 / 6.0
            s = float(np.exp(np.log1p(3.0 * b2).sum(axis=1)).mean())
            if best_s is None or s < best_s:
                best_a, best_s = int(cand), s
        for dd in range(1, N64):
            g[dd] = (g[dd - 1] * best_a) % N
        arr = []
        for rep in range(16):
            shift = rng.random(N64)
            U = (np.outer(np.arange(N), g) / N + shift) % 1.0
            Zl = acklam_ppf(np.clip(U, 1e-12, 1 - 1e-12)).astype(np.float32)
            arr.append(mse(fwd_batch(W, Zl).astype(np.float64).mean(axis=0)))
        record("LAT", N, arr)
    print("LAT done t=%.0fs" % (time.time() - t0), flush=True)

    # exponents + fail-fast
    fits = {}
    for name, Ns in (("MC", RACE_NS), ("CV", RACE_NS), ("ANTI", RACE_NS),
                     ("GH2", [1024, 4096, 16384]), ("LAT", LAT_PRIMES)):
        xs = np.log2(np.array(Ns, dtype=float))
        ys = np.log2(np.array([race[name][str(N)]["mse_mean"] for N in Ns]))
        fits[name] = float(np.polyfit(xs, ys, 1)[0])
    ratios = {}
    Nref = 4096
    mc_ref = race["MC"][str(Nref)]["mse_mean"]
    lat_ref = race["LAT"]["4093"]["mse_mean"] if "4093" in race["LAT"] else None
    for name in ("CV", "ANTI", "GH2"):
        ratios[name] = race[name][str(Nref)]["mse_mean"] / mc_ref
    ratios["LAT_4093_vs_MC4096"] = (lat_ref / mc_ref) if lat_ref else None
    best_rmse_gain = max(1.0 / math.sqrt(r) for r in
                         [ratios["CV"], ratios["ANTI"], ratios["GH2"]] +
                         ([ratios["LAT_4093_vs_MC4096"]] if lat_ref else []))
    out["e2_race"] = race
    out["e2_exponents"] = fits
    out["e2_ratio_vs_mc_at_4096"] = ratios
    out["fail_fast"] = {"best_rmse_gain_at_matched_N": best_rmse_gain,
                        "beats_3x": bool(best_rmse_gain > 3.0)}
    out["wall_s"] = time.time() - t0
    out["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 30)
    (HERE / "quad_verify.json").write_text(json.dumps(out, indent=1))
    print("exponents: " + json.dumps(fits))
    print("ratios@4096: " + json.dumps({k: (round(v, 3) if v else None) for k, v in ratios.items()}))
    print("FAIL-FAST best RMSE gain=%.2fx beats3x=%s" % (best_rmse_gain, out["fail_fast"]["beats_3x"]))
    print("DONE quadverify-v1 wall=%.0fs rss=%.2fGB" % (out["wall_s"], out["peak_rss_gb"]), flush=True)


if __name__ == "__main__":
    main()
