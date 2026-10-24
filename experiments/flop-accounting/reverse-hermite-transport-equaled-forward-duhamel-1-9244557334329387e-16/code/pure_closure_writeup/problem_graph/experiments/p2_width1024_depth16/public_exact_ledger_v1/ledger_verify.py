#!/usr/bin/env python3
"""TAG public_exact_ledger_v1/ledger_verify.py: verification rig for the COMPLETE EXACT
CARRY LEDGER of the mean channel (width-1024 depth-16 ReLU MLP, Phase-2 white-box).

Sections (run: ledger_verify.py T1 [T2 ...]; no args = all):
  T1  scalar emission-series identity: E[relu(h)] column formula
      m = base + sigma*phi(alpha) * sum_q c_q He_{q-2}(-alpha)  vs exact quadrature,
      at truncation levels {gauss, +g3, +g4+g3^2 (incumbent), +order-3 triad,
      +kappa6-free order-4}; slope checks in the eps grading (g3~eps, g4~eps^2, ...).
  T2  n=1024 FROM-DISK reconciliation (no new MC): incumbent univariate mean map
      evaluated at the 8M-sample MC-oracle per-layer inputs vs the oracle post_mean
      (public_oracle_kappa_probe_v1/mc_oracle_deep_rank0035.npz), against the
      PREDICTED missing-column triad with kappa5 = ratio*closure (ratio from
      K5_MEASURED_RESULT.json) and per-unit measured k5 at L8/11/14.
      Layer 1 (exactly Gaussian pre-activations) is the built-in null control.
  T3  n=1024 accumulation: per-layer predicted fresh mean bias pushed through the
      true weight chain with the relu mean gain (J-chain) -> predicted teacher-forced
      MSE vs the measured 8.2e-9 anchor; same with post-ledger residual bias.
  T4  n=16 depth-5 rig: exact m_final by antithetic MC (2e7+) vs the ledger marginal
      recursion at 3+ truncation levels; per-step and end-to-end predicted-vs-measured
      within 2x.
  T5  width power counting: n in {8,16,32} measured partition-class contributions to
      kappa5(h_i) (diag / K41+K32 planes / rest) + slope fits.
  T6  cost table at n=1024 (mult+add counted separately, board convention).

Numeric discipline: plain numpy, float64, small-n MC batched (<=1e6 x n), peak RSS
well under 3GB, no production-estimator import (its univariate map is REIMPLEMENTED
from the read source: 1 + L3 + L4 + L3^2/2 columns, c[1,q] = sigma*phi*He_{q-2}(-alpha)).
Results: LEDGER_VERIFY_RESULTS.json (per-section blocks, appended as sections run).
"""
import json
import sys
import time
from math import comb, factorial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
ORACLE = P2 / "public_oracle_kappa_probe_v1"
RESULTS = HERE / "LEDGER_VERIFY_RESULTS.json"

INV_SQRT_2PI = 0.3989422804014327


# ---------------------------------------------------------------- hermite + map
def hermite_rows(x, qmax):
    """He_0..He_qmax (probabilists') at x, shape (qmax+1, len(x))."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty((qmax + 1,) + x.shape, dtype=np.float64)
    out[0] = 1.0
    if qmax >= 1:
        out[1] = x
    for q in range(2, qmax + 1):
        out[q] = x * out[q - 1] - (q - 1) * out[q - 2]
    return out


def norm_cdf(x):
    from math import sqrt
    from numpy import vectorize
    try:
        from scipy.stats import norm  # noqa
        return norm.cdf(x)
    except Exception:
        from math import erf
        return 0.5 * (1.0 + np.vectorize(erf)(np.asarray(x, dtype=np.float64) / np.sqrt(2.0)))


def mean_columns(mu, var, qmax=12):
    """Return (base, col[q]) with E[relu(h)] = base + sum_{q>=3} c_q col[q],
    col[q] = sigma*phi(alpha)*He_{q-2}(-alpha); c_q = E[He_q(std h)]/q!."""
    mu = np.asarray(mu, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    sigma = np.sqrt(var)
    alpha = mu / sigma
    pdf = np.exp(-0.5 * alpha * alpha) * INV_SQRT_2PI
    cdf = norm_cdf(alpha)
    base = sigma * pdf + mu * cdf
    he = hermite_rows(-alpha, qmax - 2)
    cols = np.zeros((qmax + 1,) + mu.shape, dtype=np.float64)
    for q in range(2, qmax + 1):
        cols[q] = sigma * pdf * he[q - 2]
    return base, cols


# Gram-Charlier coefficients of the density-side Hermite columns, grouped by the
# cumulant monomials the tilt-exponential generates (standardized cumulants g_r):
#   q=3: g3/6           q=4: g4/24          q=5: g5/120
#   q=6: g6/720 + g3^2/72
#   q=7: g7/5040 + g3*g4/144
#   q=8: g8/40320 + g3*g5/720 + g4^2/1152
#   q=9: g9/362880 + g3*g6/4320 + g4*g5/2880 + g3^3/1296
#   q=10: ... + g3^2*g4/1728        q=12: ... + g3^4/31104
def model_mean(mu, var, g3, g4, level, g5=None, g6=None):
    """Mean emission at a truncation level. g_r are STANDARDIZED cumulants.
    Levels: 'gauss', 'g3', 'incumbent' (= 1+L3+L4+L3^2/2, the frozen map),
    'order3' (+ g5/120 He3 + g3g4/144 He5 + g3^3/1296 He7),
    'order4' (+ g6/720 He4 + (g3 g5/720 + g4^2/1152) He6 + g3^2 g4/1728 He8
              + g3^4/31104 He10)."""
    base, col = mean_columns(mu, var, qmax=12)
    m = base.copy()
    if level == "gauss":
        return m
    m = m + (g3 / 6.0) * col[3]
    if level == "g3":
        return m
    m = m + (g4 / 24.0) * col[4] + (g3 * g3 / 72.0) * col[6]
    if level == "incumbent":
        return m
    g5 = np.zeros_like(m) if g5 is None else g5
    m = m + (g5 / 120.0) * col[5] + (g3 * g4 / 144.0) * col[7] \
        + (g3 ** 3 / 1296.0) * col[9]
    if level == "order3":
        return m
    g6 = np.zeros_like(m) if g6 is None else g6
    m = m + (g6 / 720.0) * col[6] + (g3 * g5 / 720.0 + g4 * g4 / 1152.0) * col[8] \
        + (g3 * g3 * g4 / 1728.0) * col[10] + (g3 ** 4 / 31104.0) * col[12]
    if level == "order4":
        return m
    raise ValueError(level)


def missing_columns_from_incumbent(mu, var, g3, g4, g5, g6=0.0):
    """truth - incumbent, through tilt order 4 (kappa7+ and tilt-5 dropped)."""
    base, col = mean_columns(mu, var, qmax=12)
    tri = (g5 / 120.0) * col[5] + (g3 * g4 / 144.0) * col[7] + (g3 ** 3 / 1296.0) * col[9]
    o4 = (g6 / 720.0) * col[6] + (g3 * g5 / 720.0 + g4 * g4 / 1152.0) * col[8] \
        + (g3 * g3 * g4 / 1728.0) * col[10] + (g3 ** 4 / 31104.0) * col[12]
    return tri, o4


def closure_k5_std(g3, g4):
    """Two-vertex diagonal closure (standardized): g5 = 5 g3 g4 - (40/9) g3^3."""
    return 5.0 * g3 * g4 - (40.0 / 9.0) * g3 ** 3


# ---------------------------------------------------------------------- results
def save_block(name, block):
    data = {}
    if RESULTS.exists():
        data = json.loads(RESULTS.read_text())
    data[name] = block
    data["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    RESULTS.write_text(json.dumps(data, indent=1))
    print(f"[{name}] saved to {RESULTS.name}")


# ------------------------------------------------------------------------- T1
def scalar_truth_mean(mu, sigma, a, b, n_leg=80, zlim=13.0):
    """Exact E[relu(mu + sigma*(z + a He2(z) + b He3(z)))], z~N(0,1), by
    root-split composite Gauss-Legendre in z (integrand smooth per piece)."""
    # h(z) = mu + sigma*(z + a(z^2-1) + b(z^3-3z)) as a plain polynomial in z
    coef = np.array([mu - sigma * a, sigma * (1.0 - 3.0 * b), sigma * a, sigma * b])

    def h(z):
        return coef[0] + coef[1] * z + coef[2] * z * z + coef[3] * z ** 3

    roots = np.roots([coef[3], coef[2], coef[1], coef[0]]) if abs(coef[3]) > 0 else (
        np.roots([coef[2], coef[1], coef[0]]) if abs(coef[2]) > 0 else
        np.roots([coef[1], coef[0]]))
    cuts = sorted([float(r.real) for r in np.atleast_1d(roots)
                   if abs(r.imag) < 1e-12 and -zlim < r.real < zlim])
    edges = [-zlim] + cuts + [zlim]
    xg, wg = np.polynomial.legendre.leggauss(n_leg)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        z = 0.5 * (hi - lo) * xg + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * wg
        hz = h(z)
        total += float(np.sum(w * np.maximum(hz, 0.0)
                              * np.exp(-0.5 * z * z) * INV_SQRT_2PI))
    return total


def scalar_exact_cumulants(mu, sigma, a, b, order=12):
    """Exact cumulants of h = mu + sigma*(z + a He2 + b He3): moments by
    Gauss-Hermite (exact for polynomials), cumulants by the moment recursion."""
    ngh = 3 * order // 2 + 8
    zg, wg = np.polynomial.hermite_e.hermegauss(ngh)
    wg = wg / np.sum(wg)
    hz = mu + sigma * (zg + a * (zg ** 2 - 1.0) + b * (zg ** 3 - 3.0 * zg))
    moments = [1.0] + [float(np.sum(wg * hz ** k)) for k in range(1, order + 1)]
    from math import comb
    kappa = [0.0] * (order + 1)
    for r in range(1, order + 1):
        kappa[r] = moments[r] - sum(comb(r - 1, j - 1) * kappa[j] * moments[r - j]
                                    for j in range(1, r))
    return moments, kappa


def t1():
    print("T1: scalar emission-series identity + eps slopes")
    rng = np.random.default_rng(0)
    configs = [
        {"mu": 0.4, "sigma": 1.1, "a0": 0.35, "b0": 0.30},
        {"mu": -0.5, "sigma": 0.9, "a0": 0.30, "b0": -0.25},
        {"mu": 0.0, "sigma": 1.0, "a0": 0.40, "b0": 0.20},
        {"mu": 1.1, "sigma": 1.3, "a0": -0.30, "b0": 0.25},
    ]
    eps_grid = [0.32, 0.16, 0.08, 0.04]
    levels = ["gauss", "g3", "incumbent", "order3", "order4"]
    rows = []
    for ci, cfg in enumerate(configs):
        for eps in eps_grid:
            a, b = cfg["a0"] * eps, cfg["b0"] * eps * eps
            mom, kap = scalar_exact_cumulants(cfg["mu"], cfg["sigma"], a, b)
            mean_h, var_h = kap[1], kap[2]
            sd = np.sqrt(var_h)
            g = {r: kap[r] / sd ** r for r in range(3, 9)}
            truth = scalar_truth_mean(cfg["mu"], cfg["sigma"], a, b)
            row = {"config": ci, "eps": eps, "truth": truth,
                   "g3": g[3], "g4": g[4], "g5": g[5], "g6": g[6]}
            for lv in levels:
                m = model_mean(np.array([mean_h]), np.array([var_h]),
                               np.array([g[3]]), np.array([g[4]]), lv,
                               g5=np.array([g[5]]), g6=np.array([g[6]]))
                row[f"err_{lv}"] = float(m[0] - truth)
            tri, o4 = missing_columns_from_incumbent(
                np.array([mean_h]), np.array([var_h]), np.array([g[3]]),
                np.array([g[4]]), np.array([g[5]]), np.array([g[6]]))
            row["predicted_incumbent_err"] = float(-(tri + o4)[0])
            row["triad_only_predicted"] = float(-tri[0])
            rows.append(row)

    # identity check: err_incumbent ~ -(triad + order4); ratio per row where signal
    checks = []
    for row in rows:
        pred = row["predicted_incumbent_err"]
        meas = row["err_incumbent"]
        if abs(meas) > 1e-14:
            checks.append(meas / pred if abs(pred) > 0 else np.inf)
    # slopes: fit log|err| vs log eps per config/level
    slopes = {}
    for lv, expect in [("g3", 2.0), ("incumbent", 3.0), ("order3", 4.0),
                       ("order4", 5.0)]:
        ss = []
        for ci in range(len(configs)):
            xs, ys = [], []
            for row in rows:
                if row["config"] == ci and abs(row[f"err_{lv}"]) > 1e-15:
                    xs.append(np.log(row["eps"]))
                    ys.append(np.log(abs(row[f"err_{lv}"])))
            if len(xs) >= 3:
                ss.append(float(np.polyfit(xs, ys, 1)[0]))
        slopes[lv] = {"fit": ss, "expected_at_least": expect}
    block = {
        "levels": levels,
        "rows": rows,
        "incumbent_err_over_predicted_ratio": {
            "min": float(np.min(checks)), "max": float(np.max(checks)),
            "median": float(np.median(checks))},
        "slopes": slopes,
        "pass_rule": "ratio in [0.5,2] at eps<=0.16 rows with |err|>1e-12; slopes >= expected-0.35",
    }
    small = [c for c, row in zip(checks, rows) if row["eps"] <= 0.16
             and abs(row["err_incumbent"]) > 1e-12]
    block["pass"] = bool(all(0.5 <= c <= 2.0 for c in small)) and all(
        min(v["fit"]) >= v["expected_at_least"] - 0.35 for v in slopes.values())
    save_block("T1", block)
    print(json.dumps({k: block[k] for k in
                      ("incumbent_err_over_predicted_ratio", "slopes", "pass")},
                     indent=1))


def same_sample_noise_floor(mu, var, n_samples):
    """Per-unit sd of [sample mean of relu(h)] minus [map(sample moments 1..4)]:
    to first order the map cancels the projection of the fluctuation onto
    span{1,h,h^2,h^3,h^4}; the floor is the OLS residual sd of relu on that span
    under the (Gaussian) marginal, / sqrt(N)."""
    z, w = np.polynomial.hermite_e.hermegauss(96)
    w = w / np.sum(w)
    sd = np.sqrt(var)
    h = mu[:, None] + sd[:, None] * z[None, :]
    r = np.maximum(h, 0.0)
    basis = np.stack([np.ones_like(h), h, h * h, h ** 3, h ** 4], axis=1)
    gram = np.einsum("upq,uqr->upr", basis * w[None, None, :],
                     basis.transpose(0, 2, 1))
    rhs = np.einsum("upq,uq->up", basis * w[None, None, :], r)
    beta = np.linalg.solve(gram + 1e-12 * np.eye(5)[None], rhs[..., None])[..., 0]
    fit = np.einsum("up,upq->uq", beta, basis)
    resid_var = np.einsum("q,uq->u", w, (r - fit) ** 2)
    return np.sqrt(np.maximum(resid_var, 0.0) / n_samples)


# ------------------------------------------------------------------------- T2
def t2():
    print("T2: n=1024 from-disk reconciliation (rank 35, 8M-sample oracle)")
    z = np.load(ORACLE / "mc_oracle_deep_rank0035.npz")
    k5z = np.load(ORACLE / "mc_k5_rank0035.npz")
    k5_layers = [int(v) for v in k5z["layers"]]
    pre_mean, pre_var = z["pre_mean"], z["pre_var"]
    k3, k4 = z["k3"], z["k4"]
    post_mean = z["post_mean"]
    se_post = z["se_post_mean"]
    n_layers = pre_mean.shape[0]
    ratio_r = 0.74  # measured pre-k5 / closure masked-median band 0.716-0.768

    per_layer = []
    for li in range(n_layers):
        mu, var = pre_mean[li], pre_var[li]
        sd = np.sqrt(var)
        g3 = k3[li] / sd ** 3
        g4 = k4[li] / sd ** 4
        m_inc = model_mean(mu, var, g3, g4, "incumbent")
        meas = m_inc - post_mean[li]          # measured map error at oracle inputs
        g5_closure = ratio_r * closure_k5_std(g3, g4)
        tri, o4 = missing_columns_from_incumbent(mu, var, g3, g4, g5_closure, 0.0)
        pred = -(tri + o4)                     # predicted (model - truth)
        tri_pure, _ = missing_columns_from_incumbent(mu, var, g3, g4,
                                                     closure_k5_std(g3, g4), 0.0)
        halfless, _ = missing_columns_from_incumbent(mu, var, g3, g4,
                                                     np.zeros_like(g3), 0.0)
        nf = same_sample_noise_floor(mu, var, int(z["n_samples"]))
        sig_rms = float(np.sqrt(max(np.mean(meas ** 2) - np.mean(nf ** 2), 0.0)))
        row = {
            "layer": li + 1,
            "noise_floor_rms_analytic": float(np.sqrt(np.mean(nf ** 2))),
            "rms_measured_noise_corrected": sig_rms,
            "rms_measured_map_err": float(np.sqrt(np.mean(meas ** 2))),
            "rms_predicted_missing": float(np.sqrt(np.mean(pred ** 2))),
            "rms_predicted_triad_only": float(np.sqrt(np.mean(tri ** 2))),
            "rms_predicted_k5free_half": float(np.sqrt(np.mean(halfless ** 2))),
            "rms_predicted_r1_closure": float(np.sqrt(np.mean(tri_pure ** 2))),
            "pearson_meas_vs_pred": float(np.corrcoef(meas, pred)[0, 1]),
            "regression_slope_meas_on_pred": float(
                np.dot(meas, pred) / max(np.dot(pred, pred), 1e-300)),
            "rms_se_post_mean": float(np.sqrt(np.mean(se_post[li] ** 2))),
            "g3_med_abs": float(np.median(np.abs(g3))),
            "g4_med": float(np.median(g4)),
        }
        pred_half = -halfless
        pred_r1 = -(tri_pure + o4)
        row["pearson_meas_vs_k5free_half"] = float(np.corrcoef(meas, pred_half)[0, 1])
        row["pearson_meas_vs_r1_closure"] = float(np.corrcoef(meas, pred_r1)[0, 1])
        if (li + 1) in k5_layers:
            kli = k5_layers.index(li + 1)
            g5_meas = k5z["k5_std"][kli]
            # oracle-deep g3/g4 (same net, tighter) with measured per-unit g5
            tri_m, o4_m = missing_columns_from_incumbent(mu, var, g3, g4, g5_meas, 0.0)
            pred_m = -(tri_m + o4_m)
            row["rms_predicted_measured_k5"] = float(np.sqrt(np.mean(pred_m ** 2)))
            row["pearson_meas_vs_pred_measured_k5"] = float(
                np.corrcoef(meas, pred_m)[0, 1])
            row["regression_slope_meas_on_pred_measured_k5"] = float(
                np.dot(meas, pred_m) / max(np.dot(pred_m, pred_m), 1e-300))
            row["k5_se_rms_std"] = float(np.sqrt(np.mean(k5z["se_k5"][kli] ** 2)))
        per_layer.append(row)
        keep = {"meas": meas, "pred": pred, "pred_tri": -tri,
                "pred_half": pred_half, "pred_r1": pred_r1}
        if li == 0:
            pooled = {k: [] for k in keep}
        if li + 1 >= 10:
            for k, v in keep.items():
                pooled[k].append(v)

    # L16 from the final oracle tap (the FINAL_SENSITIVITY anchor layer)
    zf = np.load(ORACLE / "mc_oracle_final_rank0035.npz")
    mu, var = zf["pre_mean"], zf["pre_var"]
    sd = np.sqrt(var)
    g3, g4 = zf["k3"] / sd ** 3, zf["k4"] / sd ** 4
    meas16 = model_mean(mu, var, g3, g4, "incumbent") - zf["post_mean"]
    g5c = ratio_r * closure_k5_std(g3, g4)
    tri, o4 = missing_columns_from_incumbent(mu, var, g3, g4, g5c, 0.0)
    pred16 = -(tri + o4)
    nf16 = same_sample_noise_floor(mu, var, int(zf["n_samples"]))
    l16 = {
        "layer": 16,
        "noise_floor_rms_analytic": float(np.sqrt(np.mean(nf16 ** 2))),
        "rms_measured_noise_corrected": float(np.sqrt(max(
            np.mean(meas16 ** 2) - np.mean(nf16 ** 2), 0.0))),
        "rms_measured_map_err": float(np.sqrt(np.mean(meas16 ** 2))),
        "rms_predicted_missing": float(np.sqrt(np.mean(pred16 ** 2))),
        "rms_predicted_triad_only": float(np.sqrt(np.mean(tri ** 2))),
        "pearson_meas_vs_pred": float(np.corrcoef(meas16, pred16)[0, 1]),
        "regression_slope_meas_on_pred": float(
            np.dot(meas16, pred16) / np.dot(pred16, pred16)),
        "anchor_final_sensitivity_rms": 1.4589202401693608e-05,
        "g3_med_abs": float(np.median(np.abs(g3))),
        "g4_med": float(np.median(g4)),
    }
    pooled["meas"].append(meas16)
    pooled["pred"].append(pred16)
    pooled["pred_tri"].append(-tri)
    hl, _ = missing_columns_from_incumbent(mu, var, g3, g4, np.zeros_like(g3), 0.0)
    tp, _ = missing_columns_from_incumbent(mu, var, g3, g4, closure_k5_std(g3, g4), 0.0)
    pooled["pred_half"].append(-hl)
    pooled["pred_r1"].append(-(tp + o4))

    pool = {k: np.concatenate(v) for k, v in pooled.items()}
    pooled_stats = {}
    for arm in ("pred", "pred_tri", "pred_half", "pred_r1"):
        p = pool[arm]
        pooled_stats[arm] = {
            "rms": float(np.sqrt(np.mean(p ** 2))),
            "pearson": float(np.corrcoef(pool["meas"], p)[0, 1]),
            "slope_meas_on_arm": float(np.dot(pool["meas"], p) / np.dot(p, p)),
            "explained_var_frac": float(
                1.0 - np.mean((pool["meas"] - p) ** 2) / np.mean(pool["meas"] ** 2)),
        }

    l1 = per_layer[0]
    block = {
        "note": "layer 1 pre-activations are exactly Gaussian: its measured map err "
                "is the same-sample estimator's noise floor (null control)",
        "noise_floor_rms_L1": l1["rms_measured_map_err"],
        "kappa5_ratio_used": ratio_r,
        "per_layer": per_layer,
        "layer16": l16,
        "pooled_L10_16": pooled_stats,
    }
    sig = [r for r in per_layer[1:]
           if r["rms_measured_map_err"] > 2.0 * l1["rms_measured_map_err"]]
    if sig:
        ratios = [r["rms_predicted_missing"] / r["rms_measured_map_err"] for r in sig]
        block["signal_layers"] = [r["layer"] for r in sig]
        block["pred_over_meas_rms_ratio"] = {
            "min": float(np.min(ratios)), "median": float(np.median(ratios)),
            "max": float(np.max(ratios))}
    save_block("T2", block)
    for r in per_layer + [dict(l16, noise_floor_rms_analytic=l16[
            "noise_floor_rms_analytic"])]:
        print(" L{layer:2d} meas {rms_measured_map_err:.3e} noise {noise_floor_rms_analytic:.3e} "
              "sig {rms_measured_noise_corrected:.3e} pred {rms_predicted_missing:.3e} "
              "corr {pearson_meas_vs_pred:+.3f} slope {regression_slope_meas_on_pred:+.3f}".format(
                  **{**{"rms_measured_noise_corrected": float("nan")}, **r}))


# ------------------------------------------------------------------------ T2b
def var_map_model(mu, var, g3, g4):
    """Incumbent p=2 row: E[X+^2] = raw2 + (g3/6)c[2,3] + (g4/24)c[2,4]
    + (g3^2/72)c[2,6]; c[2,q] = 2 sigma^2 phi He_{q-3}(-alpha) for q>=3."""
    sd = np.sqrt(var)
    alpha = mu / sd
    pdf = np.exp(-0.5 * alpha * alpha) * INV_SQRT_2PI
    cdf = norm_cdf(alpha)
    raw1 = sd * pdf + mu * cdf
    raw2 = (mu * mu + var) * cdf + mu * sd * pdf
    he = hermite_rows(-alpha, 10)
    c2 = {q: 2.0 * var * pdf * he[q - 3] for q in range(3, 13)}
    m2 = raw2 + (g3 / 6.0) * c2[3] + (g4 / 24.0) * c2[4] + (g3 * g3 / 72.0) * c2[6]
    return m2, c2


def t2b():
    """Var-channel from disk: (i) the p=2 map truncation at oracle inputs,
    (ii) predicted next-layer pre-var injection from the DIAGONAL post biases
    (var + mean) through W^2 / W-contractions, vs the observed carried pre-var
    noise-corrected growth (~1.2e-4/layer rel)."""
    print("T2b: var-map truncation and pre-var growth attribution (rank 35)")
    z = np.load(ORACLE / "mc_oracle_deep_rank0035.npz")
    ratio_r = 0.74
    store = P2 / ("learned_residual_observer_design/v4_feature_store_v3/"
                  "sealed_target_blind_feature_store_v3")
    manifest = json.loads((store / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][35]
    weights = np.load((store / entry["path"]).parent / "weights.npy", mmap_mode="r")

    rows = []
    for li in range(15):
        mu, var = z["pre_mean"][li], z["pre_var"][li]
        sd = np.sqrt(var)
        g3, g4 = z["k3"][li] / sd ** 3, z["k4"][li] / sd ** 4
        m2_model, c2 = var_map_model(mu, var, g3, g4)
        m1_model = model_mean(mu, var, g3, g4, "incumbent")
        v_model = m2_model - m1_model ** 2
        v_true = z["post_var"][li]
        m_true = z["post_mean"][li]
        meas_v = v_model - v_true
        # predicted missing p=2 columns (order-3 triad + kappa6-free order-4)
        g5 = ratio_r * closure_k5_std(g3, g4)
        miss_m2 = ((g5 / 120.0) * c2[5] + (g3 * g4 / 144.0) * c2[7]
                   + (g3 ** 3 / 1296.0) * c2[9]
                   + (g3 * g5 / 720.0 + g4 * g4 / 1152.0) * c2[8]
                   + (g3 * g3 * g4 / 1728.0) * c2[10] + (g3 ** 4 / 31104.0) * c2[12])
        tri_m, o4_m = missing_columns_from_incumbent(mu, var, g3, g4, g5, 0.0)
        miss_m1 = tri_m + o4_m
        pred_v = -(miss_m2 - 2.0 * m_true * miss_m1)
        scale = np.sqrt(np.mean(v_true ** 2))
        rows.append({
            "layer": li + 1,
            "rms_rel_var_map_err": float(np.sqrt(np.mean(meas_v ** 2)) / scale),
            "rms_rel_var_pred_missing": float(np.sqrt(np.mean(pred_v ** 2)) / scale),
            "pearson": float(np.corrcoef(meas_v, pred_v)[0, 1]),
            "regression_slope_meas_on_pred": float(
                np.dot(meas_v, pred_v) / max(np.dot(pred_v, pred_v), 1e-300)),
        })
        # next-layer pre-var injection from DIAGONAL post biases alone:
        if li < 14:
            W = np.asarray(weights[li + 1], dtype=np.float64)
            inj_var_diag = (W * W).T @ pred_v             # post-var bias -> pre-var
            pre_var_next = z["pre_var"][li + 1]
            rows[-1]["pred_prevar_injection_rel_rms_diagonly"] = float(
                np.sqrt(np.mean((inj_var_diag / pre_var_next) ** 2)))
    block = {
        "observed_carried_prevar_nc_growth_per_layer_rel": 1.2e-04,
        "observed_carried_paircov_nc_L15_rel": 0.00955,
        "rows": rows,
        "reading": "if the diag-only injection is far below the observed 1.2e-4/layer "
                   "pre-var growth, the off-diagonal (pair/Mehler) channel owns the "
                   "var-state error by elimination",
    }
    save_block("T2b", block)
    for r in rows:
        line = (" L{layer:2d} varmap meas {rms_rel_var_map_err:.3e} "
                "pred {rms_rel_var_pred_missing:.3e} corr {pearson:+.3f}")
        if "pred_prevar_injection_rel_rms_diagonly" in r:
            line += "  -> prevar inj diag {pred_prevar_injection_rel_rms_diagonly:.3e}"
        print(line.format(**r))


# ------------------------------------------------------------------------- T3
def t3():
    """n=1024 accumulation: per-layer predicted fresh mean-map bias vectors pushed
    through the true weight chain with the relu mean gain Phi(alpha) -> predicted
    'substituted-machine' final MSE vs the measured 8.2e-9 anchor."""
    print("T3: J-chain accumulation of the predicted per-layer map bias (rank 35)")
    import json as _json
    store = P2 / ("learned_residual_observer_design/v4_feature_store_v3/"
                  "sealed_target_blind_feature_store_v3")
    manifest = _json.loads((store / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][35]
    weights = np.load((store / entry["path"]).parent / "weights.npy", mmap_mode="r")

    z = np.load(ORACLE / "mc_oracle_deep_rank0035.npz")
    zf = np.load(ORACLE / "mc_oracle_final_rank0035.npz")
    ratio_r = 0.74
    biases, alphas = [], []
    for li in range(15):
        mu, var = z["pre_mean"][li], z["pre_var"][li]
        sd = np.sqrt(var)
        g3, g4 = z["k3"][li] / sd ** 3, z["k4"][li] / sd ** 4
        tri, o4 = missing_columns_from_incumbent(
            mu, var, g3, g4, ratio_r * closure_k5_std(g3, g4), 0.0)
        biases.append(-(tri + o4))
        alphas.append(mu / sd)
    mu, var = zf["pre_mean"], zf["pre_var"]
    sd = np.sqrt(var)
    g3, g4 = zf["k3"] / sd ** 3, zf["k4"] / sd ** 4
    tri, o4 = missing_columns_from_incumbent(
        mu, var, g3, g4, ratio_r * closure_k5_std(g3, g4), 0.0)
    biases.append(-(tri + o4))
    alphas.append(mu / sd)

    def accumulate(bs):
        err = np.zeros(1024)
        for li in range(16):
            if li > 0:
                gain = norm_cdf(alphas[li])
                err = gain * (err @ np.asarray(weights[li], dtype=np.float64))
            err = err + bs[li]
        return err

    e16 = accumulate(biases)
    mse_pred = float(np.mean(e16 ** 2))
    rms_fresh_only = float(np.sqrt(np.mean(biases[-1] ** 2)))
    # incoherent bracket: same chain but random signs per layer (5 draws)
    rng = np.random.default_rng(7)
    mse_inco = []
    for _ in range(5):
        bs = [b * rng.choice((-1.0, 1.0)) for b in biases]
        mse_inco.append(float(np.mean(accumulate(bs) ** 2)))
    block = {
        "anchor_substituted_machine_mse": 8.2e-09,
        "predicted_mse_coherent_jchain": mse_pred,
        "predicted_rms": float(np.sqrt(mse_pred)),
        "rms_last_layer_fresh_bias": rms_fresh_only,
        "amplification_vs_single_step": float(np.sqrt(mse_pred) / rms_fresh_only),
        "incoherent_sign_bracket_mse": mse_inco,
        "ratio_pred_over_anchor": mse_pred / 8.2e-09,
        "note": "biases from the r=0.74-closure order-3+order-4(kappa6-free) columns "
                "at oracle inputs; gain chain = Phi(alpha_L) o W_L; anchor run also "
                "carries var/plane-state residuals my mean-only chain omits",
    }
    save_block("T3", block)
    print(_json.dumps(block, indent=1))


# ------------------------------------------------------------------------- T4
def t4_gates():
    """Pre-gates: push chains vs brute-force full-tensor push; c-table vs
    quadrature; connected assembly vs an analytic shared-component pair."""
    import ledger_rig as rig
    rep = {"chains": rig.chain_check()}
    # c-table vs kink-split Gauss-Legendre (the integrand has a kink at -mu/sd)
    xg, wgl = np.polynomial.legendre.leggauss(120)
    errs = []
    for mu, var in ((0.4, 1.3), (-0.6, 0.7), (0.0, 1.0)):
        sd = np.sqrt(var)
        table, _sd, a, pdf = rig.c_table(np.array([mu]), np.array([var]),
                                         pmax=5, qmax=14)
        kink = -mu / sd
        edges = [-13.0, kink, 13.0]
        zs, ws = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            zs.append(0.5 * (hi - lo) * xg + 0.5 * (hi + lo))
            ws.append(0.5 * (hi - lo) * wgl)
        zg = np.concatenate(zs)
        wg = np.concatenate(ws) * np.exp(-0.5 * zg * zg) * INV_SQRT_2PI
        he = rig.hermite_rows(zg, 14)
        xpos = np.maximum(mu + sd * zg, 0.0)
        for p in range(1, 6):
            scale_p = 1e-6 * factorial(p) * sd ** p
            for q in range(0, 13):
                ref = float(np.sum(wg * xpos ** p * he[q]))
                got = float(table[p][q][0])
                errs.append(abs(got - ref) / max(abs(ref), scale_p))
    rep["c_table_max_rel_err"] = float(np.max(errs))
    # connected assembly: X = A + C, Y = B + C (independent gammas, theta=1):
    # cum(X:p, Y:q) = kappa_{p+q}(C) = kC * (p+q-1)! for p,q >= 1
    kA, kB, kC = 1.3, 0.7, 0.9

    def gamma_raw(k, orders):
        out = {0: 1.0}
        for r in range(1, orders + 1):
            out[r] = out[r - 1] * (k + r - 1)
        return out

    gA, gB, gC = (gamma_raw(k, 6) for k in (kA, kB, kC))
    raw = {}
    for p in range(0, 6):
        for q in range(0, 4):
            if p + q > 6:
                continue
            acc = 0.0
            for i in range(p + 1):
                for j in range(q + 1):
                    acc += (comb(p, i) * comb(q, j) * gA[p - i] * gB[q - j]
                            * gC[i + j])
            raw[(p, q)] = np.array([acc])
    mx, my = raw[(1, 0)], raw[(0, 1)]
    central = rig.central_pair_from_raw(raw, mx, my)
    errs2 = []
    for (p, q) in ((2, 1), (3, 1), (2, 2), (4, 1), (3, 2)):
        got = float(rig.connected(p, q, central)[0])
        ref = kC * factorial(p + q - 1)
        errs2.append(abs(got - ref) / abs(ref))
    rep["connected_max_rel_err"] = float(np.max(errs2))
    from math import comb as _comb  # noqa
    return rep


def t4(n=16, depth=5, n_base=int(3e7), seeds=(11, 12)):
    print(f"T4: n={n} depth={depth} closed-loop + teacher-forced rig")
    from math import comb as _comb  # noqa
    import ledger_rig as rig
    gates = t4_gates()
    print(" gates:", json.dumps(gates))
    chain_worst = max(max(v.values()) for v in gates["chains"].values())
    assert chain_worst < 1e-10, gates["chains"]
    assert gates["c_table_max_rel_err"] < 1e-4
    assert gates["connected_max_rel_err"] < 1e-9

    levels = ["A", "B", "C", "D", "E"]
    all_rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        Ws = [np.asarray(rng.standard_normal((n, n)) * np.sqrt(2.0 / n),
                         dtype=np.float64) for _ in range(depth)]
        t0 = time.time()
        mc = rig.mc_truth(Ws, n_base, seed=seed + 1000)
        t_mc = time.time() - t0
        truth_m = mc[depth - 1]["x_mom"][0]
        se_floor = float(np.sqrt(np.mean(
            (mc[depth - 1]["x_mom"][1] - truth_m ** 2)) / (2.0 * n_base)))

        row = {"seed": seed, "mc_wall_s": round(t_mc, 1),
               "mc_se_per_unit_approx": se_floor, "closed_loop": {},
               "teacher_forced": []}
        # closed loop
        for lev in levels:
            posts = rig.run_ledger(Ws, lev)
            err = posts[-1]["m"] - truth_m
            traj, vtraj = [], []
            for li in range(depth):
                tm = mc[li]["x_mom"][0]
                tv = mc[li]["x_mom"][1] - tm ** 2
                traj.append(float(np.sqrt(np.mean(
                    (posts[li]["m"] - tm) ** 2))))
                vtraj.append(float(np.sqrt(np.mean(
                    (np.diag(posts[li]["Sigma"]) - tv) ** 2))
                    / np.sqrt(np.mean(tv ** 2))))
            row["closed_loop"][lev] = {
                "rms_final_mean_err": float(np.sqrt(np.mean(err ** 2))),
                "mse": float(np.mean(err ** 2)),
                "mean_err_rms_by_layer": traj,
                "postvar_rel_rms_by_layer": vtraj,
            }
        # teacher-forced per-step (layers 2..depth), same-sample comparisons
        for li in range(1, depth):
            pre = rig.state_from_mc(mc[li])
            state = {"m": pre["m_pre"], "Sigma": pre["Sigma"],
                     "k3d": pre["k3d"], "K21": pre["K21"], "k4d": pre["k4d"],
                     "K31": pre["K31"], "K22": pre["K22"], "k5d": pre["k5d"],
                     "K41": pre["K41"], "K32": pre["K32"]}
            true_post_m = mc[li]["x_mom"][0]
            sd = np.sqrt(pre["var_pre"])
            g3 = pre["k3d"] / sd ** 3
            g4 = pre["k4d"] / sd ** 4
            g5 = pre["k5d"] / sd ** 5
            g6 = pre["k6d"] / sd ** 6
            tri, o4 = missing_columns_from_incumbent(
                pre["m_pre"], pre["var_pre"], g3, g4, g5, g6)
            entry = {"layer": li + 1,
                     "g3_med_abs": float(np.median(np.abs(g3))),
                     "g4_med": float(np.median(g4)),
                     "g5_med_abs": float(np.median(np.abs(g5))),
                     "g6_med": float(np.median(g6)),
                     "predicted_C_step_rms": float(np.sqrt(np.mean(
                         (tri + o4) ** 2))),
                     "predicted_D_step_rms": float(np.sqrt(np.mean(o4 ** 2)))}
            true_post_cov = mc[li]["x_pair"][(1, 1)] - np.outer(true_post_m,
                                                                true_post_m)
            offmask = ~np.eye(n, dtype=bool)
            for lev in ("A", "B", "C", "D", "E"):
                post = rig.reset_state(state, lev, n)
                meas = post["m"] - true_post_m
                entry[f"measured_step_rms_{lev}"] = float(np.sqrt(np.mean(
                    meas ** 2)))
                dcov = (post["Sigma"] - true_post_cov)[offmask]
                entry[f"paircov_step_rel_rms_{lev}"] = float(
                    np.sqrt(np.mean(dcov ** 2))
                    / np.sqrt(np.mean(true_post_cov[offmask] ** 2)))
                if lev == "C":
                    entry["pearson_C_meas_vs_pred"] = float(np.corrcoef(
                        meas, -(tri + o4))[0, 1])
            entry["ratio_pred_over_meas_C"] = (
                entry["predicted_C_step_rms"] / entry["measured_step_rms_C"]
                if entry["measured_step_rms_C"] > 0 else None)
            row["teacher_forced"].append(entry)
        all_rows.append(row)
        print(json.dumps(row["closed_loop"], indent=1))
        for e in row["teacher_forced"]:
            print("  L{layer} g3~{g3_med_abs:.3f} g4~{g4_med:.3f} "
                  "predC {predicted_C_step_rms:.3e} measC {measured_step_rms_C:.3e} "
                  "measD {measured_step_rms_D:.3e} measE {measured_step_rms_E:.3e} "
                  "corrC {pearson_C_meas_vs_pred:+.3f}".format(**e))
    block = {"gates": gates, "n": n, "depth": depth,
             "n_evals": 2 * n_base, "rows": all_rows}
    save_block("T4", block)
    return block


SECTIONS = {"T1": t1, "T2": t2, "T2b": t2b, "T3": t3, "T4": t4}

if __name__ == "__main__":
    names = sys.argv[1:] or list(SECTIONS)
    for name in names:
        SECTIONS[name]()
