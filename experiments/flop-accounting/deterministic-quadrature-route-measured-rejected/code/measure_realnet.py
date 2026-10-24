#!/usr/bin/env python3
"""TAG quadmeasure-v1: real-net spectra for the quadrature feasibility verdict.

Net: observer_fit rank 35 (width 1024, depth 16), weights loaded directly from
the sealed store's weights.npy with a SELF-WRITTEN forward
    x <- relu(x @ W[l]),  x0 = z ~ N(0, I_1024), float32,
matching whestbench/simulation.py. NO production estimator import; ONE forward
at a time in this single process (memory discipline; peak RSS reported).
Truth access: lawful observer_fit.parquet all_layer_means row 35 only.

Measures (all feeding QUADRATURE_VERDICT.json):
  P0 stats: per-layer preactivation marginals (mu_l, s_l), final mean/var,
     MC-vs-truth check, layer-1 analytic orientation check.
  P1 c-curve: hat c(rho) = aggregate corr of x_16 across rho-correlated input
     pairs; by Mehler this equals sum_k M_k rho^k with M_k the TRUE Hermite
     degree masses of this exact net. NNLS -> measured mass spectrum.
  P2 Jacobian: C = E[J^T J] spectrum (active subspace), E[J] (Stein best-linear
     capture ||E J||_F^2 / trVar).
  P3 conditional variance: E[Var(f | top-k active subspace)] for k in
     {8,32,128,512} by nested MC = exact variance floor of any rule that
     integrates the top-k subspace PERFECTLY and samples the complement.
  P4 Gaussian-equivalent affine surrogate (the task's CV candidate):
     x~_l = a_l (h~_l - mu_l) + m_l, a_l = Phi(mu_l/s_l) (Stein-optimal slope),
     m_l = mu_l Phi + s_l phi; exact mean = its own constant chain. Measured
     paired variance ratio Var[f - g]/Var[f] = the CV's real variance floor.
Run: nice -n 10 python measure_realnet.py   (~3-5 min, RSS < 1 GB)
"""
import hashlib
import json
import math
import resource
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P2DIR = HERE.parent
STORE = P2DIR / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
TARGETS = P2DIR / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
RANK = 35
SEED = 20260825
S0 = 2048            # stats pass forwards
PAIRS = 64           # c-curve pairs per rho
RHOS = [-0.95, -0.8, -0.6, -0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]
SJ = 160             # jacobian samples
COND_KS = [8, 32, 128, 512]
COND_OUTER = 16
COND_INNER = 32
P4N = 1024           # paired surrogate forwards
NNLS_DEGS = list(range(1, 81)) + [128, 129, 512, 513, 2048, 2049, 8192, 8193]


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 30)


def load_weights():
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][RANK]
    assert entry["role_rank"] == RANK
    rmp = STORE / entry["path"]
    row = json.loads(rmp.read_text())
    raw = np.load(rmp.parent / "weights.npy", mmap_mode="r")
    sha = hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()
    W = [np.ascontiguousarray(raw[l], dtype=np.float32) for l in range(16)]
    WT = [np.ascontiguousarray(w.T) for w in W]
    return W, WT, {"mlp_id": entry["mlp_id"], "weights_sha256": sha,
                   "weights_raw_sha256_row": row.get("weights_raw_sha256")}


def load_truth():
    import pyarrow.parquet as pq
    import pyarrow as pa
    table = pq.read_table(TARGETS, columns=["all_layer_means"])
    values = table.column("all_layer_means").combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    t = storage.to_numpy(zero_copy_only=False).reshape(688, 16, 1024)[RANK].astype(np.float64)
    return t.copy()


def forward(W, z):
    x = z
    for w in W:
        x = np.maximum(x @ w, np.float32(0.0))
    return x


def forward_with_h(W, z):
    x = z
    hs = []
    for w in W:
        h = x @ w
        hs.append(h)
        x = np.maximum(h, np.float32(0.0))
    return x, hs


def phi(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def Phi(x):
    v = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.array([math.erf(t / math.sqrt(2.0)) for t in v.ravel()]).reshape(v.shape))


def nnls_cd(A, y, w, iters=4000):
    """min ||diag(sqrt w)(A c - y)||^2, c >= 0, cyclic coordinate descent."""
    Aw = A * w[:, None]
    G = A.T @ Aw
    b = Aw.T @ y
    c = np.zeros(A.shape[1])
    d = np.diag(G).copy()
    d[d == 0] = 1.0
    for _ in range(iters):
        for j in range(len(c)):
            r = b[j] - G[j] @ c + G[j, j] * c[j]
            c[j] = max(0.0, r / d[j])
    return c


def main():
    t0 = time.time()
    print("TAG quadmeasure-v1 | rank=%d seed=%d S0=%d pairs=%d SJ=%d condO/I=%d/%d P4N=%d"
          % (RANK, SEED, S0, PAIRS, SJ, COND_OUTER, COND_INNER, P4N), flush=True)
    rng = np.random.default_rng(SEED)
    W, WT, prov = load_weights()
    truth = load_truth()
    out = {"tag": "quadmeasure-v1", "rank": RANK, "seed": SEED, "provenance": prov}

    # ---- orientation check: layer-1 mean is analytic ----
    col_sig = np.sqrt((W[0].astype(np.float64) ** 2).sum(axis=0))
    row_sig = np.sqrt((W[0].astype(np.float64) ** 2).sum(axis=1))
    pred_col = col_sig / math.sqrt(2.0 * math.pi)
    pred_row = row_sig / math.sqrt(2.0 * math.pi)
    rel = lambda p: float(np.sqrt(np.mean((p - truth[0]) ** 2) / np.mean(truth[0] ** 2)))
    out["orientation"] = {"rel_rms_col_convention": rel(pred_col),
                          "rel_rms_row_convention": rel(pred_row)}
    print("orientation: col %.3e row %.3e (col must be ~f32-eps)" %
          (out["orientation"]["rel_rms_col_convention"], out["orientation"]["rel_rms_row_convention"]), flush=True)

    # ---- P0: stats pass ----
    sum_f = np.zeros(1024)
    sum_f2 = np.zeros(1024)
    sum_h = np.zeros((16, 1024))
    sum_h2 = np.zeros((16, 1024))
    for s in range(S0):
        z = rng.standard_normal(1024, dtype=np.float32)
        x, hs = forward_with_h(W, z)
        sum_f += x
        sum_f2 += x.astype(np.float64) ** 2
        for l in range(16):
            h = hs[l].astype(np.float64)
            sum_h[l] += h
            sum_h2[l] += h * h
    mhat = sum_f / S0
    vhat = np.maximum(sum_f2 / S0 - mhat**2, 0.0) * S0 / (S0 - 1)
    mu_l = sum_h / S0
    s_l = np.sqrt(np.maximum(sum_h2 / S0 - mu_l**2, 1e-20))
    vbar = float(vhat.mean())
    mc_mse = float(np.mean((mhat - truth[15]) ** 2))
    out["p0"] = {
        "vbar_percoord_mean_variance": vbar,
        "mean_abs_final_mean": float(np.abs(truth[15]).mean()),
        "mc2048_vs_truth_mse": mc_mse,
        "mc2048_predicted_mse": vbar / S0,
        "pred_mc_mse_at_65536": vbar / 65536.0,
        "layer_h_sigma_mean": s_l.mean(axis=1).tolist(),
    }
    print("P0 vbar=%.4e mc2048 mse=%.3e (pred %.3e) predMC@65536=%.3e  rss=%.2fGB"
          % (vbar, mc_mse, vbar / S0, vbar / 65536.0, rss_gb()), flush=True)

    # ---- P1: c-curve ----
    m2corr = float((mhat**2).sum() - vhat.sum() / S0)  # debiased sum m_j^2
    denom = float(vhat.sum())
    cross = {r: [] for r in RHOS}
    for p in range(PAIRS):
        z = rng.standard_normal(1024, dtype=np.float32)
        wv = rng.standard_normal(1024, dtype=np.float32)
        fz = forward(W, z).astype(np.float64)
        for r in RHOS:
            zp = (np.float32(r) * z + np.float32(math.sqrt(1.0 - r * r)) * wv)
            fp = forward(W, zp).astype(np.float64)
            cross[r].append(float(fz @ fp))
    chat, cse = {}, {}
    for r in RHOS:
        arr = np.array(cross[r])
        est = (arr.mean() - m2corr) / denom
        chat[r] = float(est)
        cse[r] = float(arr.std(ddof=1) / math.sqrt(len(arr)) / denom)
    out["p1_chat"] = {str(r): chat[r] for r in RHOS}
    out["p1_se"] = {str(r): cse[r] for r in RHOS}
    print("P1 chat: " + " ".join("%.2f:%.4f" % (r, chat[r]) for r in RHOS), flush=True)

    # NNLS mass fit
    rho_arr = np.array(RHOS)
    y = np.array([chat[r] for r in RHOS])
    wts = 1.0 / np.maximum(np.array([cse[r] for r in RHOS]), 1e-6) ** 2
    A = np.stack([np.sign(rho_arr) ** d * np.abs(rho_arr) ** d for d in NNLS_DEGS], axis=1)
    c_fit = nnls_cd(A, y, wts)
    fit_resid = float(np.sqrt(np.mean((A @ c_fit - y) ** 2)))
    mass = {int(d): float(c) for d, c in zip(NNLS_DEGS, c_fit) if c > 1e-6}
    cum = np.cumsum([c_fit[NNLS_DEGS.index(d)] if d in NNLS_DEGS else 0.0 for d in range(1, 81)])
    out["p1_nnls"] = {
        "masses": {str(k): v for k, v in sorted(mass.items())},
        "total_fitted": float(c_fit.sum()),
        "mass_le_1": float(cum[0]), "mass_le_2": float(cum[1]),
        "mass_le_4": float(cum[3]), "mass_le_8": float(cum[7]),
        "mass_le_16": float(cum[15]), "mass_le_64": float(cum[63]),
        "odd_mass_le_80": float(sum(c_fit[NNLS_DEGS.index(d)] for d in range(1, 81, 2))),
        "fit_rms_resid": fit_resid,
    }
    print("P1 nnls: le1=%.4f le2=%.4f le8=%.4f le64=%.4f tot=%.4f resid=%.2e"
          % (out["p1_nnls"]["mass_le_1"], out["p1_nnls"]["mass_le_2"],
             out["p1_nnls"]["mass_le_8"], out["p1_nnls"]["mass_le_64"],
             out["p1_nnls"]["total_fitted"], fit_resid), flush=True)

    # ---- P2: Jacobian spectrum ----
    C = np.zeros((1024, 1024))
    Jsum = np.zeros((1024, 1024))
    for s in range(SJ):
        z = rng.standard_normal(1024, dtype=np.float32)
        x = z
        J = None
        for l in range(16):
            h = x @ W[l]
            mask = (h > 0).astype(np.float32)
            if J is None:
                J = mask[:, None] * WT[0]
            else:
                J = mask[:, None] * (WT[l] @ J)
            x = np.maximum(h, np.float32(0.0))
        C += (J.T @ J).astype(np.float64)
        Jsum += J.astype(np.float64)
        if (s + 1) % 40 == 0:
            print("P2 jac %d/%d rss=%.2fGB t=%.0fs" % (s + 1, SJ, rss_gb(), time.time() - t0), flush=True)
    C /= SJ
    EJ = Jsum / SJ
    evals = np.linalg.eigvalsh(C)[::-1]
    evals = np.maximum(evals, 0.0)
    cumfrac = np.cumsum(evals) / evals.sum()
    kat = lambda f: int(np.searchsorted(cumfrac, f) + 1)
    best_lin_capture = float((EJ**2).sum() / vhat.sum())
    out["p2"] = {
        "trace_C": float(evals.sum()),
        "trVar": float(vhat.sum()),
        "poincare_slack_trC_over_trVar": float(evals.sum() / vhat.sum()),
        "top_eigs": evals[:16].tolist(),
        "k_at_50pct": kat(0.50), "k_at_90pct": kat(0.90),
        "k_at_99pct": kat(0.99), "k_at_99.9pct": kat(0.999),
        "cum_frac_at_k": {str(k): float(cumfrac[k - 1]) for k in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]},
        "best_linear_capture_stein": best_lin_capture,
    }
    print("P2 k50=%d k90=%d k99=%d k99.9=%d  lin_capture=%.5f  slack=%.2f"
          % (out["p2"]["k_at_50pct"], out["p2"]["k_at_90pct"], out["p2"]["k_at_99pct"],
             out["p2"]["k_at_99.9pct"], best_lin_capture,
             out["p2"]["poincare_slack_trC_over_trVar"]), flush=True)
    Uall = None
    evals_v, evecs = np.linalg.eigh(C)
    order = np.argsort(evals_v)[::-1]
    evecs = evecs[:, order].astype(np.float32)

    # ---- P3: conditional variance (nested MC) ----
    out["p3"] = {}
    for k in COND_KS:
        U = evecs[:, :k]
        within = np.zeros(1024)
        for o in range(COND_OUTER):
            z0 = rng.standard_normal(1024, dtype=np.float32)
            a = U @ (U.T @ z0)
            fs = np.empty((COND_INNER, 1024))
            for i in range(COND_INNER):
                wv = rng.standard_normal(1024, dtype=np.float32)
                zi = a + wv - U @ (U.T @ wv)
                fs[i] = forward(W, zi)
            within += fs.var(axis=0, ddof=1)
        within /= COND_OUTER
        resid_frac = float(within.sum() / vhat.sum())
        out["p3"][str(k)] = {"resid_frac": resid_frac, "capture": 1.0 - resid_frac}
        print("P3 k=%d resid_frac=%.4f capture=%.4f rss=%.2fGB"
              % (k, resid_frac, 1.0 - resid_frac, rss_gb()), flush=True)

    # ---- P4: Gaussian-equivalent affine surrogate ----
    a_l = np.empty((16, 1024))
    em_l = np.empty((16, 1024))
    for l in range(16):
        al = Phi(mu_l[l] / s_l[l])
        a_l[l] = al
        em_l[l] = mu_l[l] * al + s_l[l] * phi(mu_l[l] / s_l[l])
    Aeff = np.eye(1024)
    ceff = np.zeros(1024)
    for l in range(16):
        Aeff = (Aeff @ W[l].astype(np.float64)) * a_l[l][None, :]
        ceff = (ceff @ W[l].astype(np.float64) - mu_l[l]) * a_l[l] + em_l[l]
    Aeff32 = Aeff.astype(np.float32)
    ceff32 = ceff.astype(np.float32)
    surro_mean_mse = float(np.mean((ceff - truth[15]) ** 2))
    sfd = np.zeros(1024)
    sfd2 = np.zeros(1024)
    sf = np.zeros(1024)
    sf2 = np.zeros(1024)
    sg = np.zeros(1024)
    sg2 = np.zeros(1024)
    sfg = np.zeros(1024)
    for s in range(P4N):
        z = rng.standard_normal(1024, dtype=np.float32)
        f = forward(W, z).astype(np.float64)
        g = (z @ Aeff32 + ceff32).astype(np.float64)
        d = f - g
        sfd += d
        sfd2 += d * d
        sf += f
        sf2 += f * f
        sg += g
        sg2 += g * g
        sfg += f * g
    n = P4N
    var_d = np.maximum(sfd2 / n - (sfd / n) ** 2, 0.0) * n / (n - 1)
    var_f = np.maximum(sf2 / n - (sf / n) ** 2, 0.0) * n / (n - 1)
    var_g = np.maximum(sg2 / n - (sg / n) ** 2, 0.0) * n / (n - 1)
    cov_fg = (sfg / n - (sf / n) * (sg / n)) * n / (n - 1)
    vr = float(var_d.sum() / var_f.sum())
    vr_opt = float(np.sum(var_f - cov_fg**2 / np.maximum(var_g, 1e-20)) / var_f.sum())
    out["p4"] = {
        "vr_affine_paired": vr,
        "capture_affine": 1.0 - vr,
        "vr_percoord_optimal_beta": vr_opt,
        "capture_optimal_beta": 1.0 - vr_opt,
        "surrogate_exact_mean_vs_truth_mse": surro_mean_mse,
        "aggregate_corr_fg": float(cov_fg.sum() / math.sqrt(var_f.sum() * var_g.sum())),
    }
    print("P4 VR=%.5f capture=%.5f (opt-beta %.5f) surro-mean mse=%.3e"
          % (vr, 1.0 - vr, 1.0 - vr_opt, surro_mean_mse), flush=True)

    out["wall_s"] = time.time() - t0
    out["peak_rss_gb"] = rss_gb()
    (HERE / "realnet_measurements.json").write_text(json.dumps(out, indent=1))
    print("DONE quadmeasure-v1  wall=%.0fs rss=%.2fGB" % (out["wall_s"], out["peak_rss_gb"]), flush=True)


if __name__ == "__main__":
    main()
