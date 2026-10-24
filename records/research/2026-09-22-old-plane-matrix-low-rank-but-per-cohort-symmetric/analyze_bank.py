#!/usr/bin/env python3
"""TAG fable-oldtier-economics (OT-3): offline (numpy, uncharged research instrument) structure of the dumped cube-column bank:
per layer -> column counts, old vs young (2,1)-plane norms, singular-value energy profile of the old plane, per-cohort column
rank profile, per-cohort plane norms and cross-cohort plane cosines (coherence), and symmetric-CP compressibility of each old
cohort's tensor (CP-ALS in the cohort's own frame, ranks K') measured as the relative error of its (2,1) plane on
response rows x live columns.  usage: analyze_bank.py <bank_dir> <layers e.g. 6,9,12,14> [Kprimes e.g. 1,4,16,64,256]"""
import sys, json, time
from pathlib import Path
import numpy as np
D = Path(sys.argv[1]); layers = [int(x) for x in sys.argv[2].split(",")]
Kp_list = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "1,4,16,64,256").split(",")]
rng = np.random.default_rng(0)

def plane(P, w, rows, cols):  # (2,1) slice: K21[i,c] = sum_k w_k P_ik^2 P_ck  on rows x cols
    return (P[rows] ** 2 * w[None, :]) @ P[cols].T

def cp_als(F, w, Kp, iters=60):
    """Symmetric CP fit of sum_k w_k f_k^{x3} (F: r x K) with Kp components; returns factors A,B,C (r x Kp) with weights folded in."""
    r, K = F.shape
    # init: the Kp columns with the largest |w| |f|^3, signed by w
    score = np.abs(w) * np.linalg.norm(F, axis=0) ** 3
    idx = np.argsort(-score)[:Kp]
    A = F[:, idx] * np.cbrt(w[idx])[None, :]
    if Kp > K: A = np.concatenate([A, rng.standard_normal((r, Kp - K)) * 1e-3], axis=1)
    B, C = A.copy(), A.copy()
    def mttkrp(X, Y):  # G_(1) (Y (.) X) for G = sum_k w_k f_k^3 -> r x Kp
        return F @ (w[:, None] * ((F.T @ X) * (F.T @ Y)))
    for it in range(iters):
        A = mttkrp(B, C) @ np.linalg.pinv((B.T @ B) * (C.T @ C))
        B = mttkrp(A, C) @ np.linalg.pinv((A.T @ A) * (C.T @ C))
        C = mttkrp(A, B) @ np.linalg.pinv((A.T @ A) * (B.T @ B))
    return A, B, C

def cp_plane(Q, A, B, C, rows, cols):
    """(2,1) slice of sum_j a_j x b_j x c_j (in unit space via frame Q), symmetrised over which leg is the c-leg."""
    QA, QB, QC = Q @ A, Q @ B, Q @ C
    out = (QA[rows] * QB[rows]) @ QC[cols].T + (QA[rows] * QC[rows]) @ QB[cols].T + (QB[rows] * QC[rows]) @ QA[cols].T
    return out / 3.0

results = {}
for l in layers:
    z = np.load(D / f"layer_{l:02d}.npz"); P = z["pp"].astype(np.float64); w = z["lam"].astype(np.float64)
    oc = int(z["old_columns"]); coh = z["cohorts"]; rows = z["response"]; cols = z["live"]
    n, K = P.shape; t0 = time.time()
    rec = dict(layer=l, n=n, K=K, old_columns=oc, young_columns=K - oc, response_rows=int(len(rows)), live_cols=int(len(cols)),
               cohorts=[(int(b), int(s), int(wd)) for b, (s, wd) in enumerate(coh)])
    K21_old = plane(P[:, :oc], w[:oc], rows, cols) if oc else np.zeros((len(rows), len(cols)))
    K21_young = plane(P[:, oc:], w[oc:], rows, cols)
    K21 = K21_old + K21_young
    nrm = lambda M: float(np.linalg.norm(M))
    rec.update(norm_plane=nrm(K21), norm_old=nrm(K21_old), norm_young=nrm(K21_young),
               cos_old_young=float((K21_old * K21_young).sum() / (nrm(K21_old) * nrm(K21_young) + 1e-30)))
    if oc:
        s = np.linalg.svd(K21_old, compute_uv=False); e = np.cumsum(s ** 2) / (s ** 2).sum()
        rec["old_plane_energy_at_rank"] = {int(k): float(e[k - 1]) for k in (1, 4, 16, 64, 256) if k <= len(e)}
        # per-cohort
        per = []; planes = []
        for b, (st, wd) in enumerate(coh):
            if st + wd > oc or wd == 0: continue
            Pb, wb = P[:, st:st + wd], w[st:st + wd]
            U, S, Vt = np.linalg.svd(Pb, full_matrices=False)
            en = np.cumsum(S ** 2) / (S ** 2).sum()
            Kb = plane(Pb, wb, rows, cols); planes.append(Kb)
            r_eff = int(np.searchsorted(en, 0.999) + 1)
            Q = U[:, :max(r_eff, 1)]; Fb = Q.T @ Pb  # frame coords
            fits = {}
            for Kp in Kp_list:
                A, B, C = cp_als(Fb, wb, Kp)
                Kb_hat = cp_plane(Q, A, B, C, rows, cols)
                fits[Kp] = dict(rel_err_own=float(nrm(Kb - Kb_hat) / (nrm(Kb) + 1e-30)), err_vs_total=float(nrm(Kb - Kb_hat) / (nrm(K21) + 1e-30)))
            per.append(dict(cohort=b, age=l - b, columns=int(wd), rank999=r_eff, col_energy_at=dict((k, float(en[min(k, len(en)) - 1])) for k in (32, 128, 256, 384)),
                            norm_vs_total=float(nrm(Kb) / nrm(K21)), cp=fits))
            print(json.dumps(dict(layer=l, cohort=b, age=l - b, cols=int(wd), rank999=r_eff, norm_vs_total=round(nrm(Kb) / nrm(K21), 4),
                                  cp={k: round(v["rel_err_own"], 4) for k, v in fits.items()})), flush=True)
        rec["per_cohort"] = per
        if len(planes) > 1:
            V = np.stack([p.ravel() for p in planes]); V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-30
            G = V @ V.T; rec["cohort_plane_cosines"] = [[round(float(x), 3) for x in row] for row in G]
    rec["wall"] = round(time.time() - t0, 1)
    results[l] = rec
    print(json.dumps({k: v for k, v in rec.items() if k not in ("per_cohort", "cohorts")}), flush=True)
    json.dump(results, open(D / "analysis.json", "w"), indent=1)
