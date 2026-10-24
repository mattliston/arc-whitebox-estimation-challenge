#!/usr/bin/env python3
"""TAG fable-mc (MC-4): does the exact one-step Gaussian-born STAR term explain the true (3,1) plane? From the MC-true pre-activation
state at layer l-1 (mean, full covariance from mc2), build K31_star at layer l and regress the MC-true K31(l) off-diagonal on
[iso = v_a C_ac (the card's form, with the true C at l), star]; report R2 of iso alone, star alone, both, and the noise floor from the
independent 8M draw (mps_planes_rank0035_noise2 where available). usage: star_k31_truthcheck.py <rank> <prev_layers e.g. 4,7,10,13>"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import norm
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
rank = int(sys.argv[1]); prevs = [int(x) for x in sys.argv[2].split(',')]
mc2 = np.load(HERE / f"mc2_rank{rank:04d}_P32M.npz"); planes = np.load(P2 / "public_g19_attribution_v1" / f"mps_planes_rank{rank:04d}.npz")
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy").astype(np.float64)
off = ~np.eye(1024, dtype=bool)
for lp in prevs:
    l = lp + 1
    if f"K31_{l}" not in planes.files: continue
    mu = mc2[f"mean_pre_{lp}"]; C = mc2[f"cov_pre_{lp}"]; v = np.diag(C); s = np.sqrt(v); al = mu / s
    g = norm.cdf(al); d3 = -al * norm.pdf(al) / v
    bp = g[:, None] * C; np.fill_diagonal(bp, 0.0)
    Wl = W[l]; B = Wl.T @ bp; Wt = Wl.T
    star = 3 * ((Wt * B * B) * d3[None, :]) @ B.T + ((B * B * B) * d3[None, :]) @ Wt.T
    Cl = Wl.T @ mc2[f"cov_post_{lp}"] @ Wl   # true pre covariance at l
    vl = np.diag(Cl); iso = vl[:, None] * Cl
    y = planes[f"K31_{l}"].astype(np.float64)[off]
    def r2(cols):
        X = np.stack([c[off] for c in cols], 1); b, *_ = np.linalg.lstsq(X, y, rcond=None); return 1 - np.sum((y - X @ b) ** 2) / np.sum((y - y.mean()) ** 2), b
    r_iso, b_iso = r2([iso]); r_star, b_star = r2([star]); r_both, b_both = r2([iso, star])
    print(json.dumps(dict(layer=l, r2_iso=round(float(r_iso), 4), r2_star=round(float(r_star), 4), r2_both=round(float(r_both), 4), coef_both=[float(x) for x in b_both],
                          star_rms_over_truth_rms=round(float(np.sqrt(np.mean(star[off] ** 2)) / np.sqrt(np.mean(y ** 2))), 4))), flush=True)
