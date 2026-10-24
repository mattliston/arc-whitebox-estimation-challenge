"""[opus-k4unit] offline test (pre-registered TEAM_LOG [opus-newcomp] 10:44Z / lead GO): on captured S2tr state vs the 2 x 32M per-unit
truths (analyst/fresh/pu/mps_truth_rank*_A32M / B32M).
#1  per-unit L15 k4 from the L14 (2,2) plane: k4p_i = 3 [(WoW)' post22 (WoW)]_ii - 2 sum_a W_ai^4 post_k4_a (variant A), + the K4L final
    terms (variant B); today = radial v^2 + K4L. t = k4_true - today, p = k4p - today. Weights = the mean formula's k4 sensitivity
    ((a^2 - 1) phi(a) / (v s))^2 at the card's (mu, v). Reports weighted R^2 (corr^2), fitted slope, slope-1 energy left, truth-noise share.
#2  L15 variance error e = log(v_card / v_true) on the top-8 mode shares F_ir = lam_r (W_i . u_r)^2 / v_i of the post covariance entering
    L15; weights (sigma phi(a) / 2)^2; in-sample R^2, transfer R^2 (fit on the other networks), coefficient sign agreement.
usage: k4u_analyze_v1.py RANK..."""
import json, sys
import numpy as np
from pathlib import Path
from scipy.stats import norm
D = Path(__file__).resolve().parent; O = D.parents[2]
ranks = [int(x) for x in sys.argv[1:]]
res = {}; X2 = {}


def wls(y, X, w):
    sw = np.sqrt(w); b, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None); r = y - X @ b
    ybar = np.sum(w * y) / np.sum(w); return b, 1 - np.sum(w * r ** 2) / np.sum(w * (y - ybar) ** 2)


for r in ranks:
    c = np.load(D/'cap'/f'cap_rank{r}.npz')
    A = np.load(O/'analyst/fresh/pu'/f'mps_truth_rank{r:04d}_A32M.npz'); B = np.load(O/'analyst/fresh/pu'/f'mps_truth_rank{r:04d}_B32M.npz')
    W = c['L15_W']; mu, v = c['L15_mu'], c['L15_v']; s = np.sqrt(v); a = mu / s; ph = norm.pdf(a)
    kt = (A['mc_pre_k4'][15] + B['mc_pre_k4'][15]) / 2; kn = (A['mc_pre_k4'][15] - B['mc_pre_k4'][15]) / 2
    today = c['L15_k4']; k4l = c['L15_k4l']
    P = c['L14_post22']; WW = W * W
    d = np.einsum('ai,ai->i', WW, P @ WW)
    kpA = 3 * d - 2 * np.einsum('ai,a->i', WW * WW, c['L14_post_k4'])
    out = dict(radial=float(c['L15_radial']), kurt_true_mean=float(np.mean(kt / v ** 2)), kurt_true_sd=float(np.std(kt / v ** 2)),
               kurt_today_sd=float(np.std(today / v ** 2)), k4l_share=float(np.std(k4l) / max(np.std(today), 1e-300)))
    wk = ((a * a - 1) * ph / (v * s)) ** 2
    t = kt - today
    for name, kp in (('A_aabb', kpA), ('B_aabb_plus_K4L', kpA + k4l), ('C_aabb_rescaled_to_radial', kpA * np.sum(wk * today * kpA) / np.sum(wk * kpA * kpA))):
        p = kp - today
        for tag, w in (('w', wk), ('u', np.ones_like(wk))):
            b, R2 = wls(t, np.column_stack([np.ones_like(p), p]), w)
            left = np.sum(w * (t - p) ** 2) / np.sum(w * t ** 2)
            out[f'{name}_{tag}'] = dict(R2=round(float(R2), 3), slope=round(float(b[1]), 3), energy_left_slope1=round(float(left), 3),
                                        noise_share=round(float(np.sum(w * kn ** 2) / np.sum(w * t ** 2)), 3),
                                        kurt_prop_sd=round(float(np.std(kp / v ** 2)), 4))
    # #2: variance error on top-8 mode shares
    vt = (A['mc_pre_v'][15] + B['mc_pre_v'][15]) / 2; vn = 0.5 * np.log(A['mc_pre_v'][15] / B['mc_pre_v'][15])
    e = np.log(v / vt)
    ev, U = np.linalg.eigh(c['L15_C']); ev, U = ev[::-1][:8], U[:, ::-1][:, :8]
    F = (W.T @ U) ** 2 * ev[None, :] / v[:, None]
    wv = (s * ph / 2) ** 2
    b, R2 = wls(e, np.column_stack([np.ones(len(e)), F]), wv)
    X2[r] = (e, F, wv)
    out['v_err'] = dict(log_sd=round(float(np.sqrt(np.average(e ** 2, weights=wv))), 5), noise_sd=round(float(np.sqrt(np.average(vn ** 2, weights=wv))), 5),
                        R2_in=round(float(R2), 3), coef=[round(float(x), 4) for x in b[1:]], top_share_mean=[round(float(x), 4) for x in F.mean(0)])
    res[r] = out
for r in ranks:   # transfer for #2: fit on the other networks, evaluate on r
    others = [q for q in ranks if q != r]
    if not others: continue
    e = np.concatenate([X2[q][0] for q in others]); F = np.concatenate([X2[q][1] for q in others]); w = np.concatenate([X2[q][2] for q in others])
    b, _ = wls(e, np.column_stack([np.ones(len(e)), F]), w)
    er, Fr, wr = X2[r]; pr = np.column_stack([np.ones(len(er)), Fr]) @ b
    ybar = np.sum(wr * er) / np.sum(wr); res[r]['v_err']['R2_transfer'] = round(float(1 - np.sum(wr * (er - pr) ** 2) / np.sum(wr * (er - ybar) ** 2)), 3)
signs = np.sign(np.array([res[r]['v_err']['coef'] for r in ranks]))
summary = dict(ranks=ranks, coef_sign_agreement=[int(abs(x)) == len(ranks) for x in signs.sum(0)] if len(ranks) > 1 else None)
print(json.dumps(dict(per_rank=res, summary=summary), indent=1))
(D/f'K4U_ANALYSIS_{"_".join(map(str, ranks))}_v1.json').write_text(json.dumps(dict(per_rank=res, summary=summary), indent=1))
