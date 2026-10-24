#!/usr/bin/env python3
"""TAG opus-reexam E14: does the fourth cumulant BORN FROM THE THIRD CUMULANT at one layer explain the network-specific part of the
next layer's fourth-order planes and per-unit kurtosis?
At a ReLU layer with pre-activation covariance C and third cumulant T (the card's bank columns U_r, weights lam_r), the leading
connected fourth cumulant of the post-activations for distinct units is one third-cumulant insertion plus one covariance edge on a
kinked unit ("hub" h, pendant p):   Q[a,b,c,d] = sum over (h, p)  d_h C_hp g_p  T[the other three, with gates on the non-hub legs].
Pushed through the next weights W' (U' = W' (g o U), K = d C g off-diagonal, KW = K W', theta[i,r] = sum_a W'_ai KW_ai U_ar,
G[i,a] = sum_r lam_r U'_ir^2 U_ar):
   K31_T[i,j] = 3 [(G o W'^T) KW] + 6 [(theta o U') lam U'^T] + 3 [(G o KW^T) W']         (kappa(z'_i,z'_i,z'_i,z'_j))
   k4_T[i]    = 12 sum_r lam_r theta[i,r] U'_ir^2
Tested against Monte Carlo truth of the next layer (8M-sample planes, two independent draws where they exist; 32M per-unit k4):
on kink rows (|mu/sigma| < 1.5), off-diagonal, the true K31 is regressed on the isotropic form v_i C_ij alone and together with K31_T;
coefficients are fitted on one draw and scored on the other. The same for the per-unit k4 (isotropic form v_i^2).
The card's bank (closed loop, learned parts on) supplies T, so its own 6-8% plane error is included in the test.
usage: e14_tborn_fourth.py <rank> <threads> [layers l, default 4,8,11]"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'physicist'))
import numpy as np, torch
import pcard as pc
rank = int(sys.argv[1]); pc.set_threads(sys.argv[2])
LAYERS = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [4, 8, 11]
cd = pc.build(rank); cap = {}
def grab(l, st, ctx):
    if l in LAYERS:
        aux = ctx['aux']; stops = st[12]; n_old = stops[-2] if len(stops) >= 2 else 0
        cap[l] = dict(p=st[2][:, :n_old].clone(), lam=st[3][:n_old].clone(), alive=st[6].clone(), g=aux['gate'].clone(),
                      d=(aux['pdf'] / aux['sigma']).clone(), C=ctx['Cpre'].clone())
    return st
pc.wrap(cd, 'state_out', grab)
pred = pc.run(cd); tp = pc.truth(rank); print(json.dumps(dict(rank=rank, final_x1e9=round(float(pc.mse(pred, tp)[15]) * 1e9, 3))), flush=True)
MC = pc.mcall(rank); PL = pc.planes(rank); PU = pc.pu(rank); PUi = pc.pu(rank, 'indep32M')
try: PL2 = pc.planes(rank, '_noise2')
except FileNotFoundError: PL2 = None
eye = np.eye(1024, dtype=bool)
for l in LAYERS:
    t0 = time.time(); c = cap[l]; L = l + 1
    g = c['g'].double(); d = c['d'].double(); C = c['C'].double(); P = c['p'].double(); lam = c['lam'].double(); alive = c['alive'].double()
    U = torch.where((g > 1e-6)[:, None], P / g.clamp_min(1e-6)[:, None], torch.zeros_like(P)) * alive[:, None]
    Wn = torch.as_tensor(np.asarray(cd.w[L], dtype=np.float64)) if not torch.is_tensor(cd.w[L]) else cd.w[L].double()
    Up = Wn.T @ P
    K = d[:, None] * C * g[None, :]; K = K - torch.diag(torch.diagonal(K))
    KW = K @ Wn
    theta = (Wn * KW).T @ U
    G = (Up * Up * lam[None, :]) @ U.T
    K31T = (3 * (G * Wn.T) @ KW + 6 * (theta * Up * lam[None, :]) @ Up.T + 3 * (G * KW.T) @ Wn).numpy()
    k4T = (12 * (theta * Up * Up * lam[None, :]).sum(1)).numpy()
    # truth of layer L
    mu = MC[f'mean_pre_{L}'].astype(np.float64); Cp = MC[f'cov_pre_{L}'].astype(np.float64); v = np.diag(Cp).copy(); a = mu / np.sqrt(v)
    kink = np.abs(a) < 1.5
    iso = v[:, None] * Cp
    def score(Kt, Kt2):
        """rows kink, off-diagonal. returns orientation check, R2 iso / iso+T (fit on Kt, scored on Kt2), coefficient of the T term."""
        m = kink[:, None] & ~eye
        y, y2 = Kt[m], Kt2[m]; x1 = iso[m]; x2 = K31T[m]
        ori = (np.corrcoef(Kt[m], iso[m])[0, 1], np.corrcoef(Kt.T[m], iso[m])[0, 1])
        A1 = x1[:, None]; A2 = np.stack([x1, x2], 1)
        c1 = np.linalg.lstsq(A1, y, rcond=None)[0]; c2 = np.linalg.lstsq(A2, y, rcond=None)[0]
        r = lambda A, cf: 1 - ((y2 - A @ cf) ** 2).sum() / (y2 ** 2).sum()
        res = y - A1 @ c1; res2 = y2 - A1 @ c1
        return dict(orientation_corr=[round(float(x), 3) for x in ori], coef_iso=float(c1[0]), coef_iso_T=[float(x) for x in c2],
                    R2_iso=float(r(A1, c1)), R2_iso_plus_T=float(r(A2, c2)),
                    corr_residual_T=float(np.corrcoef(res, x2)[0, 1]), noise_free_residual_share=float((res * res2).sum() / np.sqrt((res ** 2).sum() * (res2 ** 2).sum())),
                    rel_size_T=float(np.sqrt((x2 ** 2).sum() / (y ** 2).sum())), rel_size_residual=float(np.sqrt((res ** 2).sum() / (y ** 2).sum())))
    Kt = PL[f'K31_{L}'].astype(np.float64)
    Kt2 = PL2[f'K31_{L}'].astype(np.float64) if (PL2 is not None and f'K31_{L}' in PL2.files) else Kt
    out = dict(layer_l=l, next_layer=L, columns=int(P.shape[1]), kink_rows=int(kink.sum()), independent_second_draw=bool(Kt2 is not Kt), K31=score(Kt, Kt2))
    # per-unit kurtosis of layer L
    k4a = PU['mc_pre_k4'][L].astype(np.float64); k4b = PUi['mc_pre_k4'][L].astype(np.float64); va = PU['mc_pre_v'][L].astype(np.float64)
    x1 = va ** 2; c1 = (x1 * k4a).sum() / (x1 * x1).sum(); ra = k4a - c1 * x1; rb = k4b - c1 * x1
    cT = (k4T * ra).sum() / (k4T * k4T).sum()
    out['k4'] = dict(radial_fit=float(c1), residual_rel=float(np.sqrt((ra ** 2).sum() / (k4a ** 2).sum())), two_draw_corr_of_residual=float(np.corrcoef(ra, rb)[0, 1]),
                     corr_residual_T=float(np.corrcoef(ra, k4T)[0, 1]), corr_residualB_T=float(np.corrcoef(rb, k4T)[0, 1]), coef_T=float(cT),
                     share_of_residual_explained_on_draw_B=float(1 - ((rb - cT * k4T) ** 2).sum() / (rb ** 2).sum()),
                     rel_size_T=float(np.sqrt((k4T ** 2).sum() / (ra ** 2).sum())),
                     kink_only_corr=float(np.corrcoef(ra[kink], k4T[kink])[0, 1]))
    out['wall'] = round(time.time() - t0, 1)
    print(json.dumps(out), flush=True)
