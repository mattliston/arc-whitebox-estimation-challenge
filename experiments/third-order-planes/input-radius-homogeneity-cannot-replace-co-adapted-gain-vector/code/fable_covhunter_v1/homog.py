#!/usr/bin/env python3
"""TAG fable-covhunter (CH-38): HOMOGENEITY CHECK on the torch S6b card (fable_analyst_v1/s6bcard.py, learned ON; lead's request).
For a bias-free ReLU net and x ~ N(0, I_n), z_l = r g_l(xhat) with r = |x| independent of xhat, so kappa3(z_l) contains the exact
input-radial term c3 Sym(C_l (x) mu_l), c3 = 1/E[r]^2 = 9.7704e-4 (n = 1024; certified in cert_homog.py). The card's K21 b terms
v_a b_c + 2 C_ac b_a are that form; its b recursion reproduces b = c3 * mean exactly iff radial = 6 c3 = 5.862e-3.
Per layer 2-14:
 (1) the card's radial (entering the layer; it sets the b source radial d v / 6, k4 = radial v^2 and K22) and new radial, vs 6 c3;
 (2) the card's b_pre = W^T b against c3 mu_pre (card mu and true mu): cosine and scale (b.mu)/(c3 |mu|^2);
 (3) least-squares fit of the TRUE K21 (8M MC planes, off-diagonal) as Sym(C_true (x) b) + rest: energy share explained, and b_fit
     against c3 mu_true and the card's b_pre (cosine, scale); plus the share explained by the pure input-radial c3 Sym(C_true (x) mu_true).
usage: homog.py <rank> <threads>"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
for _p in (str(HERE), str(P2 / 'fable_analyst_v1')):
    if _p not in sys.path: sys.path.insert(0, _p)
import numpy as np, torch
import s6bcard as s6
rank = int(sys.argv[1]); torch.set_num_threads(int(sys.argv[2]))
C3 = 9.770395e-04; n = 1024; off = ~np.eye(n, dtype=bool)
mc = np.load(HERE / f'mcall_rank{rank:04d}_P32M.npz'); PL = np.load(P2 / 'public_g19_attribution_v1' / f'mps_planes_rank{rank:04d}.npz')
pl = s6.payload(torch); w = s6.weights(rank); cd = s6.card(torch, w, *pl)
cap = {}
base_cov = cd.hooks['cov_pm']
def cov(l, C, ctx):
    cap.setdefault(l, {}).update(radial_in=float(ctx['radial']), bpre=ctx['bpre'].numpy().copy(), mu=ctx['mu'].numpy().copy(),
                                 Cpre=ctx['Cpre'].numpy().copy() if 2 <= l <= 14 else None)
    return base_cov(l, C, ctx)
cd.hooks['cov_pm'] = cov
cd.hooks['state_out'] = lambda l, st, ctx: (cap.setdefault(l, {}).update(radial_out=float(st[7]), b=st[5].numpy().copy(), m=st[0].numpy().copy()) or st)
with torch.no_grad(): cd.run_states()
print(f'rank {rank}: 6 c3 = {6 * C3:.5f}')
print('(1) radial entering layer l (b source, k4, K22) and new radial, ratio to 6 c3:')
print('    ' + ' '.join(f"L{l}:{cap[l]['radial_in']:.5f}({cap[l]['radial_in'] / (6 * C3):.1f}x)" for l in range(1, 15)))
res = {}
cosv = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
for l in range(2, 15):
    c = cap[l]; bpre, mu = c['bpre'], c['mu']; mu_t = mc[f'mean_pre_{l}']; Ct = mc[f'cov_pre_{l}']; vt = np.diag(Ct)
    r = dict(radial_in=c['radial_in'], radial_ratio=c['radial_in'] / (6 * C3),
             b_cos_mu=cosv(bpre, mu), b_scale_mu=float(bpre @ mu / (C3 * mu @ mu)), b_cos_mut=cosv(bpre, mu_t), b_scale_mut=float(bpre @ mu_t / (C3 * mu_t @ mu_t)))
    # (3) LSQ: K21t_ac ~ v_a b_c + 2 C_ac b_a on the off-diagonal; normal matrix A_jk = d_jk (|v|^2 - v_j^2 + 4 sum_c C_jc^2 - 4 v_j^2 ... )
    K = np.asarray(PL[f'K21_{l}'], dtype=np.float64); Co = Ct.copy(); np.fill_diagonal(Co, 0.)
    V = np.broadcast_to(vt[:, None], (n, n)).copy(); np.fill_diagonal(V, 0.)          # V_ac = v_a off-diagonal
    Ko = K.copy(); np.fill_diagonal(Ko, 0.)
    # model M(b) = V diag(b) (column scaling: V_ac b_c) + 2 Co * b[:, None]; normal equations built exactly
    A = np.diag((V * V).sum(0) + 4 * (Co * Co).sum(1)) + 2 * (V * Co).T + 2 * (V * Co)   # A_jk = sum over (a,c) of dM/db_j dM/db_k
    y = (V * Ko).sum(0) + 2 * (Co * Ko).sum(1)
    bfit = np.linalg.solve(A, y)
    M = lambda b: V * b[None, :] + 2 * Co * b[:, None]
    e2 = float((Ko ** 2).sum())
    r.update(fit_share=1 - float(((Ko - M(bfit)) ** 2).sum()) / e2, radial_term_share=1 - float(((Ko - M(C3 * mu_t)) ** 2).sum()) / e2,
             bfit_cos_mut=cosv(bfit, mu_t), bfit_scale_mut=float(bfit @ mu_t / (C3 * mu_t @ mu_t)), bfit_cos_card=cosv(bfit, bpre),
             bfit_over_card=float(bfit @ bpre / (bpre @ bpre)))
    res[l] = r
    print(f"L{l:02d} radial {r['radial_in']:.5f} ({r['radial_ratio']:.1f}x) | card b_pre vs c3 mu: cos {r['b_cos_mu']:.3f} scale {r['b_scale_mu']:.2f} (vs true mu cos {r['b_cos_mut']:.3f} scale {r['b_scale_mut']:.2f})"
          f" | TRUE K21 = Sym(C x b_fit): share {r['fit_share']:.3f}, b_fit vs c3 mu_true cos {r['bfit_cos_mut']:.3f} scale {r['bfit_scale_mut']:.2f}, vs card b cos {r['bfit_cos_card']:.3f} ratio {r['bfit_over_card']:.2f}"
          f" | pure c3 Sym(C x mu) share {r['radial_term_share']:+.3f}", flush=True)
json.dump(res, (HERE / f'homog_rank{rank:04d}.json').open('w'), indent=1)
