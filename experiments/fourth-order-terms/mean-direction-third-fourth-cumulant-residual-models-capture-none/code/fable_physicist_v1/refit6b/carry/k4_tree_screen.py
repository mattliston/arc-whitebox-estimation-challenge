#!/usr/bin/env python3
"""TAG fable-physicist (CR-18): SCREEN (not a certified build) for a physical per-unit k4 source. The Gaussian-born fourth cumulant of the
post activations at L-1 (distinct units) is led by the connected 3-bond trees: STAR (center a with f''', leaves f'): e_a (g_i C_ai)(g_j C_aj)
(g_k C_ak); PATH (f' f'' f'' f'): g_i d_j d_k g_l C_ij C_jk C_kl, with C the PRE covariance at L-1, g = Phi, d = phi/s, e = -a phi/s^2.
Pushed to unit b at L: star_b = sum_a W_ab e_a S_ab^3 and path_b = (T^T C T)_bb, with S = C (g * W_L) and T = W_L * d * S (elementwise rows).
Screen: least squares of the card's per-unit k4 residual (32M truth - card) at L12-14 on [star_b, path_b] (+ the card's own radial vv^2 as a
nuisance scale) -> R^2, against random-feature controls. Factors of the diagrams are absorbed by the fit (a build needs a certificate).
usage: k4_tree_screen.py <threads>"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; RF = HERE.parent; P2 = HERE.parents[2]
sys.path.insert(0, str(P2 / 'fable_analyst_v1')); sys.path.insert(1, str(P2 / 'fable_covhunter_v1'))
import numpy as np, torch
import s6bcard
torch.set_num_threads(int(sys.argv[1])); torch.set_grad_enabled(False)
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
p0, a0, c0, e0 = s6bcard.payload(torch); sv = torch.load(RF / 's6b_B2_fullnet_lr1e4_trunk5e5/params_epoch1.pt')
for k in p0: p0[k].copy_(sv['p'][k])
a0.copy_(sv['a']); c0.copy_(sv['c'])
for k in e0: e0[k].copy_(sv['e'][k])
rng = np.random.default_rng(1)
def r2(X, y):
    X = np.column_stack([np.ones(len(y)), X]); coef, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ coef
    return 1 - float(res @ res) / float((y - y.mean()) @ (y - y.mean())), coef
print('net L | k4 resid rel | R^2: star / path / star+path | +vv^2 | random 2-feature control | corr(star,resid) corr(path,resid)')
for rk in (35, 100):
    W = np.load(STORE / Path(man[rk]['path']).parent / 'weights.npy'); Z = np.load(P2 / 'public_g19_attribution_v1' / f'mps_truth_rank{rk:04d}_indep32M.npz')
    card = s6bcard.card(torch, W, p0, a0, c0, e0); rec = {}
    def grab(l, o, ctx):
        if l < 15:
            aux = ctx['aux']; rec[l] = dict(Cpre=ctx['Cpre'].numpy().copy(), g=aux['gate'].numpy().copy(), s=aux['sigma'].numpy().copy(),
                                          pdf=aux['pdf'].numpy().copy(), a=aux['alpha'].numpy().copy(), alive=ctx['alive'].numpy().copy(),
                                          k4=ctx['k4'].numpy().copy(), vv=np.diagonal(ctx['Cpre'].numpy()).copy())
        return o
    card.hooks['state_out'] = grab; card.run(False)
    for L in (12, 13, 14):
        r_, rL = rec[L - 1], rec[L]; Wl = W[L].astype(np.float64); C = r_['Cpre']; al = r_['alive']
        g, d, e = al * r_['g'], al * r_['pdf'] / r_['s'], al * (-r_['a'] * r_['pdf'] / r_['s'] ** 2)
        S = C @ (g[:, None] * Wl)                        # S_ab = sum_i C_ai g_i W_ib
        star = ((Wl * e[:, None]) * S ** 3).sum(0)       # sum_a W_ab e_a S_ab^3
        T = Wl * d[:, None] * S                          # T_jb = W_jb d_j S_jb
        path = ((C @ T) * T).sum(0)                      # (T^T C T)_bb
        t4 = np.asarray(Z['mc_pre_k4'][L], dtype=np.float64); res4 = t4 - rL['k4']; vv = rL['vv']
        r_s, _ = r2(star[:, None], res4); r_p, _ = r2(path[:, None], res4); r_sp, _ = r2(np.column_stack([star, path]), res4)
        r_spv, _ = r2(np.column_stack([star, path, vv ** 2]), res4)
        rc = np.mean([r2(rng.standard_normal((1024, 2)), res4)[0] for _ in range(5)])
        print(f'{rk:>3} {L:>2} | {np.linalg.norm(res4) / np.linalg.norm(t4):.3f} | {r_s:.3f} / {r_p:.3f} / {r_sp:.3f} | {r_spv:.3f} | {rc:.3f} | '
              f'{np.corrcoef(star, res4)[0, 1]:+.3f} {np.corrcoef(path, res4)[0, 1]:+.3f}')
