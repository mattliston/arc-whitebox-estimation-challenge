#!/usr/bin/env python3
"""TAG fable-physicist (CR-16): is the card's per-unit k3/k4 ERROR low-rank in pushed directions? Hypothesis: post activations share a few
non-Gaussian common factors f_r along directions u_r, so x_a = sum_r beta_ar f_r + (near-Gaussian rest) with beta_ar = (W^T u_r)_a and
the per-unit residuals are  k4_true - k4_card ~ sum_r c4_r beta_ar^4 (+ scale) and k3_true - k3_card ~ sum_r c3_r beta_ar^3.
Directions: the top-R eigenvectors of the card's post covariance at L-1 and the card's post mean direction. In-sample least squares per
(net, layer) with the card's own vv^2 (k4) / 3 vv bpre-like scale terms as nuisance; reported: the share of the residual energy explained
(R^2) against the same fit on RANDOM directions (control). Truth: 32M per-unit k3/k4 (r35, r100). B2 ep1 card, learned on.
usage: lowrank_kurt.py <threads>"""
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
rng = np.random.default_rng(0)
def r2(X, y):
    X = np.column_stack([np.ones(len(y)), X]); coef, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ coef
    return 1 - float(res @ res) / float((y - y.mean()) @ (y - y.mean()))
print('net L | resid rel (k3, k4) | R^2 k3: top-R eig (R=4/16) + mean dir | random dirs (R=16) | R^2 k4: top eig (4/16) + mean | random (16)')
out = {}
for rk in (35, 100):
    W = np.load(STORE / Path(man[rk]['path']).parent / 'weights.npy'); Z = np.load(P2 / 'public_g19_attribution_v1' / f'mps_truth_rank{rk:04d}_indep32M.npz')
    card = s6bcard.card(torch, W, p0, a0, c0, e0); rec = {}
    def grab(l, o, ctx):
        rec[l] = dict(m=o[0].numpy().copy(), C=o[1].numpy().copy(), k3=np.diagonal(ctx['k21'].numpy()).copy(), k4=ctx['k4'].numpy().copy(),
                      vv=np.diagonal(ctx['Cpre'].numpy()).copy()) if l < 15 else None
        return o
    card.hooks['state_out'] = grab; card.run(False); out[rk] = {}
    for L in (10, 12, 14):
        r_prev, r = rec[L - 1], rec[L]; Wl = W[L].astype(np.float64)
        ev, U = np.linalg.eigh((r_prev['C'] + r_prev['C'].T) / 2); U = U[:, ::-1]
        md = r_prev['m'] / np.linalg.norm(r_prev['m'])
        t3, t4 = np.asarray(Z['mc_pre_k3'][L], dtype=np.float64), np.asarray(Z['mc_pre_k4'][L], dtype=np.float64)
        d3, d4 = t3 - r['k3'], t4 - r['k4']; vv = r['vv']
        res = {'k3_rel': float(np.linalg.norm(d3) / np.linalg.norm(t3)), 'k4_rel': float(np.linalg.norm(d4) / np.linalg.norm(t4))}
        for R in (4, 16):
            B = Wl.T @ np.column_stack([U[:, :R], md])           # pushed directions (n x R+1)
            res[f'k3_eig{R}'] = r2(np.column_stack([B ** 3, vv ** 1.5]), d3)
            res[f'k4_eig{R}'] = r2(np.column_stack([B ** 4, vv ** 2]), d4)
        Rn = Wl.T @ np.linalg.qr(rng.standard_normal((1024, 17)))[0]
        res['k3_rand16'] = r2(np.column_stack([Rn ** 3, vv ** 1.5]), d3); res['k4_rand16'] = r2(np.column_stack([Rn ** 4, vv ** 2]), d4)
        out[rk][L] = res
        print(f'{rk:>3} {L:>2} | {res["k3_rel"]:.3f} {res["k4_rel"]:.3f} | {res["k3_eig4"]:.3f} / {res["k3_eig16"]:.3f} | {res["k3_rand16"]:.3f} | '
              f'{res["k4_eig4"]:.3f} / {res["k4_eig16"]:.3f} | {res["k4_rand16"]:.3f}')
(HERE / 'lowrank_kurt.json').write_text(json.dumps(out))
