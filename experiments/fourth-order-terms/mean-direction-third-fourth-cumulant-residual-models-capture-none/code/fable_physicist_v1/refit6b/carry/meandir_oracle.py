#!/usr/bin/env python3
"""TAG fable-physicist (CR-17): how much of the analyst's per-unit k3+k4 lever (section 34: truth k3/k4 at L12-14 with the L12-14 learned
mean correction at 0.5 = -16.8 / -10.9% on r35 / r100) does a MEAN-DIRECTION cumulant model capture? The per-unit k3/k4 error's
explainable part lies along the post mean direction (= top eigvec; lowrank_kurt.py), whose push is the pre mean mu_a. Model (2 scalars per
layer): k3_a += eta_L mu_a^3, k4_a += zeta_L mu_a^4 at L12-14, with eta_L, zeta_L fitted IN-SAMPLE on the 32M truth (an oracle for the
scalars; the question is whether the SHAPE carries the lever). Arms, all with the learned mean correction delta at 0.5 at L12-14 (the
analyst's pricing rule): control (no change), meandir (fitted scalars), meandir k4-only. B2 ep1 card, learned on, no refit.
usage: meandir_oracle.py <threads>"""
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
LAY = (12, 13, 14)
def run(W, tr, eta=None, zeta=None, dscale=0.5, base=False):
    card = s6bcard.card(torch, W, p0, a0, c0, e0)
    if not base: card.hooks['delta'] = lambda l, d, ctx: d * dscale if l in LAY else d
    if eta is not None or zeta is not None:
        def pin(l, v, ctx):
            if l not in LAY: return v
            mu, Cp, k21, k4, k22, r31 = v
            if eta is not None: k21 = k21.clone(); k21.diagonal().add_(eta[l] * mu ** 3)
            if zeta is not None: k4 = k4 + zeta[l] * mu ** 4
            return (mu, Cp, k21, k4, k22, r31)
        card.hooks['pair_in'] = pin
    return float(((card.run(False).numpy().astype(np.float64)[15] - tr) ** 2).mean())
for rk in (35, 100):
    W = np.load(STORE / Path(man[rk]['path']).parent / 'weights.npy'); Z = np.load(P2 / 'public_g19_attribution_v1' / f'mps_truth_rank{rk:04d}_indep32M.npz')
    tr = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rk:04d}.npz')['mc_post'][15].astype(np.float64)
    card = s6bcard.card(torch, W, p0, a0, c0, e0); cap = {}
    card.hooks['state_out'] = lambda l, o, ctx: (cap.__setitem__(l, (ctx['mu'].numpy().copy(), np.diagonal(ctx['k21'].numpy()).copy(), ctx['k4'].numpy().copy())) if l in LAY else None, o)[1]
    base = float(((card.run(False).numpy().astype(np.float64)[15] - tr) ** 2).mean())
    eta, zeta, fits = {}, {}, {}
    for L in LAY:
        mu, k3, k4 = cap[L]; t3, t4 = (np.asarray(Z[k][L], dtype=np.float64) for k in ('mc_pre_k3', 'mc_pre_k4'))
        f3, f4 = mu ** 3, mu ** 4
        eta[L] = float((t3 - k3) @ f3 / (f3 @ f3)); zeta[L] = float((t4 - k4) @ f4 / (f4 @ f4))
        fits[L] = dict(eta=eta[L], zeta=zeta[L], k3_err_before=float(np.linalg.norm(t3 - k3) / np.linalg.norm(t3)), k3_err_after=float(np.linalg.norm(t3 - k3 - eta[L] * f3) / np.linalg.norm(t3)),
                       k4_err_before=float(np.linalg.norm(t4 - k4) / np.linalg.norm(t4)), k4_err_after=float(np.linalg.norm(t4 - k4 - zeta[L] * f4) / np.linalg.norm(t4)))
    ctrl = run(W, tr); md = run(W, tr, eta, zeta); md4 = run(W, tr, None, zeta)
    print(json.dumps(dict(net=rk, base=base, control_dhalf=round(ctrl / base - 1, 4), meandir_k3k4=round(md / base - 1, 4), meandir_k4only=round(md4 / base - 1, 4),
                          fits={L: {k: round(v, 4) if 'err' in k else v for k, v in f.items()} for L, f in fits.items()})), flush=True)
