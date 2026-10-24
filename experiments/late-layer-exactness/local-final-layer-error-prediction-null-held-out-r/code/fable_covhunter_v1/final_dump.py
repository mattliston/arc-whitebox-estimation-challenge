#!/usr/bin/env python3
"""TAG fable-covhunter (CH-27): FINAL-LAYER STEP 0 data. On the S2 torch card with the v3 covariance knobs (= the fired S4 physics;
gn3_knobs.configure), record per final-layer unit i: the error e_i = pred15_i - truth15_i (pred clamped at 0 as scored) and card-only
features: mu, v, sigma, alpha, gate, pdf of z_15; k3_15 and its parts (bank columns, star diagonal, b term); k4_15 (= radial v^2);
the scalar map's host mean, gaussian mean, post var/k3/k4; the final GNN hook output delta; the unit's weight moments sum w^k;
diag vs off-diagonal share of v_i = w_i^T C_14 w_i; loadings of w_i on the top-16/64 eigenvectors of the card's C_14 (energy share of w_i
and variance share of v_i); the kink indicator. One npz per net in final_dump/.
usage: final_dump.py <ranks> <threads>"""
import sys, json, time, math
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
import covstep as cs
import gn3_knobs as g3
ranks = [int(x) for x in sys.argv[1].split(',')]; torch.set_num_threads(int(sys.argv[2]))
OUT = HERE / 'final_dump'; OUT.mkdir(exist_ok=True)
pl = cs.model.torch_payload(torch, np, False)
for rk in ranks:
    dest = OUT / f'final_rank{rk:04d}.npz'
    if dest.exists(): continue
    t0 = time.time(); w = cs.weights(rk); truth = cs.truthpost(rk)
    card = cs.HookCard(torch, w, *pl); g3.configure(card, g3.V3)
    cap = {}
    card.hooks['state_in'] = lambda l, st, ctx: (cap.update(C14=st[1].detach().clone(), m14=st[0].detach().clone(), radial=float(st[7])) if l == 15 else None) or st
    def k3h(l, k3, ctx):
        cap.update(k3_bank=ctx['bank'].detach().clone(), k3_star=ctx['star'].detach().clone(), k3_bterm=ctx['bterm'].detach().clone())
        return k3
    card.hooks['k3_15'] = k3h
    def dh(l, delta, ctx):
        if l == 15:
            cap.update(host=ctx['host'].detach().clone(), mu=ctx['mu'].detach().clone(), vv=ctx['vv'].detach().clone(),
                       delta=delta.detach().to(torch.float64).clone(), **{f'aux_{k}': v.detach().clone() for k, v in ctx['aux'].items() if hasattr(v, 'shape') and v.shape == (1024,)})
        return delta
    card.hooks['delta'] = dh
    with torch.no_grad(): outs, _ = card.run_states()
    pred = outs[15].clamp_min(0).to(torch.float32).numpy().astype(np.float64)
    e = pred - truth[15]
    W = w[15].astype(np.float64); C14 = cap['C14'].numpy()
    lam_, U = np.linalg.eigh(C14); order = np.argsort(lam_)[::-1]; lam_, U = lam_[order], U[:, order]
    Y = U[:, :64].T @ W                                    # (64, n) loadings of each unit's weight vector
    v = np.sum(W * (C14 @ W), axis=0); d = np.sum(W * W * np.diag(C14)[:, None], axis=0)
    wn2 = np.sum(W * W, axis=0)
    feats = dict(e=e, truth=truth[15], pred=pred, mu=cap['mu'].numpy(), vv=cap['vv'].numpy(), v_check=v, host=cap['host'].numpy(),
                 delta=cap['delta'].numpy(), k3_bank=cap['k3_bank'].numpy(), k3_star=cap['k3_star'].numpy(), k3_bterm=cap['k3_bterm'].numpy(),
                 radial=np.array(cap['radial']), diag_share=d / v,
                 wload16=np.sum(Y[:16] ** 2, 0) / wn2, wload64=np.sum(Y[:64] ** 2, 0) / wn2,
                 vload16=np.sum(lam_[:16, None] * Y[:16] ** 2, 0) / v, vload64=np.sum(lam_[:64, None] * Y[:64] ** 2, 0) / v,
                 wsum1=W.sum(0), wsum2=(W ** 2).sum(0), wsum3=(W ** 3).sum(0), wsum4=(W ** 4).sum(0),
                 m14_mean=np.array(float(cap['m14'].mean())), C14_eig=lam_[:64])
    for k, val in cap.items():
        if k.startswith('aux_'): feats[k] = val.numpy()
    np.savez(dest, **feats)
    print(json.dumps(dict(rank=rk, mse=float((e ** 2).mean()), v_check=float(np.max(np.abs(v / feats['vv'] - 1))), wall=round(time.time() - t0, 1))), flush=True)
