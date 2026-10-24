#!/usr/bin/env python3
"""TAG fable-physicist (RF6-9): candidate 4 (lead), a Tucker carrier with EXACT pinned legs, accuracy emulation, no refit.
A born pair is lam[(s + gam e)^3 - (s - gam e)^3] with s = st + u e (st = the covariance leg, e = the born unit's own direction, pushed
together by the same linear maps), i.e. lam[6 gam Sym(st,st,e) + 12 gam u Sym(st,e,e) + (6 gam u^2 + 2 gam^3) e^3]. At birth the hook
records rho = u / gam per pair (the born unit a is where p+ and p- differ; plus = p+[a], minus = p-[a]). At every layer >= L0 it takes
EVERY young pair (all cohorts not yet in the old frame tier, newborn included), splits it into e' = (p+ - p-)/2 (the exact pinned leg)
and st' = (p+ + p-)/2 - rho e' (the covariance leg), projects all st' legs onto ONE common top-r eigenspace of the weighted Gram
sum_k (|lam_k| |e'_k|)^2 st'_k st'_k^T, and rebuilds p+- = P st' + rho e' +- e'. The pinned legs and the unit-bond parts stay exact, which is the
lead's "realistic version"; the old frame tier is left as is. Logs the weighted st'-energy the frame captures per layer (the check that
one frame can hold every cohort's covariance legs after the pushes). r >= the pair count reproduces the card exactly (built-in control).
usage: tucker_eval.py <r> <L0> <nvalid> <threads> <tag>"""
import sys, json, math, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parents[1]; sys.path.insert(0, str(P2 / 'fable_analyst_v1'))
import numpy as np, torch
import s6bcard
R, L0, nvalid, threads, tag = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
torch.set_num_threads(threads); torch.set_grad_enabled(False); t = torch
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
inputs = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
VALID = inputs['valid'][:nvalid]
p0, a0, c0, e0 = s6bcard.payload(torch)
sv = torch.load(HERE / 's6b_B2_fullnet_lr1e4_trunk5e5/params_epoch1.pt')
for k in p0: p0[k].copy_(sv['p'][k])
a0.copy_(sv['a']); c0.copy_(sv['c'])
for k in e0: e0[k].copy_(sv['e'][k])
def cohorts(stops):
    out, first = [], 0
    for s in stops:
        s = int(s); out.append((first, s)); first = s
    return out
def make_hook(rho, log):
    def hook(layer, out, ctx):
        p, lam, F, stops = out[2], out[3], out[10], out[12]
        a, b = cohorts(stops)[-1]; nsel = (b - a) // 2                             # record rho for the newborn pairs
        if nsel:
            Pp, Pm = p[:, a:a + nsel].to(t.float64), p[:, a + nsel:b].to(t.float64)
            idx = (Pp - Pm).abs().argmax(0); plus = Pp[idx, t.arange(nsel)]; minus = Pm[idx, t.arange(nsel)]
            gap = plus - minus; ok = gap.abs() > 1e-30
            ratio = t.where(ok, (plus + minus) / t.where(ok, gap, t.ones_like(gap)), t.zeros_like(gap))   # a dead pair (p+ = p-) carries nothing
            for j, v in enumerate(ratio.tolist()): rho[a + j] = v
        if layer < L0:
            return out
        n_old = int(F.shape[1]) if F.dim() == 2 else 0
        ip, im = [], []
        for (s0, s1) in cohorts(stops):
            if s0 < n_old: continue
            k = (s1 - s0) // 2
            ip += list(range(s0, s0 + k)); im += list(range(s0 + k, s1))
        if not ip:
            return out
        ip_t, im_t = t.tensor(ip), t.tensor(im)
        Pp, Pm = p[:, ip_t].to(t.float64), p[:, im_t].to(t.float64)
        rr = t.tensor([rho[i] for i in ip], dtype=t.float64)
        ep = (Pp - Pm) / 2; st = (Pp + Pm) / 2 - rr[None, :] * ep
        w = lam[ip_t].to(t.float64).abs() * ep.norm(dim=0)
        X = st * w[None, :]; G = X @ X.T                                            # weighted Gram of the covariance legs (n x n)
        ev, V = t.linalg.eigh((G + G.T) / 2); ev, V = ev.flip(0).clamp_min(0), V.flip(1)
        r = min(R, V.shape[1]); Ur = V[:, :r]
        cap = float(ev[:r].sum() / ev.sum().clamp_min(1e-300))
        stp = Ur @ (Ur.T @ st)
        p = p.clone()
        p[:, ip_t] = (stp + rr[None, :] * ep + ep).to(p.dtype); p[:, im_t] = (stp + rr[None, :] * ep - ep).to(p.dtype)
        log[layer] = dict(pairs=len(ip), rank=r, captured=round(cap, 5))
        return (out[0], out[1], p) + tuple(out[3:])
    return hook
t0 = time.time(); res = {}; logs = {}
for rk in VALID:
    W = np.load(STORE / Path(man[rk]['path']).parent / 'weights.npy')
    truth = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rk:04d}.npz')['mc_post'][15].astype(np.float64)
    card = s6bcard.card(torch, W, p0, a0, c0, e0)
    rho = {}; log = {}
    card.hooks['state_out'] = make_hook(rho, log)
    pred = card.run(False).numpy().astype(np.float64)
    res[str(rk)] = float(((pred[15] - truth) ** 2).mean()); logs[str(rk)] = log
out = dict(r=R, L0=L0, mean=float(np.mean(list(res.values()))), valid=res, log=logs, wall=round(time.time() - t0, 1))
(HERE / 'tucker').mkdir(exist_ok=True); (HERE / 'tucker' / f'{tag}.json').write_text(json.dumps(out))
print(json.dumps({k: out[k] for k in ('r', 'L0', 'mean', 'wall')}), flush=True)
