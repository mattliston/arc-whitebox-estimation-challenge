#!/usr/bin/env python3
"""TAG fable-physicist (RF6-6): test 1 of the bounded-column carriers, BUDGETED BIRTHS with a CONSUMPTION ranking, no refit.
Torch S6b card (fable_analyst_v1/s6bcard.py) with B2 epoch-1 literals (--resume). A 'state_out' hook at every layer L0..14 ranks that
layer's newborn pairs (the +/- columns of each born unit) by their contribution to the NEXT layer's plane read, computed only from the
card's own state (no truth, no MC): mu' = W'^T m, v' = diag(W'^T C W'), g' = Phi(mu'/s'), d' = pdf/s', kink rows r = {min(g',1-g') >= .01},
live columns c = {g' >= .01}; pair plane A = lam (d' o pp+^2)_r (g' o pp+)_c^T - lam (d' o pp-^2)_r (g' o pp-)_c^T with pp = W'^T p;
importance = |A|_F. The lowest (1 - keep) share get lam = 0 (both columns), which is exactly their removal for every plane, diagonal
and frame. Option catch=1: their post-space plane moves into the star residual (rest += dK), so the star carries it one more layer.
Writes per-net VALID MSE and the kept pair counts per layer (for births_bill.py).
usage: births_prune_eval.py <keep> <catch 0|1> <L0> <nvalid> <threads> <tag>"""
import sys, json, math, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parents[1]; sys.path.insert(0, str(P2 / 'fable_analyst_v1'))
import numpy as np, torch
import s6bcard
keep, catch, L0, nvalid, threads, tag = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
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
Phi = lambda x: 0.5 * t.erfc(-x / math.sqrt(2)); phi = lambda x: t.exp(-x * x / 2) / math.sqrt(2 * math.pi)
def make_hook(card, kept_log):
    def hook(layer, out, ctx):
        if layer < L0 or layer > 14 or keep >= 1.0:
            return out
        m, C, p, lam, star = out[0], out[1], out[2], out[3], out[4]
        stops = out[12]; N = int(lam.numel()); first = int(stops[-2]) if len(stops) > 1 else 0
        assert int(stops[-1]) == N and (N - first) % 2 == 0
        nsel = (N - first) // 2                                                  # newborn cohort = +/- columns of nsel born units
        if nsel == 0:
            return out
        W1 = card.ws[layer + 1]                                                   # float64 next-layer weights
        mu1 = W1.T @ m.to(t.float64); v1 = ((C.to(t.float64) @ W1) * W1).sum(0).clamp_min(1e-12); s1 = v1.sqrt()
        a1 = mu1 / s1; g1 = Phi(a1); d1 = phi(a1) / s1
        rows = (t.minimum(g1, 1 - g1) >= .01); cols = (g1 >= .01)
        Pp = p[:, N - 2 * nsel:N - nsel].to(t.float64); Pm = p[:, N - nsel:].to(t.float64); lp = lam[N - 2 * nsel:N - nsel].to(t.float64)
        qp, qm = W1.T @ Pp, W1.T @ Pm
        up = (d1[:, None] * qp * qp)[rows] * lp[None, :]; um = -(d1[:, None] * qm * qm)[rows] * lp[None, :]
        vp = (g1[:, None] * qp)[cols]; vm = (g1[:, None] * qm)[cols]
        imp2 = (up * up).sum(0) * (vp * vp).sum(0) + (um * um).sum(0) * (vm * vm).sum(0) + 2 * (up * um).sum(0) * (vp * vm).sum(0)
        nkeep = int(round(keep * nsel)); order = t.argsort(imp2, descending=True)
        drop = order[nkeep:]
        lam = lam.clone()
        if catch and drop.numel():
            al = ctx['alive'].to(t.float64); live = al[:, None] * al[None, :]
            Dp, Dm, ld = Pp[:, drop], Pm[:, drop], lp[drop]
            dK = ((Dp * Dp) * ld[None, :]) @ Dp.T - ((Dm * Dm) * ld[None, :]) @ Dm.T
            dK = dK * live
            star = (star.to(t.float64) + 3 * dK.T - 2 * t.diag(t.diagonal(dK))).to(t.float32)
        lam[N - 2 * nsel + drop] = 0.0; lam[N - nsel + drop] = 0.0
        kept_log[layer] = dict(born=nsel, kept=nkeep)
        return (out[0], out[1], out[2], lam, star) + tuple(out[5:])
    return hook
t0 = time.time(); res = {}; kept = {}
for r in VALID:
    W = np.load(STORE / Path(man[r]['path']).parent / 'weights.npy')
    truth = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{r:04d}.npz')['mc_post'][15].astype(np.float64)
    card = s6bcard.card(torch, W, p0, a0, c0, e0)
    log = {}; card.hooks['state_out'] = make_hook(card, log)
    pred = card.run(False).numpy().astype(np.float64)
    res[str(r)] = float(((pred[15] - truth) ** 2).mean()); kept[str(r)] = log
out = dict(keep=keep, catch=catch, L0=L0, mean=float(np.mean(list(res.values()))), valid=res, kept=kept, wall=round(time.time() - t0, 1))
(HERE / 'births').mkdir(exist_ok=True); (HERE / 'births' / f'{tag}.json').write_text(json.dumps(out))
print(json.dumps({k: out[k] for k in ('keep', 'catch', 'L0', 'mean', 'wall')}), flush=True)
