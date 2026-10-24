#!/usr/bin/env python3
"""TAG fable-analyst (AN-51): dead units out of the NON-column work too, on top of variant (a) (rowdrop_s6b.py: column work off on rows with
a = mu/sigma < -3.5 at L1-14, kept rows = select(a + 3.5, 0) rounded up to 32). The same dead set D_l (recomputed at the 'cpre' hook from
mu and diag Cpre, identical to (a)'s) at L1-14 is also excluded from:
  i    STAR: the star push's output plane rows and columns of D (diagonal included), i.e. its W^T A rows and second-stage rows/columns;
  ii   COVARIANCE: the off-diagonal entries of D's rows and columns in W^T C W; D's own variances (the diagonal) are kept;
  iii  PAIR MAP and learned parts: D's off-diagonal pair terms in the pair map's post covariance and post plane (target; per-unit diagonals
       kept), D's rows and columns of the message-edge increment, and D's GNN rows (mean correction delta and hidden state set to 0).
Composed after the existing v3 knob hooks (cov_pm, edge). Torch S6b card (s6bcard) + B2 ep1 params, no refit, VALID16.
modes: a (check against rowdrop_s6b.jsonl) | a+i | a+ii | a+iii | a+i+ii+iii.  usage: rowdrop_nc_s6b.py <modes> <threads> <out.jsonl>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
import s6bcard
from rowdrop_s6b import card as rd_card
def dead_keep(t, mu, vv):
    score = mu / vv.clamp_min(1e-10).sqrt() + 3.5
    count = min(1024, ((int((score >= 0).sum()) + 31) // 32) * 32)
    keep = t.zeros_like(score); keep[t.argsort(-score, stable=True)[:count]] = 1.0
    return keep
def build(t, w, pl, mode):
    parts = set(mode.split('+')); assert 'a' in parts and parts <= {'a', 'i', 'ii', 'iii'}, mode
    cd = rd_card(t, w, pl, 'a'); cd.nc_keep = {}; cd.nc_counts = {}
    def mu_hook(l, mu, ctx): cd._mu = mu; return mu
    def cpre_hook(l, Cpre, ctx):
        if not 1 <= l <= 14: return Cpre
        keep = dead_keep(t, cd._mu, t.diagonal(Cpre)); cd.nc_keep[l] = keep; cd.nc_counts[l] = int((keep == 0).sum())
        if 'ii' in parts:
            d = t.diagonal(Cpre).clone(); Cpre = Cpre * (keep[:, None] * keep[None, :]); Cpre = t.diagonal_scatter(Cpre, d)
        return Cpre
    cd.hooks['mu'] = mu_hook; cd.hooks['cpre'] = cpre_hook
    if 'i' in parts:
        orig_star = cd.env['_flms_star']
        def star(w_, A):
            out = orig_star(w_, A); l = cd.call_layer
            if 1 <= l <= 14:
                k = cd.nc_keep[l].to(out.dtype); out = out * (k[:, None] * k[None, :])
            return out
        cd.env['_flms_star'] = star
    if 'iii' in parts:
        prev_cov = cd.hooks.get('cov_pm', lambda l, C, ctx: C); prev_edge = cd.hooks.get('edge', lambda l, v, ctx: v)
        def off_mask(M, k):
            d = t.diagonal(M).clone(); M2 = M * (k[:, None] * k[None, :]).to(M.dtype); return t.diagonal_scatter(M2, d)
        def cov_pm(l, C, ctx):
            C = prev_cov(l, C, ctx)
            if 1 <= l <= 14:
                k = cd.nc_keep[l]; C = off_mask(C, k)
                T = ctx['target']; T.copy_(off_mask(T, k))               # in place: the step reuses this tensor
            return C
        def edge(l, v, ctx):
            v = prev_edge(l, v, ctx)
            if 1 <= l <= 14: k = cd.nc_keep[l].to(v.dtype); v = v * (k[:, None] * k[None, :])
            return v
        def delta(l, dl, ctx):
            if 1 <= l <= 14: dl = dl * cd.nc_keep[l].to(dl.dtype)
            return dl
        def state_out(l, out, ctx):
            if 1 <= l <= 14:
                hs = out[8] * cd.nc_keep[l].to(out[8].dtype)[:, None]; out = out[:8] + (hs,) + out[9:]
            return out
        cd.hooks.update(cov_pm=cov_pm, edge=edge, delta=delta, state_out=state_out)
    return cd
if __name__ == '__main__':
    modes = sys.argv[1].split(','); torch.set_num_threads(int(sys.argv[2])); OUT = HERE / sys.argv[3]
    pl = s6bcard.payload(torch, params=P2 / 'fable_physicist_v1/refit6b/s6b_B2_fullnet_lr1e4_trunk5e5/params_epoch1.pt')
    VALID16 = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())['valid'][:16]
    truth = lambda r: np.load(P2 / 'fable_refit_v1/truth688' / f'truthpost_rank{r:04d}.npz')['mc_post'][15].astype(np.float64)
    with OUT.open('a') as fo:
        for r in VALID16:
            w, tp = s6bcard.weights(r), truth(r)
            for mode in modes:
                t0 = time.time(); cd = build(torch, w, pl, mode)
                with torch.no_grad(): pred = cd.run(False).numpy().astype(np.float64)[15]
                rec = dict(tag='AN-51 rowdrop non-column (B2 ep1, s6bcard, no refit)', rank=r, mode=mode, mse=float(((pred - tp) ** 2).mean()),
                           dead_rows={str(k): v for k, v in sorted(cd.nc_counts.items())}, wall=round(time.time() - t0, 1))
                fo.write(json.dumps(rec) + '\n'); fo.flush(); print(json.dumps({k: rec[k] for k in ('rank', 'mode', 'mse', 'wall')}), flush=True)
