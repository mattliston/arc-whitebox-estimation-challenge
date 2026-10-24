#!/usr/bin/env python3
"""TAG fable-engineer (E-63, lead 19:10: "covariance diag+low-rank test, r = 128/256/512, kill if r256 > +0.5%"): no-refit torch test
of a diag + low-rank covariance on the S18 physics (fable_physicist_v1/refit6b/s6bcard_j05 + s6b_B2ep1_j05_1ep/params_epoch0.pt, the
S17 literals, + lrstar_torch_hook = the S18 low-rank star). The covariance leaving every layer 0-14 (state index 1) is replaced by
L_r + diag(diag(C) - diag(L_r)), L_r = the rank-r eigen-truncation of C (float64 eigh, top r eigenvalues): the per-unit variances stay
exact, the off-diagonal part becomes rank r. Nets: the porter's 8 VALID nets (inputs.json valid[:8]). Final-layer MSE vs the 8M truth.
usage (python with torch): cov_lowrank_s18.py <arms e.g. base,c128,c256,c512> <threads> <out.jsonl>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
sys.path.insert(0, str(P2 / 'fable_analyst_v1')); sys.path.insert(0, str(P2 / 'fable_physicist_v1/refit6b')); sys.path.insert(0, str(HERE))
import numpy as np, torch
import s6bcard, s6bcard_j05
import lrstar_torch_hook as lh
def make_hook(arm, prev):
    def hook(layer, out, ctx):
        if prev is not None: out = prev(layer, out, ctx)
        if layer > 14 or arm == 'base': return out
        r = int(arm[1:]); C = out[1]
        C64 = C.to(torch.float64); ev, U = torch.linalg.eigh(C64)
        Ur, er = U[:, -r:], ev[-r:]
        L = (Ur * er) @ Ur.T
        Cn = L + torch.diag(torch.diagonal(C64) - torch.diagonal(L))
        return tuple(Cn.to(C.dtype) if i == 1 else v for i, v in enumerate(out))
    return hook
if __name__ == '__main__':
    arms = sys.argv[1].split(','); torch.set_num_threads(int(sys.argv[2])); OUT = HERE / sys.argv[3]
    if OUT.exists(): raise SystemExit(f"refusing to overwrite {OUT}")
    PL = s6bcard.payload(torch, params=P2 / 'fable_physicist_v1/refit6b/s6b_B2ep1_j05_1ep/params_epoch0.pt')
    NETS = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())['valid'][:8]
    truth = lambda r: np.load(P2 / 'fable_refit_v1/truth688' / f'truthpost_rank{r:04d}.npz')['mc_post'][15].astype(np.float64)
    with OUT.open('a') as fo:
        for r in NETS:
            w, tp = s6bcard.weights(r), truth(r)
            for arm in arms:
                t0 = time.time(); cd = lh.install(s6bcard_j05.card(torch, w, *PL), torch)
                cd.hooks['state_out'] = make_hook(arm, cd.hooks.get('state_out'))
                with torch.no_grad(): pred = cd.run(False).numpy().astype(np.float64)[15]
                rec = dict(tag='E-63 covariance diag+low-rank (S18 physics, no refit)', rank=r, arm=arm, mse=float(((pred - tp) ** 2).mean()), wall=round(time.time() - t0, 1))
                fo.write(json.dumps(rec) + '\n'); fo.flush(); print(json.dumps(rec), flush=True)
