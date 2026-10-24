#!/usr/bin/env python3
"""TAG fable-frame (FR-2, +FR-5 Cperp/Cpar/Cperpoff: fix only the part of dC orthogonal to the card mean m, or only the rest): closed-loop VALUE of an exact carried state. After the card's step at layer l, replace its post-activation
covariance C (all / off-diagonal only / diagonal only) and/or mean m by the 32M MC truth (fable_mc_v1/mc2_rank*_P32M.npz, same
x@W[l] convention), then continue the card unchanged (learned parts ON, no refit). Final-layer MSE vs the 1e9 truthpost.
usage: cov_oracle_closedloop.py <ranks> <configs e.g. C@13,Coff@13,Cdiag@13,m@13,mC@13,C@4+7+10+13> <threads>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
from common import *
import model_bank_research as model
ranks = [int(x) for x in sys.argv[1].split(',')]; configs = sys.argv[2].split(','); torch.set_num_threads(int(sys.argv[3]))
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]

def run(rank, cfg, mc2):
    p, a, c, e = model.torch_payload(torch, np, False)
    w = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
    card = model.Card(torch, w, p, a, c, e)
    plan = {}
    if cfg != 'base':
        kind, layers = cfg.split('@')
        for l in layers.split('+'): plan[int(l)] = kind
    orig_step = card.step
    def step(layer, state):
        out = orig_step(layer, state)
        if layer in plan:
            kind = plan[layer]; m, C = out[0], out[1]
            Ct = torch.as_tensor(mc2[f'cov_post_{layer}'], dtype=C.dtype); mt = torch.as_tensor(mc2[f'mean_post_{layer}'], dtype=m.dtype)
            if kind in ('C', 'mC'): C = Ct
            elif kind == 'Coff': C = Ct - torch.diag(torch.diagonal(Ct)) + torch.diag(torch.diagonal(C))
            elif kind == 'Cdiag': C = C - torch.diag(torch.diagonal(C)) + torch.diag(torch.diagonal(Ct))
            elif kind in ('Cperp', 'Cpar', 'Cperpoff'):
                mh = m / m.norm(); P = torch.eye(C.shape[0], dtype=C.dtype) - torch.outer(mh, mh)
                D = C - Ct; Dperp = P @ D @ P
                if kind == 'Cperpoff': Dperp = Dperp - torch.diag(torch.diagonal(Dperp))
                C = C - Dperp if kind in ('Cperp', 'Cperpoff') else C - (D - Dperp)
            if kind in ('m', 'mC'): m = mt
            out = (m, C) + tuple(out[2:])
        return out
    card.step = step
    with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
    return float(((pred[15] - truth[15]) ** 2).mean())

for rk in ranks:
    mc2 = np.load(P2 / 'fable_mc_v1' / f'mc2_rank{rk:04d}_P32M.npz')
    base = None
    for cfg in ['base'] + configs:
        t0 = time.time(); mse = run(rk, cfg, mc2)
        if base is None: base = mse
        print(json.dumps(dict(rank=rk, cfg=cfg, mse=mse, ratio=round(mse / base, 4), wall=round(time.time() - t0, 1))), flush=True)
