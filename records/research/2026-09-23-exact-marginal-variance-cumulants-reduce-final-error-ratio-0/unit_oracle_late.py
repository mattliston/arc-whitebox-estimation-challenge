#!/usr/bin/env python3
"""TAG fable-frame (FR-4): closed-loop value of exact PER-UNIT pre-activation cumulants at late layers. At the chosen layers the pair map
(L1-14) / final scalar map (L15) receives the independent-draw 32M MC truth v (diagonal of C only), k3 (diagonal of K21) and/or k4
(public_g19_attribution_v1/mps_truth_rank*_indep32M.npz: mc_pre_v/k3/k4); everything else (mean, off-diagonal C, planes, learned
parts) stays the card's. Final MSE vs the 1e9 truthpost. usage: unit_oracle_late.py <ranks> <configs e.g. v@14,k3@15,vk3k4@10-15> <threads>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
from common import *
import model_bank_research as model
ranks = [int(x) for x in sys.argv[1].split(',')]; configs = sys.argv[2].split(','); torch.set_num_threads(int(sys.argv[3]))
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]

def parse(cfg):
    if cfg == 'base': return set(), set()
    what, where = cfg.split('@')
    chans = {c for c in ('v', 'k3', 'k4') if c in what}
    if '-' in where: a, b = map(int, where.split('-')); layers = set(range(a, b + 1))
    else: layers = {int(x) for x in where.split('+')}
    return chans, layers

def run(rank, cfg, tr):
    chans, layers = parse(cfg)
    p, a, c, e = model.torch_payload(torch, np, False)
    w = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
    card = model.Card(torch, w, p, a, c, e)
    pair, scal = card.env['_nonlinear_pair_batch'], card.env['nonlinear_scalar']
    T = lambda key, l, like: torch.as_tensor(tr[key][l], dtype=like.dtype)
    def wpair(mu, C, K21, k4, k22_override=None, metric='identity'):
        l = card.call_layer
        if l in layers:
            if 'v' in chans: C = torch.diagonal_scatter(C, T('mc_pre_v', l, C))
            if 'k3' in chans: K21 = torch.diagonal_scatter(K21, T('mc_pre_k3', l, K21))
            if 'k4' in chans:
                k4 = T('mc_pre_k4', l, k4)
                if k22_override is not None: k22_override = torch.diagonal_scatter(k22_override, k4.to(k22_override.dtype))
        return pair(mu, C, K21, k4, k22_override, metric)
    def wscal(mu, v, k3, k4):
        if 15 in layers:
            if 'v' in chans: v = T('mc_pre_v', 15, v)
            if 'k3' in chans: k3 = T('mc_pre_k3', 15, k3)
            if 'k4' in chans: k4 = T('mc_pre_k4', 15, k4)
        return scal(mu, v, k3, k4)
    card.env['_nonlinear_pair_batch'] = wpair; card.env['nonlinear_scalar'] = wscal
    with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
    return float(((pred[15] - truth[15]) ** 2).mean())

for rk in ranks:
    tr = np.load(P2 / 'public_g19_attribution_v1' / f'mps_truth_rank{rk:04d}_indep32M.npz')
    base = None
    for cfg in ['base'] + configs:
        t0 = time.time(); mse = run(rk, cfg, tr)
        if base is None: base = mse
        print(json.dumps(dict(rank=rk, cfg=cfg, mse=mse, ratio=round(mse / base, 4), wall=round(time.time() - t0, 1))), flush=True)
