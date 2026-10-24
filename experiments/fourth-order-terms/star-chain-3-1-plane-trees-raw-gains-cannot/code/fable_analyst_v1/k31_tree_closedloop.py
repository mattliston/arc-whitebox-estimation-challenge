#!/usr/bin/env python3
"""TAG fable-analyst (AN-14): CLOSED-LOOP value of the Gaussian-born K31 TREE terms in the torch card (learned parts ON, no refit), through
the physicist's harness (fable_physicist_v1/model_bank_phys.PHYS_SUPPLY 'fn' -> remainder31 added to the card's t4*rho form, pair map only).
At layer l the trees are built from the CARD's own pre-activation state at l-1 (mu, Cpre) and W_l (formulas certified in k31_tree_cert.py):
STAR = 3((W^T o B o B) D3) B^T + ((B o B o B) D3) W ; CHAIN = 6[((W^T D) o (P C_off)) B^T] + 6[(B o (P C_off)) D W]; B = W^T (g o C_off).
spec = <star|chain|tree>@<L0>-<L1>[x<scale>] | base. Cost if deployed: star 3 n^3 MACs/layer (B shared), chain +3 n^3 MACs/layer.
usage: k31_tree_closedloop.py <ranks> <specs> <threads>"""
import sys, json, time, math
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
sys.path.insert(0, str(P2 / 'fable_physicist_v1'))
import numpy as np, torch
import run_supply as rs
ranks = [int(x) for x in sys.argv[1].split(',')]; specs = sys.argv[2].split(',')
SQ2 = math.sqrt(2.0)
def trees(t, mu, C, W, use_star, use_chain):
    v = torch.diagonal(C); s = v.sqrt(); al = mu / s
    phi = torch.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = torch.erfc(-al / SQ2) / 2
    d = phi / s; d3 = -al * phi / v
    Co = C - torch.diag(v)
    Wt = W.T; B = Wt @ (g[:, None] * Co)
    out = torch.zeros_like(C)
    if use_star:
        out = out + 3 * ((Wt * B * B) * d3[None, :]) @ B.T + ((B * B * B) * d3[None, :]) @ W
    if use_chain:
        WtD = Wt * d[None, :]; PC = (WtD * B) @ Co
        out = out + 6 * ((WtD * PC) @ B.T) + 6 * (((B * PC) * d[None, :]) @ W)
    return out
def make(kind, L0, L1, scale):
    prev = {}
    def fn(card, layer, mu, Cpre, vv, radial, k21, k4, k22, parts):
        t = card.t; r31 = None
        if (layer - 1) in prev and L0 <= layer <= L1:
            mu_p, C_p = prev[layer - 1]
            K = trees(t, mu_p, C_p, card.ws[layer], kind in ('star', 'tree'), kind in ('chain', 'tree'))
            s = vv.sqrt(); eye = torch.eye(vv.shape[0], dtype=torch.bool)
            r31 = (scale * K / (s[:, None] ** 3 * s[None, :])).masked_fill(eye, 0.0).to(torch.float32)
        prev[layer] = (mu.detach().to(torch.float64), Cpre.detach().to(torch.float64))
        return k21, k4, k22, r31
    return fn
base = {}
for rank in ranks:
    w = np.load(rs.STORE / Path(rs.man[rank]['path']).parent / 'weights.npy')
    for spec in specs:
        if spec == 'base':
            rs.model.PHYS_SUPPLY = None
        else:
            kind, rng_ = spec.split('@'); scale = 1.0
            if 'x' in rng_: rng_, sc = rng_.split('x'); scale = float(sc)
            L0, L1 = [int(x) for x in rng_.split('-')]
            fn = make(kind, L0, L1, scale); rs.model.PHYS_SUPPLY = {l: {'fn': fn} for l in range(0, 15)}
        cap = {}; rs.model.PHYS_CAPTURE = cap
        card = rs.model.Card(torch, w, rs.p, rs.a, rs.c, rs.e); t0 = time.time()
        with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
        g = rs.grade(rank, pred, cap)
        if spec == 'base': base[rank] = g['final']
        print(json.dumps(dict(rank=rank, spec=spec, final=g['final'], rel=(g['final'] / base[rank] - 1) if rank in base else None,
                              post={k: float('%.3e' % v) for k, v in g['post'].items()},
                              v_rel={k: round(g['v_rel'][k], 5) for k in (5, 8, 11, 14, 15)} if 'v_rel' in g else None,
                              wall=round(time.time() - t0, 1))), flush=True)
        del card, cap
