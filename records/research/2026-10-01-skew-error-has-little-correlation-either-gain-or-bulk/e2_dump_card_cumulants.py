#!/usr/bin/env python3
"""TAG opus-reexam E2: dump the torch S6b card's per-unit pre-activation cumulants at every layer (learned ON, closed loop):
mu, v, k3 (and its three parts: CP bank diagonal, star diagonal, gain carrier 3 v b), k4 (= radial v^2), radial, host, delta, m_out.
Used with the E1 Monte Carlo decomposition to ask where the card's skew error sits: in the collective gain part or the bulk part.
usage: e2_dump_card_cumulants.py <rank> <threads>  -> e2_card_rank{rank}.npz"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
sys.path.insert(0, str(P2 / 'fable_analyst_v1'))
import numpy as np, torch
import s6bcard as S
rank = int(sys.argv[1]); torch.set_num_threads(int(sys.argv[2]))
pl = S.payload(torch); w, tp = S.weights(rank), S.truthpost(rank)
card = S.card(torch, w, *pl); cap = {}
k3hook = card.hooks.get('k3_15'); pin_old = card.hooks.get('pair_in'); delta_old = card.hooks.get('delta')
def npy(x): return x.detach().double().numpy().copy()
def pair_in(layer, value, ctx):
    if pin_old is not None: value = pin_old(layer, value, ctx)
    mu, Cpre, k21, k4, k22, rem = value
    vv = torch.diagonal(Cpre)
    cap[f'mu_{layer}'] = npy(mu); cap[f'v_{layer}'] = npy(vv); cap[f'k3_{layer}'] = npy(torch.diagonal(k21))
    cap[f'k3bank_{layer}'] = npy(torch.diagonal(ctx['cp'])); cap[f'k3star_{layer}'] = npy(torch.diagonal(ctx['sp']))
    cap[f'k3b_{layer}'] = npy(3 * vv * ctx['bpre']); cap[f'bpre_{layer}'] = npy(ctx['bpre'])
    cap[f'k4_{layer}'] = npy(k4 if k4.ndim else k4.expand(1024)); cap[f'radial_{layer}'] = npy(ctx['radial'])
    cap[f'alive_{layer}'] = npy(ctx['alive']); cap[f'resp_{layer}'] = npy(ctx['response_mask'])
    return value
def k3_15(layer, k3, ctx):
    k3n = k3hook(layer, k3, ctx) if k3hook is not None else k3
    cap['mu_15'] = npy(ctx['mu']); cap['v_15'] = npy(ctx['vv']); cap['k3_15'] = npy(k3n); cap['k3raw_15'] = npy(k3)
    cap['k3bank_15'] = npy(ctx['bank']); cap['k3star_15'] = npy(ctx['star']); cap['k3b_15'] = npy(ctx['bterm']); cap['radial_15'] = npy(ctx['radial'])
    return k3n
def delta(layer, d, ctx):
    if delta_old is not None: d = delta_old(layer, d, ctx)
    cap[f'host_{layer}'] = npy(ctx['host']); cap[f'delta_{layer}'] = npy(d)
    return d
card.hooks.update(pair_in=pair_in, k3_15=k3_15, delta=delta)
with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
cap['pred'] = pred; cap['truth'] = tp
mse = ((pred - tp) ** 2).mean(1)
print('rank', rank, 'per-layer MSE x1e9:', ' '.join(f'{x * 1e9:.2f}' for x in mse))
np.savez(HERE / f'e2_card_rank{rank:04d}.npz', **cap)
