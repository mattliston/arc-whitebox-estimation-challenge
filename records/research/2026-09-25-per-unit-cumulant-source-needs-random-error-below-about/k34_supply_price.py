#!/usr/bin/env python3
"""TAG fable-covhunter (CH-67): ACCURACY-REQUIREMENT CURVE for any per-unit k3+k4 source (lead's option (b), falsifier). Closed-loop price of a
partial-strength supply  k = card + eps * (truth - card)  of the per-unit pre-activation k3 (diag K21) and k4 (the pair map's k4) at a layer
band, with the L12-14 learned mean correction (GNN delta) at s in {1, 0.5} (the analyst's section-34 pricing rule), on the CURRENT card:
s6bcard_joint2 + round-1 dials (dials_r1_ep2.json) + lrstar r32 + gn_linear_hook + fine_ckpt=False + pair_series loop form, loopR1 params,
learned ON (= the S22 torch counterpart). Truths: per-unit k3/k4 = public_g19 mps_truth 32M (r35, r100) or 8M (7 held-out nets, the
analyst's set); final MSE vs the organizer truth (independent of every supply). Certificate first: the same injection on the analyst's card
(s6bcard + B2 ep1, no dials/hooks) must reproduce fable_analyst_v1/l11_oracle_heldout8M.json on net 150. usage: k34_supply_price.py <threads>
v2 (v1 kept; it crashed at net 250 because SJ.truthpost reads public_g22, which holds only 200 nets): the final truth comes from
fable_refit_v1/truth688 (bit-identical to public_g22 on its 200 nets); the grid is the lead's eps {0.5, 1} x {L12-14, L2-14} x s {1, 0.5}."""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(P2 / 'fable_analyst_v1')); sys.path.insert(1, str(P2 / 'fable_engineer_v1'))
import numpy as np, torch
import s6bcard, s6bcard_joint2 as SJ, lrstar_torch_hook as lh, gn_linear_hook as gl, pair_series_torch as ps
torch.set_num_threads(int(sys.argv[1])); torch.set_grad_enabled(False)
R6 = P2 / 'fable_physicist_v1/refit6b'; G19 = P2 / 'public_g19_attribution_v1'
NETS = [35, 100, 150, 250, 300, 400, 500, 600, 650]
DIALS = json.load(open(HERE / 'dials_r1_ep2.json'))
def literals(path):
    sv = torch.load(path); p, a, c, e = s6bcard.payload(torch)
    for k in p: p[k].copy_(sv['p'][k])
    a.copy_(sv['a']); c.copy_(sv['c'])
    for k in e: e[k].copy_(sv['e'][k])
    return p, a, c, e
def truth_units(r):
    f = G19 / f'mps_truth_rank{r:04d}_P32M.npz'
    f = f if f.exists() else G19 / f'mps_truth_rank{r:04d}_P8M.npz'
    T = np.load(f); return f.name, T['mc_pre_k3'].astype(np.float64), T['mc_pre_k4'].astype(np.float64)
def install_supply(cd, T3, T4, eps, a, b, s):
    old = cd.hooks.get('pair_in')
    def pin(l, v, ctx):
        if old is not None: v = old(l, v, ctx)
        if not (a <= l <= b and eps): return v
        mu, Cp, k21, k4, k22, r31 = v
        c3 = torch.diagonal(k21).to(torch.float64); c4 = (k4.to(torch.float64) * torch.ones(k21.shape[0], dtype=torch.float64))
        t3, t4 = torch.tensor(T3[l]), torch.tensor(T4[l])
        k21 = torch.diagonal_scatter(k21, c3 + eps * (t3 - c3)); ctx['k21'] = k21
        k4 = c4 + eps * (t4 - c4); ctx['k4'] = k4
        return (mu, Cp, k21, k4, k22, r31)
    cd.hooks['pair_in'] = pin
    if s != 1.0: cd.hooks['delta'] = lambda l, d, ctx: d * s if 12 <= l <= 14 else d
    return cd
def final_truth(r): return torch.tensor(np.load(P2 / 'fable_refit_v1/truth688' / f'truthpost_rank{r:04d}.npz')['mc_post'][15].astype(np.float64))
def mse(cd, tp): return float(((cd.run(False)[15].to(torch.float64) - tp) ** 2).mean())
OUT = dict(card='s6bcard_joint2 + dials_r1_ep2 + lrstar r32 + gn_linear_hook + fine_ckpt=False + pair_series loop form; loopR1 params; learned ON',
           nets=NETS); t0 = time.time()
print(json.dumps(dict(TAG=OUT['card'])), flush=True)
# certificate on the analyst's card (s6bcard + B2 ep1): net 150, base and k3k4@12-14 (eps 1, s 1)
LB2 = literals(R6 / 's6b_B2_fullnet_lr1e4_trunk5e5/params_epoch1.pt'); ref = json.load(open(P2 / 'fable_analyst_v1/l11_oracle_heldout8M.json'))['150']
_, T3, T4 = truth_units(150); w = s6bcard.weights(150); tp = final_truth(150)
cb = mse(s6bcard.card(torch, w, *LB2), tp); cs_ = mse(install_supply(s6bcard.card(torch, w, *LB2), T3, T4, 1.0, 12, 14, 1.0), tp)
OUT['cert'] = dict(base=cb, base_ref=ref['base'], supply=cs_, supply_ref=ref['base+k3k4@12-14'])
print(json.dumps(dict(cert=OUT['cert'], rel_base=cb / ref['base'] - 1, rel_supply=cs_ / ref['base+k3k4@12-14'] - 1, wall=round(time.time() - t0))), flush=True)
del LB2
LIT = literals(R6 / 's6b_loopR1/params_epoch0.pt')
VAR = [('base', 0, 99, 99, 1.0), ('ctrl_s.5', 0, 99, 99, 0.5)]
VAR += [(f'L12-14_eps{e}_s{s}', e, 12, 14, s) for s in (1.0, 0.5) for e in (0.5, 1.0)]
VAR += [(f'L2-14_eps{e}_s{s}', e, 2, 14, s) for s in (1.0, 0.5) for e in (0.5, 1.0)]
rows = {}
for r in NETS:
    src, T3, T4 = truth_units(r); w = SJ.weights(r); tp = final_truth(r); rows[r] = dict(_truth=src)
    for name, e, a, b, s in VAR:
        SJ.DIALS = dict(DIALS); cd = SJ.card(torch, w, *LIT); lh.install(cd, torch); gl.install(cd, torch); ps.install(cd, torch)
        rows[r][name] = mse(install_supply(cd, T3, T4, e, a, b, s), tp)
    print(json.dumps(dict(net=r, truth=src, **{k: round(v / rows[r]['base'] - 1, 5) for k, v in rows[r].items() if k not in ('_truth', 'base')},
                          wall=round(time.time() - t0))), flush=True)
held = [r for r in NETS if r not in (35, 100)]
S = {}
for name, *_ in VAR[1:]:
    rel = np.array([rows[r][name] / rows[r]['base'] for r in held])
    S[name] = dict(heldout7_geo=float(np.exp(np.log(rel).mean()) - 1), heldout7_wins=int((rel < 1).sum()),
                   r35=rows[35][name] / rows[35]['base'] - 1, r100=rows[100][name] / rows[100]['base'] - 1)
OUT.update(rows=rows, summary=S); print(json.dumps(dict(summary=S)), flush=True)
(HERE / 'k34_supply_price.json').write_text(json.dumps(OUT, indent=1))
