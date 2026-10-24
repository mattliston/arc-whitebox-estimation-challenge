#!/usr/bin/env python3
"""TAG fable-physicist (CR-12b): plug the one-GEMM per-unit k4 (kurt_screen.py 'gemm' form, the marginally better of the two) into the card:
the pair map's k4 at L1-14 (pair_in) and k4_15, from layer L-1's post_k4, post22 (patched S2 pair source), carried C and radial.
r35, learned on/off, no refit. usage: kurt_plugin.py <threads> [nets: valid:16] [learned on|off]
(With nets, ratios vs B2 ep1 VALID40 literals (on) or tucker/tk2_control_off.json (off).)"""
import sys, json, ast, os
from pathlib import Path
HERE = Path(__file__).resolve().parent; RF = HERE.parent; P2 = HERE.parents[2]
sys.path.insert(0, str(P2 / 'fable_analyst_v1')); sys.path.insert(1, str(P2 / 'fable_covhunter_v1'))
import numpy as np, torch
import covstep as cs, s6bcard, s2card
torch.set_num_threads(int(sys.argv[1])); torch.set_grad_enabled(False)
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
def WT(rk): return (np.load(STORE / Path(man[rk]['path']).parent / 'weights.npy'),
                   np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rk:04d}.npz')['mc_post'][15].astype(np.float64))
W, TR = WT(35)
sv = torch.load(RF / 's6b_B2_fullnet_lr1e4_trunk5e5/params_epoch1.pt')
def run(off, plug, W=None, TR=None):
    W = globals()['W'] if W is None else W; TR = globals()['TR'] if TR is None else TR
    p0, a0, c0, e0 = s6bcard.payload(torch)
    with torch.no_grad():
        for k in p0: p0[k].copy_(sv['p'][k])
        a0.copy_(sv['a']); c0.copy_(sv['c'])
        for k in e0: e0[k].copy_(sv['e'][k])
        if off:
            for k in ('head', 'head_b'): p0[k].zero_()
            for k in ('output', 'bias'): e0[k].zero_()
    card = s6bcard.card(torch, W, p0, a0, c0, e0)
    key = 'post_v=var, post_k3=post_k3, post_k4=post_k4)'
    t_ = cs.model.Translate().visit(ast.parse(s2card._src.replace(key, key[:-1] + ', post22=post22)'))); ast.fix_missing_locations(t_)
    exec(compile(t_, 's2pair_post22', 'exec'), card.env)
    prev = {}
    def grab(l, out, ctx):
        aux = ctx['aux']
        if 'post22' in aux: prev[l] = (aux['post_k4'], aux['post22'], out[1], float(out[7]))
        return out
    def corr(L):
        if L - 1 not in prev: return None
        pk4, p22, C, rad = prev[L - 1]; Wl = torch.as_tensor(W[L], dtype=torch.float64); Q = Wl * Wl; cd = torch.diagonal(C)
        di = pk4 - rad * cd ** 2; M = 3 * (p22 - (rad / 3) * (torch.outer(cd, cd) + 2 * C * C)); M.diagonal().copy_(di)
        return (Q * (M @ Q)).sum(0)
    if plug:
        card.hooks['state_out'] = grab
        def pin(l, v, ctx):
            dk = corr(l)
            return v if dk is None else (v[0], v[1], v[2], v[3] + dk, v[4], v[5])
        card.hooks['pair_in'] = pin
        card.hooks['k4_15'] = lambda l, k4, ctx: k4 + corr(15)
    pred = card.run(False).numpy().astype(np.float64); return float(((pred[15] - TR) ** 2).mean())
if len(sys.argv) > 2:
    inputs = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
    nets = inputs['valid'][:int(sys.argv[2].split(':')[1])]; off = sys.argv[3] == 'off'
    BASE = json.loads((RF / ('tucker/tk2_control_off.json' if off else 's6b_B2_ep1_valid40/eval_n40.json')).read_text())['valid']
    res = {}
    for rk in nets:
        Wn, Tn = WT(rk); res[str(rk)] = run(off, True, Wn, Tn); print(json.dumps(dict(net=rk, mse=res[str(rk)], ratio=res[str(rk)] / BASE[str(rk)])), flush=True)
    ks = list(res); rat = [res[k] / BASE[k] for k in ks]
    summ = dict(learned=sys.argv[3], nets=len(ks), pooled_ratio=float(np.mean([res[k] for k in ks]) / np.mean([BASE[k] for k in ks])), better=int(sum(x < 1 for x in rat)))
    print(json.dumps(summ), flush=True); (HERE / f'kurt_plugin_v16_{sys.argv[3]}.json').write_text(json.dumps(dict(summary=summ, nets=res))); sys.exit(0)
for off in (False, True):
    b, pl = run(off, False), run(off, True)
    print(json.dumps(dict(learned='off' if off else 'on', base=b, gemm_k4=pl, change=round(pl / b - 1, 4))), flush=True)
