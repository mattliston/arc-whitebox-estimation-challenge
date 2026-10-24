#!/usr/bin/env python3
"""TAG fable-physicist (CR-8): the distinct-content regression (distinct_regress.py) says the b channel's DISTINCT content is ~2.5-4x too
large (best-fit scale 0.24-0.42 at L6-14). On the torch card (s6bcard + B2 ep1, learned on, NO refit) scale the carried b vector by s at
layers 0-14 through covstep's 'b_post' hook (the star absorbs b's post-plane part, so s changes b's distinct push, its carry, and the
final layer's 3 v bpre k3 term). Reports the card's own kink-plane error at L10/L14 (8M truth, k-statistic floors 2.08% / 1.45%) and
the final-layer MSE; also the per-unit k3 (alive diagonal) errors. Scale 'Rs' (e.g. R0.6) = scale s at L0-14 but the final layer's
3 v bpre k3 term RESTORED from an unscaled b recursion tracked in the hook (b_u = g W^T b_u + new, new = the card's own new b).
(R variants are INVALID: with b scaled the star absorbs b's plane part, so restoring the unscaled L15 b term double counts.)
Env LEARNED=off zeroes the GNN mean head (head, head_b) and the covariance-edge output/bias (ratios then vs tucker/tk2_control_off.json).
Scale 'G<g>' (e.g. G0) = b unscaled but the S2 gain b-fix gamma set to g (card default 0.1).
usage: b_scale_test.py <scales e.g. 1,0.6,R0.6,G0> <nets e.g. 35 or valid:16> <threads> <tag>"""
import sys, json, time, math
from pathlib import Path
HERE = Path(__file__).resolve().parent; RF = HERE.parent; P2 = HERE.parents[2]
sys.path.insert(0, str(P2 / 'fable_analyst_v1')); sys.path.insert(1, str(P2 / 'fable_covhunter_v1'))
import os
import numpy as np, torch
import covstep as cs, s6bcard
SC = sys.argv[1].split(','); NETS = sys.argv[2]; torch.set_num_threads(int(sys.argv[3])); TAG = sys.argv[4]
torch.set_grad_enabled(False)
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
inputs = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
nets = inputs['valid'][:int(NETS.split(':')[1])] if NETS.startswith('valid:') else [int(x) for x in NETS.split(',')]
OFF = os.environ.get('LEARNED') == 'off'
BASE = json.loads((RF / ('tucker/tk2_control_off.json' if OFF else 's6b_B2_ep1_valid40/eval_n40.json')).read_text())['valid']
TP = np.load(P2 / 'public_g19_attribution_v1' / 'mps_planes_rank0035.npz'); FLOOR = {10: 0.0208, 14: 0.0145}
p0, a0, c0, e0 = s6bcard.payload(torch)
sv = torch.load(RF / 's6b_B2_fullnet_lr1e4_trunk5e5/params_epoch1.pt')
for k in p0: p0[k].copy_(sv['p'][k])
a0.copy_(sv['a']); c0.copy_(sv['c'])
for k in e0: e0[k].copy_(sv['e'][k])
if OFF:
    for k in ('head', 'head_b'): p0[k].zero_()
    for k in ('output', 'bias'): e0[k].zero_()
out = {}
for sarg in SC:
    R = sarg.startswith('R'); GF = float(sarg[1:]) if sarg.startswith('G') else None
    s = 1.0 if GF is not None else float(sarg.lstrip('R')); res = {}; t0 = time.time()
    for rk in nets:
        W = np.load(STORE / Path(man[rk]['path']).parent / 'weights.npy')
        truth = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rk:04d}.npz')['mc_post'][15].astype(np.float64)
        card = s6bcard.card(torch, W, p0, a0, c0, e0); pl = {}
        if GF is not None: card.knobs['bfix_gamma'] = GF
        bu = {}
        def bhook(l, b, ctx, s=s, W=W):
            new = b - ctx['g'] * ctx['bpre']                                 # this layer's new b (card's own, on the scaled path)
            bpu = torch.as_tensor(W[l], dtype=torch.float64).T @ bu[l - 1] if l > 0 else ctx['bpre']
            bu[l] = ctx['g'] * bpu + new                                     # the unscaled recursion
            return b * s
        if s != 1.0: card.hooks['b_post'] = bhook
        if R and s != 1.0:
            orig = card.hooks['k3_15']
            def k3hook(l, k3, ctx, W=W, orig=orig):
                bpre_s = (k3 - ctx['bank'] - ctx['star']) / (3 * ctx['vv'])
                bpre_u = torch.as_tensor(W[15], dtype=torch.float64).T @ bu[14]
                ctx = dict(ctx); ctx['bterm'] = 3 * ctx['vv'] * bpre_u
                return orig(l, ctx['bank'] + ctx['star'] + ctx['bterm'], ctx)
            card.hooks['k3_15'] = k3hook
        if rk == 35:
            def grab(l, o, ctx):
                if l in (10, 14): pl[l] = (ctx['k21'].numpy().copy(), ctx['response_mask'].numpy() > .5, ctx['alive'].numpy() > .5)
                return o
            card.hooks['state_out'] = grab
        pred = card.run(False).numpy().astype(np.float64)
        r = dict(mse=float(((pred[15] - truth) ** 2).mean()))
        r['ratio'] = r['mse'] / BASE[str(rk)] if str(rk) in BASE else None
        for l, (k21, rs, al) in pl.items():
            blk = np.ix_(rs, al); off = ~np.eye(1024, dtype=bool)[blk]; tb = TP[f'K21_{l}'][blk] * off
            e = float(np.linalg.norm(k21[blk] * off - tb) / np.linalg.norm(tb)); r[f'plane_L{l}_nc'] = round(math.sqrt(max(e * e - FLOOR[l] ** 2, 0)), 4)
            kt = np.diagonal(TP[f'K21_{l}'])[al]; r[f'k3_L{l}'] = round(float(np.linalg.norm(np.diagonal(k21)[al] - kt) / np.linalg.norm(kt)), 4)
        res[str(rk)] = r
    ks = [k for k in res if res[k]['ratio'] is not None]
    summ = dict(scale=sarg, nets=len(res), wall=round(time.time() - t0, 1))
    if ks:
        rat = [res[k]['ratio'] for k in ks]; summ.update(pooled_ratio=float(np.mean([res[k]['mse'] for k in ks]) / np.mean([BASE[k] for k in ks])),
                                                        better=int(sum(x < 1 for x in rat)), n=len(ks))
    if '35' in res: summ.update({k: v for k, v in res['35'].items() if k.startswith(('plane', 'k3'))}, mse35=res['35']['mse'])
    out[sarg] = dict(summary=summ, nets=res); print(json.dumps(summ), flush=True)
(HERE / f'bscale_{TAG}.json').write_text(json.dumps(out))
