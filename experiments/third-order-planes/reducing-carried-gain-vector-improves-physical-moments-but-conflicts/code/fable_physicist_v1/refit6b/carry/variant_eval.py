#!/usr/bin/env python3
"""TAG fable-physicist (CR-10): no-refit tests of the CHEAP forms of the aligned distinct-content term (leg_screen.py: the born-unit-side
plane feed j1, cos 0.40-0.49 to the distinct error at physical strength; the card learns it at c1 ~0.13-0.33 on the orthogonalized direction).
Card: s6bcard + B2 ep1 literals, learned on (env LEARNED=off zeroes head/head_b/edge output/bias). Variants:
  base      the card
  bdcopy    the card with my torch copy of the parent _bd_update installed (must be bit-exact: validation)
  c1one     the j1 base coefficient cstar[:,1] = 1 at every layer (c1 = 1, orthogonalized direction): a literal change
  j1phys    j1 at PHYSICAL strength: the full, unorthogonalized direction -(g_i/2) eta_a K21[a,i] at coefficient 1 (my _bd_update copy)
  j1physX   the same at coefficient X (e.g. j1phys1.5)
  <v>+bS    any variant plus the carried b scaled by S at every layer (b_post hook), e.g. j1phys0.5+b0.9
Metrics: final-layer MSE (ratio vs B2 ep1 per net; vs the learned-off control when LEARNED=off) and, on r35, the kink-plane (noise-
corrected, floors 2.08/1.45%) and per-unit k3 errors at L10/L14. usage: variant_eval.py <variants> <nets 35|valid:16> <threads> <tag>"""
import sys, json, time, math, os
from pathlib import Path
HERE = Path(__file__).resolve().parent; RF = HERE.parent; P2 = HERE.parents[2]
sys.path.insert(0, str(P2 / 'fable_analyst_v1')); sys.path.insert(1, str(P2 / 'fable_covhunter_v1'))
import numpy as np, torch
import s6bcard
VARS = sys.argv[1].split(','); NETS = sys.argv[2]; torch.set_num_threads(int(sys.argv[3])); TAG = sys.argv[4]
torch.set_grad_enabled(False); OFF = os.environ.get('LEARNED') == 'off'
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
inputs = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
nets = inputs['valid'][:int(NETS.split(':')[1])] if NETS.startswith('valid:') else [int(x) for x in NETS.split(',')]
BASE = json.loads((RF / ('tucker/tk2_control_off.json' if OFF else 's6b_B2_ep1_valid40/eval_n40.json')).read_text())['valid']
TP = np.load(P2 / 'public_g19_attribution_v1' / 'mps_planes_rank0035.npz'); FLOOR = {10: 0.0208, 14: 0.0145}

def diag0(x): return torch.diagonal_scatter(x, torch.zeros(x.shape[0], dtype=x.dtype))
def make_bd(j1_mode='orth', j1_coef=1.0):
    def _bd_update(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients):   # torch copy of zero.py _bd_update (+ the j1 switch)
        n = int(bp.shape[0]); c = coefficients.to(torch.float64)
        alpha, sigma, g = aux['alpha'], aux['sigma'], aux['gate']; eta = alpha / sigma
        half_d = 0.5 * d; negative_half_g = -0.5 * g; negative_half_d = -half_d
        skew = aux['pre_k3'] / (sigma * sigma * sigma); gate_jet = skew * (alpha * alpha - 1) * aux['pdf'] / 6
        denominator = torch.clamp_min(torch.sum(bp * bp, dim=0), 1e-20)
        change = torch.zeros((n, n), dtype=torch.float64)
        for j in range(4):
            if j == 0: direction = half_d[:, None] * K21
            elif j == 1: direction = (negative_half_g[:, None] * K21.T) * eta[None, :]
            elif j == 2: direction = (negative_half_d[:, None] * (Cpre * Cpre)) * eta[None, :]
            else: direction = Cpre * gate_jet[:, None]
            direction = diag0(direction)
            if j == 1 and j1_mode == 'phys':
                change = change + direction * j1_coef
            else:
                projection = torch.sum(bp * direction, dim=0) / denominator
                direction = direction - bp * projection[None, :]
                change = change + direction * c[j]
        change = diag0(change)
        old32 = (bp * alive[:, None]).to(torch.float32); new32 = ((bp + change) * alive[:, None]).to(torch.float32)
        old = old32.to(torch.float64); new = new32.to(torch.float64); difference = new - old
        a64 = a.to(torch.float32).to(torch.float64)
        t64 = (t * alive).to(torch.float32).to(torch.float64); s64 = (s * alive).to(torch.float32).to(torch.float64)
        weight = (a64 * (t64 - s64)) * births; quadratic_weight = (a64 * (t64 * t64 - s64 * s64)) * births
        delta = ((2 * old * difference + difference * difference) * weight[None, :] + quadratic_weight[:, None] * difference.T)
        return new32, diag0(delta)
    return _bd_update

p0, a0, c0, e0 = s6bcard.payload(torch); sv = torch.load(RF / 's6b_B2_fullnet_lr1e4_trunk5e5/params_epoch1.pt')
def load_literals():
    with torch.no_grad():
        for k in p0: p0[k].copy_(sv['p'][k])
        a0.copy_(sv['a']); c0.copy_(sv['c'])
        for k in e0: e0[k].copy_(sv['e'][k])
        if OFF:
            for k in ('head', 'head_b'): p0[k].zero_()
            for k in ('output', 'bias'): e0[k].zero_()
out = {}
for vfull in VARS:
    var, _, bsuf = vfull.partition('+b'); BS = float(bsuf) if bsuf else None
    load_literals()
    if var == 'c1one':
        with torch.no_grad(): c0[:, 1] = 1.0
    res = {}; t0 = time.time()
    for rk in nets:
        W = np.load(STORE / Path(man[rk]['path']).parent / 'weights.npy')
        truth = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rk:04d}.npz')['mc_post'][15].astype(np.float64)
        card = s6bcard.card(torch, W, p0, a0, c0, e0); pl = {}
        if var == 'bdcopy': card.env['_bd_update'] = make_bd('orth')
        if var.startswith('j1phys'): card.env['_bd_update'] = make_bd('phys', float(var[6:]) if len(var) > 6 else 1.0)
        if BS is not None: card.hooks['b_post'] = lambda l, b, ctx, BS=BS: b * BS
        if rk == 35:
            def grab(l, o, ctx):
                if l in (10, 14): pl[l] = (ctx['k21'].numpy().copy(), ctx['response_mask'].numpy() > .5, ctx['alive'].numpy() > .5)
                return o
            card.hooks['state_out'] = grab
        pred = card.run(False).numpy().astype(np.float64)
        r = dict(mse=float(((pred[15] - truth) ** 2).mean())); r['ratio'] = r['mse'] / BASE[str(rk)] if str(rk) in BASE else None
        for l, (k21, rs, al) in pl.items():
            blk = np.ix_(rs, al); off = ~np.eye(1024, dtype=bool)[blk]; tb = TP[f'K21_{l}'][blk] * off
            e = float(np.linalg.norm(k21[blk] * off - tb) / np.linalg.norm(tb)); r[f'plane_L{l}_nc'] = round(math.sqrt(max(e * e - FLOOR[l] ** 2, 0)), 4)
            kt = np.diagonal(TP[f'K21_{l}'])[al]; r[f'k3_L{l}'] = round(float(np.linalg.norm(np.diagonal(k21)[al] - kt) / np.linalg.norm(kt)), 4)
        res[str(rk)] = r
    ks = [k for k in res if res[k]['ratio'] is not None]; summ = dict(variant=vfull, learned='off' if OFF else 'on', nets=len(res), wall=round(time.time() - t0, 1))
    if ks:
        rat = [res[k]['ratio'] for k in ks]
        summ.update(pooled_ratio=round(float(np.mean([res[k]['mse'] for k in ks]) / np.mean([BASE[k] for k in ks])), 5), better=int(sum(x < 1 for x in rat)), n=len(ks))
    if '35' in res: summ.update({k: v for k, v in res['35'].items() if k.startswith(('plane', 'k3'))}, mse35=res['35']['mse'])
    out[vfull] = dict(summary=summ, nets=res); print(json.dumps(summ), flush=True)
(HERE / f'variant_{TAG}.json').write_text(json.dumps(out))
