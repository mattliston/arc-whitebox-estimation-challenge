#!/usr/bin/env python3
"""TAG fable-physicist (CR-12c, lead: porter's per-band k4 scalar as a cheap lever-1 arm). The card's per-unit k4 = radial * vv^2 fed to the
pair map (pair_in value[3]) and, for the band containing L15, the final-layer k4 (k4_15) multiplied by s for layers in ONE band; the other
uses of radial (the K22 model, the birth factor 1 + radial/3, the b term) are unchanged. Zero state, zero bill: one literal per band.
B2 ep1 card, learned on (env LEARNED=off for the learned-off control), no refit; VALID16 ratios vs B2 ep1.
usage: band_k4.py <band e.g. 1-7> <scales e.g. 1.2,1.5,2.0> <nets valid:16> <threads> <tag>"""
import sys, json, time, os
from pathlib import Path
HERE = Path(__file__).resolve().parent; RF = HERE.parent; P2 = HERE.parents[2]
sys.path.insert(0, str(P2 / 'fable_analyst_v1')); sys.path.insert(1, str(P2 / 'fable_covhunter_v1'))
import numpy as np, torch
import s6bcard
lo, hi = (int(x) for x in sys.argv[1].split('-')); BAND = set(range(lo, hi + 1)); SC = [float(x) for x in sys.argv[2].split(',')]
torch.set_num_threads(int(sys.argv[4])); TAG = sys.argv[5]; torch.set_grad_enabled(False); OFF = os.environ.get('LEARNED') == 'off'
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
inputs = json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())
nets = inputs['valid'][:int(sys.argv[3].split(':')[1])]
BASE = json.loads((RF / ('tucker/tk2_control_off.json' if OFF else 's6b_B2_ep1_valid40/eval_n40.json')).read_text())['valid']
p0, a0, c0, e0 = s6bcard.payload(torch); sv = torch.load(RF / 's6b_B2_fullnet_lr1e4_trunk5e5/params_epoch1.pt')
for k in p0: p0[k].copy_(sv['p'][k])
a0.copy_(sv['a']); c0.copy_(sv['c'])
for k in e0: e0[k].copy_(sv['e'][k])
if OFF:
    for k in ('head', 'head_b'): p0[k].zero_()
    for k in ('output', 'bias'): e0[k].zero_()
out = {}
for s in SC:
    res = {}; t0 = time.time()
    for rk in nets:
        W = np.load(STORE / Path(man[rk]['path']).parent / 'weights.npy')
        tr = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rk:04d}.npz')['mc_post'][15].astype(np.float64)
        card = s6bcard.card(torch, W, p0, a0, c0, e0)
        card.hooks['pair_in'] = lambda l, v, ctx, s=s: (v[0], v[1], v[2], v[3] * s, v[4], v[5]) if l in BAND else v
        if 15 in BAND: card.hooks['k4_15'] = lambda l, k4, ctx, s=s: k4 * s
        pred = card.run(False).numpy().astype(np.float64); res[str(rk)] = float(((pred[15] - tr) ** 2).mean())
    ks = list(res); rat = [res[k] / BASE[k] for k in ks]
    summ = dict(band=sys.argv[1], scale=s, learned='off' if OFF else 'on', nets=len(ks), pooled_ratio=round(float(np.mean([res[k] for k in ks]) / np.mean([BASE[k] for k in ks])), 5),
                better=int(sum(x < 1 for x in rat)), wall=round(time.time() - t0, 1))
    out[str(s)] = dict(summary=summ, nets=res); print(json.dumps(summ), flush=True)
(HERE / f'band_k4_{TAG}.json').write_text(json.dumps(out))
