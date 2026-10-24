#!/usr/bin/env python3
"""TAG fable-physicist (CR-11): where is the FINAL error born once the second-order state is exact? B2 ep1 card (s6bcard, learned on, r35)
with the 8M-truth pre mean + covariance injected at every layer 0-14 ('mu'/'cpre' hooks; = second_order_oracle.py mC), then, cumulatively:
  mC15   + the truth pre mean and variance at L15 ('mu' at 15, 'vv15')
  k15    + the truth per-unit k3 and k4 at L15 ('k3_15' after the row cut, 'k4_15')
  k14    + the truth per-unit k3 (the k21 diagonal) and k4 at L14 (pair_in hook) -> the L14 host mean from true marginal cumulants
  k1014  + the same at L10-13
  ...+nd15 any of these with the learned final-layer mean correction delta_15 set to zero ('delta' hook at L15)
Truth: public_g19_attribution_v1/mps_cov2_rank0035.npz (8M; independent of the 1e9 final truth). usage: mean_path_oracle.py <threads>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; RF = HERE.parent; P2 = HERE.parents[2]
sys.path.insert(0, str(P2 / 'fable_analyst_v1')); sys.path.insert(1, str(P2 / 'fable_covhunter_v1'))
import numpy as np, torch
import s6bcard
torch.set_num_threads(int(sys.argv[1])); torch.set_grad_enabled(False)
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
W = np.load(STORE / Path(man[35]['path']).parent / 'weights.npy')
TS = np.load(P2 / 'public_g19_attribution_v1' / 'mps_cov2_rank0035.npz')
truth15 = np.load(P2 / 'public_g22_operator_features_v1/truthpost/truthpost_rank0035.npz')['mc_post'][15].astype(np.float64)
p0, a0, c0, e0 = s6bcard.payload(torch); sv = torch.load(RF / 's6b_B2_fullnet_lr1e4_trunk5e5/params_epoch1.pt')
for k in p0: p0[k].copy_(sv['p'][k])
a0.copy_(sv['a']); c0.copy_(sv['c'])
for k in e0: e0[k].copy_(sv['e'][k])
T = lambda key, l: torch.as_tensor(np.asarray(TS[key][l], dtype=np.float64))
def run(tag, mu15=False, k15=False, klayers=(), nd15=False, base=False):
    card = s6bcard.card(torch, W, p0, a0, c0, e0)
    if nd15: card.hooks['delta'] = lambda l, d, ctx: d * 0 if l == 15 else d
    if base:
        t0 = time.time(); pred = card.run(False).numpy().astype(np.float64)
        r = dict(tag=tag, mse=float(((pred[15] - truth15) ** 2).mean()), wall=round(time.time() - t0, 1)); print(json.dumps(r), flush=True); return r
    card.hooks['mu'] = lambda l, v, ctx: T('mc_pre_m', l) if (l < 15 or mu15) else v
    card.hooks['cpre'] = lambda l, v, ctx: T('mc_pre_cov', l)
    if mu15: card.hooks['vv15'] = lambda l, v, ctx: T('mc_pre_v', 15)
    if k15:
        rc = card.hooks['k3_15']
        card.hooks['k3_15'] = lambda l, k3, ctx: T('mc_pre_k3', 15)
        card.hooks['k4_15'] = lambda l, k4, ctx: T('mc_pre_k4', 15)
    if klayers:
        def pin(l, v, ctx):
            if l not in klayers: return v
            mu, Cp, k21, k4, k22, r31 = v
            k21 = k21.clone(); k21.diagonal().copy_(T('mc_pre_k3', l))
            return (mu, Cp, k21, T('mc_pre_k4', l), k22, r31)
        card.hooks['pair_in'] = pin
    t0 = time.time(); pred = card.run(False).numpy().astype(np.float64)
    r = dict(tag=tag, mse=float(((pred[15] - truth15) ** 2).mean()), wall=round(time.time() - t0, 1)); print(json.dumps(r), flush=True); return r
res = [run('base+nd15', nd15=True, base=True), run('mC15+nd15', mu15=True, nd15=True), run('k15+nd15', mu15=True, k15=True, nd15=True)]
(HERE / 'mean_path_oracle_nd15.json').write_text(json.dumps(res))
