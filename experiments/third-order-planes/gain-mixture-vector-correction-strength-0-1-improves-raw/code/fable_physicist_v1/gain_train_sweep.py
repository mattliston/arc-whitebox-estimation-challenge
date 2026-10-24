#!/usr/bin/env python3
"""TAG fable-physicist (AU-4): TRAIN-only choice of the zero-cost b-partition gain strength (b_post += gamma (radial_new - radial)/6 m,
model_bank_audit BFIX_MODE partition), and its additivity with the K31 row factor (0.8, 0.5, L2-14). Torch, learned ON, no refit.
usage: gain_train_sweep.py <train ranks> <gammas> <threads> <out.jsonl>"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(1, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
import model_bank_audit as model
ranks = [int(x) for x in sys.argv[1].split(',')]; gammas = [float(x) for x in sys.argv[2].split(',')]; torch.set_num_threads(int(sys.argv[3])); out = Path(sys.argv[4])
train = set(json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())['train'])
assert all(r in train for r in ranks), 'TRAIN ranks only'
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
p, a, c, e = model.torch_payload(torch, np, False)
def k31row(card, layer, mu, Cpre, vv, radial, k21, k4, k22, parts):
    t = card.t; s = vv.sqrt(); eye = t.eye(vv.shape[0], dtype=t.bool); k3 = t.diagonal(k21)
    R = (radial * ((0.8 - 1) + 0.5 * mu * k3 / (vv * vv)))[:, None] * (Cpre / (s[:, None] * s[None, :]))
    return k21, k4, k22, R.masked_fill(eye, 0.0).to(t.float32)
configs = [('base', 0.0, False)] + [(f'gain:{g}', g, False) for g in gammas] + [('k31', 0.0, True)] + [(f'k31+gain:{g}', g, True) for g in gammas]
with out.open('a') as fo:
    for rank in ranks:
        w = np.load(STORE / Path(man[rank]['path']).parent / 'weights.npy')
        TP = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
        for name, gam, use31 in configs:
            model.BFIX_MODE, model.BFIX_GAMMA = 'partition', gam
            model.PHYS_SUPPLY = {l: {'fn': k31row} for l in range(2, 15)} if use31 else None
            card = model.Card(torch, w, p, a, c, e)
            with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
            fo.write(json.dumps(dict(rank=rank, cfg=name, mse=float(((pred[15] - TP[15]) ** 2).mean()))) + '\n'); fo.flush()
        model.PHYS_SUPPLY = None
