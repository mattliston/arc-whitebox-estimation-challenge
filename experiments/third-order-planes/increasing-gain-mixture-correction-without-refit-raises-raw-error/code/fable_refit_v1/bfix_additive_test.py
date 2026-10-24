#!/usr/bin/env python3
"""TAG fable-refit (RF-16): ADDITIVE gain-mixture term. The partition fix (b_post += gamma*dradial/6*m, star compensates) left the rank-1
Sym(C,mu) plane error unchanged (kappa L6 2.26e-3 -> 2.16e-3) and hurt even the physics-only card: re-partitioning is not the defect;
the TOTAL post-activation K3 lacks the gain-mixture term. Additive mode carries b_carry = b + gamma*dradial/6*m to the next layer while
the post-activation partition (trace_post, innovation, star) uses the old b, so the next layer's K21 gains exactly
gamma*dradial/6*Sym(C,mu)-transported. Reports final MSE (learned parts on / off) and the rank-1 plane residual at L4-L9 on rank 35.
usage: bfix_additive_test.py <ranks> <gammas> <threads> [--plane]"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
from common import *
import model_bank_bfix as model
ranks = [int(x) for x in sys.argv[1].split(',')]; gammas = [float(x) for x in sys.argv[2].split(',')]; torch.set_num_threads(int(sys.argv[3])); PLANE = '--plane' in sys.argv
model.BFIX_MODE = 'additive'
P2 = HERE.parent; STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
off = ~np.eye(1024, dtype=bool)
for learned in ('on', 'off'):
    p, a, c, e = model.torch_payload(torch, np, False)
    if learned == 'off':
        with torch.no_grad(): p['head'].zero_(); p['head_b'].zero_(); e['output'].zero_(); e['bias'].zero_()
    for gamma in [0.0] + gammas:
        model.BFIX_GAMMA = gamma
        for rank in ranks:
            w = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
            truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
            card = model.Card(torch, w, p, a, c, e); cap = {}
            if PLANE and rank == 35 and learned == 'on':
                orig = card.env['_nonlinear_pair_batch']
                def pair(mu, C, K21, k4, k22_override=None, metric='identity', card=card, cap=cap):
                    if card.call_layer in (4, 6, 8, 9, 12, 13): cap[card.call_layer] = (mu.detach().numpy().copy(), C.detach().numpy().astype(np.float64), K21.detach().numpy().astype(np.float64))
                    return orig(mu, C, K21, k4, k22_override, metric)
                card.env['_nonlinear_pair_batch'] = pair
            t0 = time.time()
            with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
            rec = dict(learned=learned, gamma=gamma, rank=rank, mse=float(((pred[15] - truth[15]) ** 2).mean()), L8=float(((pred[8] - truth[8]) ** 2).mean()), L12=float(((pred[12] - truth[12]) ** 2).mean()), wall=round(time.time() - t0, 1))
            if cap:
                Tz = np.load(P2 / "public_g19_attribution_v1/hub_planes_rank0035_honest.npz"); pl = {}
                for l, (mu, C, K) in cap.items():
                    Kt = Tz[f'pre_{l}'].astype(np.float64); E = (Kt - K)[off]; v = np.diag(C); F = (np.outer(v, mu) + 2 * C * mu[:, None])[off]
                    pl[l] = dict(rel=round(float(np.linalg.norm(E) / np.linalg.norm(Kt[off])), 3), kappa=float(F @ E / (F @ F)))
                rec['plane'] = pl
            print(json.dumps(rec), flush=True)
