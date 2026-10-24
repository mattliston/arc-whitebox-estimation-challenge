#!/usr/bin/env python3
"""TAG fable-refit (RF-15): GAIN-MIXTURE b FIX. The card stores the Sym(C x b) third-cumulant family in b (exactly transported) and
the isotropic fourth cumulant as radial. For a common gain fluctuation, K3 contains (radial/6) Sym(C x mean); the card's recursion
b_post = g*bpre + (radial/6)*sigma*phi only adds the kink part of each layer's radial increment, so b stalls at ~1/3 of radial/6
and the rest is carried by the approximately-transported star (measured: rank-1 Sym(C,mu) plane error, 56-64% of error energy at L12-14).
Fix: b_post += gamma * (radial_new - radial)/6 * m (post mean). Post-activation planes unchanged (the star/newborns re-partition).
usage: bfix_test.py <ranks> <gammas e.g. 0.5,0.85,1.0> <mean m|host> <threads>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
from common import *
import model_bank_bfix as model
ranks = [int(x) for x in sys.argv[1].split(',')]; gammas = [float(x) for x in sys.argv[2].split(',')]; model.BFIX_MEAN = sys.argv[3]; torch.set_num_threads(int(sys.argv[4]))
P2 = HERE.parent; STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
p, a, c, e = model.torch_payload(torch, np, False)
for gamma in [0.0] + gammas:
    model.BFIX_GAMMA = gamma
    for rank in ranks:
        w = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
        truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
        card = model.Card(torch, w, p, a, c, e); t0 = time.time()
        with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
        per = [float(((pred[l] - truth[l]) ** 2).mean()) for l in (8, 11, 13, 15)]
        print(json.dumps(dict(gamma=gamma, mean=model.BFIX_MEAN, rank=rank, mse=per[-1], L8=per[0], L11=per[1], L13=per[2], wall=round(time.time() - t0, 1))), flush=True)
