#!/usr/bin/env python3
"""TAG fable-frame (FR-3): where the card's final error is BORN, via the mean channel. After the card's step at layer l, replace its
post-activation mean m by the 1e9 truth (public_g22_operator_features_v1/truthpost mc_post[l]); the card continues unchanged
(learned ON, no refit). ratio(l) = final MSE with the reset at l / card. 1 - ratio(l) = share of the final MSE carried by mean error
inherited through layer l. Also 'mall@L' = reset m at EVERY layer <= L.
usage: mean_oracle_sweep.py <ranks> <layers e.g. 1,2,4,6,8,10,12,13,14> <threads>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
from common import *
import model_bank_research as model
ranks = [int(x) for x in sys.argv[1].split(',')]; layers = [int(x) for x in sys.argv[2].split(',')]; torch.set_num_threads(int(sys.argv[3]))
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]

def run(rank, reset_layers):
    p, a, c, e = model.torch_payload(torch, np, False)
    w = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
    card = model.Card(torch, w, p, a, c, e)
    orig_step = card.step
    def step(layer, state):
        out = orig_step(layer, state)
        if layer in reset_layers:
            out = (torch.as_tensor(truth[layer], dtype=out[0].dtype),) + tuple(out[1:])
        return out
    card.step = step
    with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
    return float(((pred[15] - truth[15]) ** 2).mean())

for rk in ranks:
    t0 = time.time(); base = run(rk, set())
    print(json.dumps(dict(rank=rk, cfg='base', mse=base, ratio=1.0, wall=round(time.time() - t0, 1))), flush=True)
    for l in layers:
        mse = run(rk, {l}); print(json.dumps(dict(rank=rk, cfg=f'm@{l}', mse=mse, ratio=round(mse / base, 4))), flush=True)
    for L in (7, 13):
        mse = run(rk, set(range(1, L + 1))); print(json.dumps(dict(rank=rk, cfg=f'mall@{L}', mse=mse, ratio=round(mse / base, 4))), flush=True)
