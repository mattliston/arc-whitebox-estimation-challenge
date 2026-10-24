#!/usr/bin/env python3
"""TAG fable-refit (RF-12): how much do the LEARNED parts carry? Torch card forward on given nets with (a) the GNN mean hook output
zeroed (P head/head_b = 0), (b) the covariance edge outputs zeroed (edge output/bias = 0), (c) both. usage: learned_off_test.py <ranks> <threads>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
from common import *
import model_bank_research as model
ranks = [int(x) for x in sys.argv[1].split(',')]; torch.set_num_threads(int(sys.argv[2]))
P2 = HERE.parent; STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
for mode in ('bank', 'gnn_off', 'edge_off', 'both_off'):
    p, a, c, e = model.torch_payload(torch, np, False)
    with torch.no_grad():
        if mode in ('gnn_off', 'both_off'): p['head'].zero_(); p['head_b'].zero_()
        if mode in ('edge_off', 'both_off'): e['output'].zero_(); e['bias'].zero_()
    for rank in ranks:
        w = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
        truth = torch.tensor(np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"][15].astype(np.float64))
        card = model.Card(torch, w, p, a, c, e); t0 = time.time()
        with torch.no_grad(): pred = card.run(False)
        print(json.dumps(dict(mode=mode, rank=rank, mse=float(((pred[15].to(torch.float64) - truth) ** 2).mean()), wall=round(time.time() - t0, 1))), flush=True)
