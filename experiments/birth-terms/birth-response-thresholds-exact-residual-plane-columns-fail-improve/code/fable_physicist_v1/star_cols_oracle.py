#!/usr/bin/env python3
"""TAG fable-physicist (AU-2): how much third-order information does the STAR discard each layer? The star pushes the plane-supported
residual exactly but keeps only the (2,1) slice (and diagonal) of the pushed tensor; the distinct-index part created by the push is
lost every layer. Oracle: represent each layer's residual tensor by exact CP columns (4 per alive row, model_bank_audit AU_STAR_COLS)
so it is carried like births. Both arms without the old-tier frame (FIRST_JOIN past the end) to isolate the star from the join
truncation. usage: star_cols_oracle.py <ranks> <threads>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(1, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
import model_bank_audit as model
ranks = [int(x) for x in sys.argv[1].split(',')]; torch.set_num_threads(int(sys.argv[2]))
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
p, a, c, e = model.torch_payload(torch, np, False)
for rank in ranks:
    w = np.load(STORE / Path(man[rank]['path']).parent / 'weights.npy')
    TP = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
    base = None
    for name in ('noframe', 'noframe+starcols'):
        model.AU_STAR_COLS = (name == 'noframe+starcols')
        card = model.Card(torch, w, p, a, c, e); card.shared['FIRST_JOIN'] = 99
        t0 = time.time()
        with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
        mse = float(((pred[15] - TP[15]) ** 2).mean()); base = mse if base is None else base
        print(json.dumps(dict(rank=rank, variant=name, mse=mse, rel_vs_noframe=round(mse / base - 1, 5), wall=round(time.time() - t0, 1))), flush=True)
    model.AU_STAR_COLS = False
