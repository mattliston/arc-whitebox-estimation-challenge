#!/usr/bin/env python3
"""TAG fable-physicist (OT-5): old-tier diagonal needed on NON-ALIVE rows (gate < 0.01)? Torch card (model_bank_oldtier.py), learned parts
ON: old columns' per-unit diagonal zeroed on non-alive rows only (omitted-radial untouched: it only uses alive rows). If harmless, the
old-column rebuild can be restricted to alive rows (exact at the consumers). usage: ot_alive_ablate.py <ranks> <threads>"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(1, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
import model_bank_oldtier as model
ranks = [int(x) for x in sys.argv[1].split(',')]; torch.set_num_threads(int(sys.argv[2]))
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
base = {35: 1.6484657558908164e-08, 100: 1.683409586118487e-08, 37: 1.8468252790360353e-08, 39: 1.5381853412410622e-08}
p, a, c, e = model.torch_payload(torch, np, False)
for rank in ranks:
    w = np.load(STORE / Path(man[rank]['path']).parent / 'weights.npy')
    TP = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
    model.OT_DIAG_RESP_ONLY, model.OT_OMITTED_OLD_OFF, model.OT_CAPTURE = 'alive', False, None
    card = model.Card(torch, w, p, a, c, e)
    with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
    f = float(((pred[15] - TP[15]) ** 2).mean())
    print(json.dumps(dict(rank=rank, variant='diag_alive_only', final=f, rel=f / base[rank] - 1)), flush=True)
