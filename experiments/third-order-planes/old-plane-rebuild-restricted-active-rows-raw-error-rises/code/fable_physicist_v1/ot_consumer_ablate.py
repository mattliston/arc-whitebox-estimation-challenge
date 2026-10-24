#!/usr/bin/env python3
"""TAG fable-physicist (OT-2): can the old-tier rebuild be restricted to the RESPONSE rows? Torch card (model_bank_oldtier.py), learned
parts ON, no refit: base vs (diag) old columns' per-unit diagonal counted on response rows only, (omit) old columns dropped from the
omitted-row radial correction, (both). Also prints per-layer response/alive counts and the old diagonal share on non-response rows.
usage: ot_consumer_ablate.py <ranks> <threads>"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(1, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
import model_bank_oldtier as model
ranks = [int(x) for x in sys.argv[1].split(',')]; torch.set_num_threads(int(sys.argv[2]))
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
p, a, c, e = model.torch_payload(torch, np, False)
for rank in ranks:
    w = np.load(STORE / Path(man[rank]['path']).parent / 'weights.npy')
    TP = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
    base = None
    for name, (dg, om) in dict(base=(False, False), diag=(True, False), omit=(False, True), both=(True, True)).items():
        model.OT_DIAG_RESP_ONLY, model.OT_OMITTED_OLD_OFF = dg, om
        cap = {} if name == 'base' else None; model.OT_CAPTURE = cap
        card = model.Card(torch, w, p, a, c, e)
        with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
        f = float(((pred[15] - TP[15]) ** 2).mean()); base = f if base is None else base
        rec = dict(rank=rank, variant=name, final=f, rel=f / base - 1)
        if cap: rec['layers'] = {l: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()} for l, d in sorted(cap.items())}
        print(json.dumps(rec), flush=True)
