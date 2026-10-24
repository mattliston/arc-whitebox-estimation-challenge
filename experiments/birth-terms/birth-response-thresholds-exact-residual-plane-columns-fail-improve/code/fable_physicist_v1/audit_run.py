#!/usr/bin/env python3
"""TAG fable-physicist (AU-1): closed-loop price of the card's bookkeeping discards (torch, learned ON, no refit). Variants:
  base | sel<slot>:<thr> (select threshold: slot 0 alive rows, 1 response rows, 2 birth units) | clamp:<u>:<gapmin>:<gapmax>
  (combine with '+'). Prints the final MSE change and per-layer counts (alive / response / births) and birth-clamp activity.
usage: audit_run.py <ranks> <variants> <threads> [--stats]"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(1, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
import model_bank_audit as model
ranks = [int(x) for x in sys.argv[1].split(',')]; variants = sys.argv[2].split(','); torch.set_num_threads(int(sys.argv[3])); want_stats = '--stats' in sys.argv
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
p, a, c, e = model.torch_payload(torch, np, False)
for rank in ranks:
    w = np.load(STORE / Path(man[rank]['path']).parent / 'weights.npy')
    TP = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
    base = None
    for var in variants:
        model.AU_SEL = {}; model.AU_UCLAMP, model.AU_GAPMIN, model.AU_GAPMAX = 8.0, .01, 16.0
        if var != 'base':
            for term in var.split('+'):
                if term.startswith('sel'):
                    slot, thr = term[3:].split(':'); model.AU_SEL[int(slot)] = float(thr)
                elif term.startswith('clamp'):
                    _, u_, gmin, gmax = term.split(':'); model.AU_UCLAMP, model.AU_GAPMIN, model.AU_GAPMAX = float(u_), float(gmin), float(gmax)
        stats = {}; model.AU_STATS = stats
        card = model.Card(torch, w, p, a, c, e); t0 = time.time()
        with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
        model.AU_STATS = None
        mse = float(((pred[15] - TP[15]) ** 2).mean()); base = mse if var == 'base' else base
        cols = sum(d.get('count2', 0) for d in stats.values()) * 2
        rec = dict(rank=rank, variant=var, mse=mse, rel=round(mse / base - 1, 5) if base else None, born_columns_total=cols,
                   resp_rows_sum=sum(d.get('count1', 0) for d in stats.values()), alive_rows_sum=sum(d.get('count0', 0) for d in stats.values()), wall=round(time.time() - t0, 1))
        if want_stats: rec['layers'] = {l: stats[l] for l in sorted(stats)}
        print(json.dumps(rec), flush=True)
