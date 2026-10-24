#!/usr/bin/env python3
"""TAG fable-physicist (AU-3): partial-strength grading of the card's lossy bookkeeping (torch, learned ON, no refit), per covhunter's
overshoot finding. Variants (combine with '+'):
  star:<eps>      fraction eps of each layer's star residual carried as exact CP columns (4 per alive row), 1-eps as the plane star;
                  run with the frame OFF (both base and variant: FIRST_JOIN past the end) -> compared with 'noframe'
  birth:<eps>     birth units with .001 <= min(g,1-g) < .01 added at weight eps (born plane, learned increment and columns scaled)
  resp:<eps>      response rows .001..0.01 added with their read plane scaled by eps
  alive:<eps>     alive rows with .001 <= gate < .01 added at mask weight eps
  gain:<gamma>    b-partition gain fix b_post += gamma (radial_new - radial)/6 m (moves gain content from the plane star to b)
usage: audit_blend.py <ranks> <variants> <threads>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(1, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
import model_bank_audit as model
ranks = [int(x) for x in sys.argv[1].split(',')]; variants = sys.argv[2].split(','); torch.set_num_threads(int(sys.argv[3]))
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
p, a, c, e = model.torch_payload(torch, np, False)
for rank in ranks:
    w = np.load(STORE / Path(man[rank]['path']).parent / 'weights.npy')
    TP = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
    refs = {}
    for var in variants:
        model.AU_STAR_COLS, model.AU_STAR_EPS, model.AU_SEL_BLEND = False, 1.0, {}
        model.BFIX_MODE, model.BFIX_GAMMA = 'partition', 0.0
        noframe = False
        if var not in ('base', 'noframe'):
            for term in var.split('+'):
                k, v = term.split(':'); v = float(v)
                if k == 'star': model.AU_STAR_COLS, model.AU_STAR_EPS = True, v; noframe = True
                elif k == 'birth': model.AU_SEL_BLEND[2] = (.001, v)
                elif k == 'resp': model.AU_SEL_BLEND[1] = (.001, v)
                elif k == 'alive': model.AU_SEL_BLEND[0] = (.001, v)
                elif k == 'gain': model.BFIX_GAMMA = v
        if var == 'noframe': noframe = True
        card = model.Card(torch, w, p, a, c, e)
        if noframe: card.shared['FIRST_JOIN'] = 99
        t0 = time.time()
        with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
        mse = float(((pred[15] - TP[15]) ** 2).mean())
        if var in ('base', 'noframe'): refs[var] = mse
        ref = refs.get('noframe' if noframe else 'base')
        print(json.dumps(dict(rank=rank, variant=var, mse=mse, rel=round(mse / ref - 1, 5) if ref else None, ref='noframe' if noframe else 'base', wall=round(time.time() - t0, 1))), flush=True)
