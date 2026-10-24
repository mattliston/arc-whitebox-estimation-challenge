#!/usr/bin/env python3
"""TAG fable-physicist (PH-4): multi-layer supply test in the torch card (model_bank_phys.py, no refit).
Oracle variants use the 8M Monte Carlo truth planes (public_g19_attribution_v1/mps_planes_rank<r>.npz, layers 2..14) and per-unit
k3/k4 (mps_cov2_rank<r>.npz); model variants are functions of the card's own state (supplies.py). Grades: final-layer MSE vs
truthpost mc_post[15] (the score's target), per-layer post MSE, per-layer pre-activation variance rel-rms error and k3 rel error vs the
32M truth (P32M if present, else P8M).
usage: run_supply.py <ranks> <variants comma list> <threads>
variant grammar: base | t<parts>@<L0> (truth parts from {K21off,k3,k4,K22,K31} joined by '+', layers L0..14, pair-map only) |
                 ...!all (also feed the supplied K21 to the card's other consumers) | m:<name> (model supply from supplies.py)"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(1, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
import model_bank_phys as model
ranks = [int(x) for x in sys.argv[1].split(',')]; variants = sys.argv[2].split(','); torch.set_num_threads(int(sys.argv[3]))
G19 = P2 / 'public_g19_attribution_v1'
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
p, a, c, e = model.torch_payload(torch, np, False)

def truth_supply(rank, parts, L0, allc):
    MP = np.load(G19 / f'mps_planes_rank{rank:04d}.npz'); CV = np.load(G19 / f'mps_cov2_rank{rank:04d}.npz')
    sup = {}
    for l in range(L0, 15):
        d = {}
        if 'K21off' in parts: d['K21off'] = MP[f'K21_{l}']
        if 'K22' in parts: d['K22'] = MP[f'K22_{l}']
        if 'K31' in parts: d['K31'] = MP[f'K31_{l}']
        if 'k3' in parts: d['k3'] = CV['mc_pre_k3'][l]
        if 'k4' in parts: d['k4'] = CV['mc_pre_k4'][l]
        d['all_consumers'] = allc
        sup[l] = d
    return sup

def grade(rank, pred, cap):
    TP = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
    tf = G19 / f'mps_truth_rank{rank:04d}_P32M.npz'
    tf = tf if tf.exists() else G19 / f'mps_truth_rank{rank:04d}_P8M.npz'
    T = np.load(tf) if tf.exists() else None
    r = dict(final=float(((pred[15] - TP[15]) ** 2).mean()), post={l: float(((pred[l] - TP[l]) ** 2).mean()) for l in (5, 8, 11, 13, 14)})
    if T is not None:
        vr, k3r = {}, {}
        for l in range(1, 16):
            d = cap[l]
            v = d['v'].numpy() if l == 15 else np.diag(d['Cpre'].numpy()).astype(np.float64)
            k3 = d['k3'].numpy() if l == 15 else np.diag(d['K21'].numpy()).astype(np.float64)
            tv, tk3 = T['mc_pre_v'][l], T['mc_pre_k3'][l]
            vr[l] = float(np.sqrt(np.mean((v / tv - 1) ** 2))); k3r[l] = float(np.linalg.norm(k3 - tk3) / np.linalg.norm(tk3))
        r['v_rel'] = vr; r['k3_rel'] = k3r
    return r

if __name__ == '__main__':
    import supplies
    for rank in ranks:
        w = np.load(STORE / Path(man[rank]['path']).parent / 'weights.npy')
        for var in variants:
            allc = var.endswith('!all'); vv = var[:-4] if allc else var
            if vv == 'base':
                sup = None
            elif vv.startswith('m:'):
                sup = supplies.make(vv[2:], allc)
            else:
                parts, L0 = vv[1:].split('@'); sup = truth_supply(rank, parts.split('+'), int(L0), allc)
            model.PHYS_SUPPLY = sup; cap = {}; model.PHYS_CAPTURE = cap
            card = model.Card(torch, w, p, a, c, e); t0 = time.time()
            with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
            r = grade(rank, pred, cap); r.update(rank=rank, variant=var, wall=round(time.time() - t0, 1))
            print(json.dumps(r), flush=True)
            del card, cap
