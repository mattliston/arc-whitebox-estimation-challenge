#!/usr/bin/env python3
"""TAG fable-frame (FR-6): structure of the mean-orthogonal carried-covariance error. Capture the card's post-activation (m, C) at the
mc2 layers (torch card, learned ON), dC = C_card - C_truth(32M), P = I - m m^T/|m|^2, E = offdiag(P dC P). Report: |E| vs the MC
noise floor offdiag(P (A-B)/2 P); SVD energy of E in top-k; single-candidate scalar fits (energy explained) for cheap structures.
usage: dcperp_structure.py <rank> <threads>"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
from common import *
import model_bank_research as model
rank = int(sys.argv[1]); torch.set_num_threads(int(sys.argv[2]))
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
mc2 = np.load(P2 / 'fable_mc_v1' / f'mc2_rank{rank:04d}_P32M.npz')
layers = [4, 7, 10, 13]
p, a, c, e = model.torch_payload(torch, np, False)
w = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
card = model.Card(torch, w, p, a, c, e)
cap = {}
orig_step = card.step
def step(layer, state):
    out = orig_step(layer, state)
    if layer in layers:
        cap[layer] = dict(m=out[0].detach().double().numpy().copy(), C=out[1].detach().double().numpy().copy())
    return out
card.step = step
with torch.no_grad(): card.run(False)
def off(X): return X - np.diag(np.diag(X))
for l in layers:
    m, C = cap[l]['m'], cap[l]['C']
    Ct = mc2[f'cov_post_{l}']; A, B = mc2[f'cov_post_A_{l}'].astype(np.float64), mc2[f'cov_post_B_{l}'].astype(np.float64)
    mh = m / np.linalg.norm(m); P = np.eye(1024) - np.outer(mh, mh)
    E = off(P @ (C - Ct) @ P); N = off(P @ ((A - B) / 2) @ P)
    Cp = off(P @ C @ P); norm = np.linalg.norm(Cp)
    sv = np.linalg.svd(P @ (C - Ct) @ P, compute_uv=False); en = sv ** 2 / (sv ** 2).sum()
    g = np.clip(m / (np.sqrt(np.clip(np.diag(C), 1e-12, None)) + 1e-12), 0, None)
    sd = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    cands = {
        'C': Cp,
        'C2': off(P @ (C @ C) @ P),
        'CoC': off(P @ (C * C) @ P),
        'corr': off(P @ (C / np.outer(sd, sd)) @ P),
        'C3': off(P @ (C @ C @ C) @ P),
    }
    fits = {}
    for k, X in cands.items():
        alpha = float((E * X).sum() / (X * X).sum()); fits[k] = dict(alpha=round(alpha, 5), explained=round(float(1 - ((E - alpha * X) ** 2).sum() / (E ** 2).sum()), 4))
    print(json.dumps(dict(rank=rank, layer=l, rel_err=round(float(np.linalg.norm(E) / norm), 5), noise_rel=round(float(np.linalg.norm(N) / norm), 5),
                          svd_top=dict(k1=round(float(en[:1].sum()), 3), k4=round(float(en[:4].sum()), 3), k16=round(float(en[:16].sum()), 3), k64=round(float(en[:64].sum()), 3)),
                          fits=fits)), flush=True)
