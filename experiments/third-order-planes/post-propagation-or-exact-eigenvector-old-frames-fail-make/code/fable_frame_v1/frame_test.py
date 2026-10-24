#!/usr/bin/env python3
"""TAG fable-frame (FR-1): where the shared old tier compresses. Bank: range finder on the joining cohort's lam^2-weighted column
Gram in POST space (layer l-1, before the push by W_l), then push the frame. Variant 'pre': push first, range finder on the PUSHED
Gram A G A^T in PRE space (where the (2,1) plane and diagonal are read), coordinates via B = A^T V (no dense push of the cohort).
Same one-pass sketch width (rank) in both. Torch research card, learned parts ON, no refit.
usage: frame_test.py <ranks e.g. 35,100> <configs e.g. orig:384,pre:384,orig:256,pre:256> <threads>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; RF = HERE.parent / 'fable_refit_v1'; sys.path.insert(0, str(RF))
import numpy as np, torch
from common import *
import model_bank_research as model
ranks = [int(x) for x in sys.argv[1].split(',')]; configs = sys.argv[2].split(','); torch.set_num_threads(int(sys.argv[3]))
P2 = HERE.parent; STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]

def make_pre_push(orig_push, qr):
    def push(torch, layer, w32, p_full, lam, previous_indices, Q, F, core, cohort_stops, rank=384):
        if layer < 5 or layer > 14:
            return orig_push(torch, layer, w32, p_full, lam, previous_indices, Q, F, core, cohort_stops, rank=rank)
        n = int(w32.shape[0]); wp = w32[previous_indices, :].T          # A: post(active) -> pre
        old_columns = 0 if layer == 5 else cohort_stops[min(layer - 6, 9)]
        Qp = Q[previous_indices, :] if old_columns else None
        stop = cohort_stops[layer - 5]
        J = p_full[previous_indices, old_columns:stop]; weight = lam[old_columns:stop] ** 2
        g = torch.Generator().manual_seed(1000 + layer)
        Omega = torch.randn((n, rank), generator=g, dtype=w32.dtype)   # pre-space sketch
        Z = wp.T @ Omega
        inner = J @ (weight[:, None] * (J.T @ Z))
        if old_columns:
            inner = inner + Qp @ (core @ (Qp.T @ Z))
        V = qr(torch, wp @ inner)                                       # n x rank, orthonormal in PRE space
        B = wp.T @ V                                                    # V^T A x = B^T x
        Fj = B.T @ J; joined = Fj @ (weight[:, None] * Fj.T)
        if old_columns:
            overlap = B.T @ Qp
            Fnew = torch.cat((overlap @ F, Fj), dim=1); corenew = overlap @ (core @ overlap.T) + joined
        else:
            Fnew, corenew = Fj, joined
        U = V
        old = U @ Fnew
        young = wp @ p_full[previous_indices, stop:]
        return torch.cat((old, young), dim=1), U, Fnew, corenew, stop
    return push

def make_eig_push(orig_push, space):
    def push(torch, layer, w32, p_full, lam, previous_indices, Q, F, core, cohort_stops, rank=384):
        if layer < 5 or layer > 14:
            return orig_push(torch, layer, w32, p_full, lam, previous_indices, Q, F, core, cohort_stops, rank=rank)
        wp = w32[previous_indices, :].T
        old_columns = 0 if layer == 5 else cohort_stops[min(layer - 6, 9)]
        Qp = Q[previous_indices, :] if old_columns else None
        stop = cohort_stops[layer - 5]
        J = p_full[previous_indices, old_columns:stop]; weight = lam[old_columns:stop] ** 2
        G = (J.double() * weight.double()[None, :]) @ J.double().T
        if old_columns:
            G = G + Qp.double() @ (core.double() @ Qp.double().T)
        if space == 'post':
            evals, evecs = torch.linalg.eigh(G); Qn = evecs[:, -rank:].to(w32.dtype); B = Qn; U = wp @ Qn
        else:
            A = wp.double(); evals, evecs = torch.linalg.eigh(A @ G @ A.T); V = evecs[:, -rank:]
            B = (A.T @ V).to(w32.dtype); U = V.to(w32.dtype)
        Fj = B.T @ J; joined = Fj @ (weight[:, None] * Fj.T)
        if old_columns:
            overlap = B.T @ Qp
            Fnew = torch.cat((overlap @ F, Fj), dim=1); corenew = overlap @ (core @ overlap.T) + joined
        else:
            Fnew, corenew = Fj, joined
        old = U @ Fnew
        young = wp @ p_full[previous_indices, stop:]
        return torch.cat((old, young), dim=1), U, Fnew, corenew, stop
    return push

def run(rank, cfg):
    kind, r = cfg.split(':'); r = int(r)
    model.TIER_RANK = r
    p, a, c, e = model.torch_payload(torch, np, False)
    w = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
    card = model.Card(torch, w, p, a, c, e)
    if kind in ('eigpost', 'eigpre'):
        card.shared['push'] = make_eig_push(card.shared['push'], kind[3:])
    if kind == 'pre':
        card.shared['push'] = make_pre_push(card.shared['push'], card.shared['_qr'])
    t0 = time.time()
    with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
    return float(((pred[15] - truth[15]) ** 2).mean()), round(time.time() - t0, 1)

for rk in ranks:
    base = None
    for cfg in configs:
        mse, wall = run(rk, cfg)
        if base is None: base = mse
        print(json.dumps(dict(rank=rk, cfg=cfg, mse=mse, ratio=round(mse / base, 4), wall=wall)), flush=True)
