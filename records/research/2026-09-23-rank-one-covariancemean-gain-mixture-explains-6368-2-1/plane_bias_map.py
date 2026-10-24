#!/usr/bin/env python3
"""TAG fable-refit (RF-14): map the card's (2,1)-plane error structure vs truth planes on several nets and all layers. Captures (mu, Cpre,
K21, radial) at every pair-map call of the research torch card; truth = hub_planes_rank*.npz 'pre_l' (convention checked against
mps_planes K21_l on r35); fits the off-diagonal error E = truth - card on F = Sym(C,mu)_(2,1) = v_a mu_c + 2 C_ac mu_a (one scalar kappa
per layer) and on the card's own b-term form; reports kappa, R2(error), relative errors before/after, top-singular share, and radial_l/6.
usage: plane_bias_map.py <ranks e.g. 35,100,150> <truth_kind honest|plain> <threads>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
from common import *
import model_bank_research as model
ranks = [int(x) for x in sys.argv[1].split(',')]; kind = sys.argv[2]; torch.set_num_threads(int(sys.argv[3]))
P2 = HERE.parent; STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
p, a, c, e = model.torch_payload(torch, np, False)
off = ~np.eye(1024, dtype=bool)
out = {}
for rank in ranks:
    fn = P2 / "public_g19_attribution_v1" / (f"hub_planes_rank{rank:04d}_honest.npz" if kind == 'honest' else f"hub_planes_rank{rank:04d}.npz")
    if not fn.exists(): print(json.dumps(dict(rank=rank, missing=str(fn)))); continue
    Tz = np.load(fn)
    w = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    card = model.Card(torch, w, p, a, c, e); cap = {}
    orig = card.env['_nonlinear_pair_batch']
    def pair(mu, C, K21, k4, k22_override=None, metric='identity', card=card, cap=cap):
        l = card.call_layer
        cap[l] = dict(mu=mu.detach().numpy().copy(), C=C.detach().numpy().astype(np.float64), K21=K21.detach().numpy().astype(np.float64), k4=k4.detach().numpy().copy())
        return orig(mu, C, K21, k4, k22_override, metric)
    card.env['_nonlinear_pair_batch'] = pair
    with torch.no_grad(): card.run(False)
    for l in sorted(cap):
        key = f"pre_{l}"
        if key not in Tz.files: continue
        Kt = Tz[key].astype(np.float64); d = cap[l]; K, C, mu = d['K21'], d['C'], d['mu']; v = np.diag(C)
        E = (Kt - K)[off]; Ko = Kt[off]
        F = (np.outer(v, mu) + 2 * C * mu[:, None])[off]
        kap = float(F @ E / (F @ F)); r2 = 1 - float(((E - kap * F) ** 2).sum() / (E ** 2).sum())
        rel0 = float(np.linalg.norm(E) / np.linalg.norm(Ko)); rel1 = float(np.linalg.norm(E - kap * F) / np.linalg.norm(Ko))
        Ef = (Kt - K).copy(); Ef[~off] = 0; S = np.linalg.svd(Ef, compute_uv=False); top1 = float(S[0] ** 2 / (S ** 2).sum())
        slope = float(K[off] @ Kt[off] / (K[off] @ K[off]))
        # per-unit k3 diagonal error vs 3 v mu form
        e3 = np.diag(Kt) - np.diag(K); f3 = 3 * v * mu; k3fit = float(f3 @ e3 / (f3 @ f3)); r2_3 = 1 - float(((e3 - k3fit * f3) ** 2).sum() / (e3 ** 2).sum())
        rec = dict(rank=rank, layer=l, kappa=kap, r2=round(r2, 3), rel_before=round(rel0, 3), rel_after=round(rel1, 3), top1=round(top1, 3), truth_card_slope=round(slope, 3),
                   diag_kappa=k3fit, diag_r2=round(r2_3, 3), diag_rel_before=round(float(np.linalg.norm(e3) / np.linalg.norm(np.diag(Kt))), 3))
        out[f"{rank}:{l}"] = rec; print(json.dumps(rec), flush=True)
json.dump(out, open(HERE / f"plane_bias_map_{kind}.json", "w"), indent=1)
