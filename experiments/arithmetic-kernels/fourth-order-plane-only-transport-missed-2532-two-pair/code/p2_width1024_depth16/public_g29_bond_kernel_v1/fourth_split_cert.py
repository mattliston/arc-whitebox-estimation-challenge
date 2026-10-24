#!/usr/bin/env python3
"""TAG g29-fourth-split-cert: split the engine's one-step fourth-order plane error between the
EMISSION (pre-layer l truth inputs -> fourth_emission_1024 -> vs truth POST planes at l) and the
PUSH (truth POST planes at l -> fourth_push_1024 with W_{l+1} -> vs truth PRE planes at l+1).
Truth post planes: post_planes_rank{rank}_P4M.npz (mc_post_planes_mps.py); truth pre planes: 8M
mps_planes; univariates 32M.  Usage: fourth_split_cert.py 35 4 8 12"""
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
G19 = HERE.parent / "public_g19_attribution_v1"
sys.path.insert(0, str(G19))
import run_tensor_recursion_v4 as eng

rank = int(sys.argv[1]); layers = [int(a) for a in sys.argv[2:]] or [4, 8, 12]
dev = torch.device("mps")
T = np.load(G19 / f"mps_truth_rank{rank:04d}_P32M.npz"); C8 = np.load(G19 / f"mps_cov2_rank{rank:04d}.npz")
Pl = np.load(G19 / f"mps_planes_rank{rank:04d}.npz"); PP = np.load(HERE / f"post_planes_rank{rank:04d}_P4M.npz")
STORE = G19.parent / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text()); entry = man["roles"]["observer_fit"]["row_manifests"][rank]
weights = np.load((STORE / entry["path"]).parent / "weights.npy", mmap_mode="r")
off = ~np.eye(1024, dtype=bool)


def cmp(est, tr, vec=False):
    e, t = (est, tr) if vec else (est[off], tr[off])
    return {"ratio": float(np.sqrt(np.mean(e**2) / np.mean(t**2))), "corr": float(e @ t / np.sqrt((e @ e) * (t @ t))),
            "relerr": float(np.linalg.norm(e - t) / np.linalg.norm(t))}


out = {}
for l in layers:
    m = T["mc_pre_m"][l].astype(np.float64); v = T["mc_pre_v"][l].astype(np.float64)
    k3 = T["mc_pre_k3"][l].astype(np.float64); k4 = T["mc_pre_k4"][l].astype(np.float64)
    C = C8["mc_pre_cov"][l].astype(np.float64)
    K21 = Pl[f"K21_{l}"].astype(np.float64); K22 = Pl[f"K22_{l}"].astype(np.float64); K31 = Pl[f"K31_{l}"].astype(np.float64)
    # EMISSION: truth pre -> post
    k4post, K22post, K31post = eng.fourth_emission_1024(m, v, k3, k4, C, K21, K22, K31)
    rec = {"emission": {"K22": cmp(K22post, PP[f"K22post_{l}"].astype(np.float64)),
                        "K31": cmp(K31post, PP[f"K31post_{l}"].astype(np.float64)),
                        "k4": cmp(k4post, PP[f"postk4_{l}"].astype(np.float64), vec=True)}}
    # PUSH: truth post -> pre l+1
    Wl = np.asarray(weights[l + 1], dtype=np.float64)
    tK22p = PP[f"K22post_{l}"].astype(np.float64).copy(); tK31p = PP[f"K31post_{l}"].astype(np.float64).copy()
    np.fill_diagonal(tK22p, 0.0); np.fill_diagonal(tK31p, 0.0)
    k4n, K22n, K31n = eng.fourth_push_1024(dev, PP[f"postk4_{l}"].astype(np.float64), tK22p, tK31p, Wl)
    rec["push"] = {"K22": cmp(K22n, Pl[f"K22_{l+1}"].astype(np.float64)), "K31": cmp(K31n, Pl[f"K31_{l+1}"].astype(np.float64)),
                   "k4": cmp(k4n, T["mc_pre_k4"][l + 1].astype(np.float64), vec=True)}
    # the truth post K21 vs emission_K21 for reference (third-order emission)
    K21post = eng.emission_K21(m, v, k3, k4, C, K21, K22, K31)
    rec["emission_K21"] = cmp(K21post, PP[f"K21post_{l}"].astype(np.float64))
    out[str(l)] = rec
    np.savez(HERE / f"push_residual_rank{rank:04d}_L{l}.npz",
             K22_pushed=K22n.astype(np.float32), K31_pushed=K31n.astype(np.float32), k4_pushed=k4n,
             K22_missing=(Pl[f"K22_{l+1}"].astype(np.float64) - K22n).astype(np.float32),
             K31_missing=(Pl[f"K31_{l+1}"].astype(np.float64) - K31n).astype(np.float32),
             k4_missing=T["mc_pre_k4"][l + 1].astype(np.float64) - k4n,
             K22post_em=K22post.astype(np.float32), K31post_em=K31post.astype(np.float32), k4post_em=k4post)
    for stage in ("emission", "push"):
        print(f"L{l} {stage:8s}: " + " | ".join(f"{k}: ratio {r['ratio']:.3f} corr {r['corr']:.3f} relerr {r['relerr']:.3f}" for k, r in rec[stage].items()), flush=True)
    print(f"L{l} emission_K21 (third order, for reference): ratio {rec['emission_K21']['ratio']:.3f} corr {rec['emission_K21']['corr']:.3f} relerr {rec['emission_K21']['relerr']:.3f}")
(HERE / f"FOURTH_SPLIT_rank{rank:04d}.json").write_text(json.dumps(out, indent=1))
