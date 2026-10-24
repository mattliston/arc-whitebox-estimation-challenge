#!/usr/bin/env python3
"""Per-layer certificate + supply build: one-step k3 at l+1 from (truth | card) inputs at l, by class
(D, D+P, D+P+T_path at NB), vs truth k3 at l+1 (rel error; truth noise floor). Card-input k3 saved as
tensor_planes_rank{rank}_k3gen{tag}.npz (k3u_{l+1}) for consumption grading. Usage: run_k3gen_layers.py <rank> <truth|card> [NB]"""
import json, os, sys, time
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from k3gen import k3_next, G19
rank = int(sys.argv[1]); src = sys.argv[2]; NB = int(sys.argv[3]) if len(sys.argv) > 3 else 6
st = np.load(G19 / f"states_rank{rank:04d}.npz"); z = np.load(G19 / f"host_cov_rank{rank:04d}_base.npz")
T = np.load(G19 / f"mps_cov2_rank{rank:04d}.npz"); Pl = np.load(G19 / f"mps_planes_rank{rank:04d}.npz"); P = float(T["P"])
STORE = HERE.parent / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text()); entry = man["roles"]["observer_fit"]["row_manifests"][rank]
W = np.load((STORE / entry["path"]).parent / "weights.npy", mmap_mode="r")
eng = np.load(G19 / f"tensor_planes_rank{rank:04d}_dep.npz")
rows = {}; K3U = {}; t0 = time.time()
for l in range(1, 15):
    i = l - 1
    if src in ("truth", "truthplus"):
        m, v = T["mc_pre_m"][l].astype(float), T["mc_pre_v"][l].astype(float)
        k3, k4 = T["mc_pre_k3"][l].astype(float), T["mc_pre_k4"][l].astype(float)
        C = T["mc_pre_cov"][l].astype(float)
        if l in Pl["layers"]:
            K21 = Pl[f"K21_{l}"].astype(float).copy(); K22 = Pl[f"K22_{l}"].astype(float); K31 = Pl[f"K31_{l}"].astype(float)
        else:
            K21 = z["k21"][i].astype(float).copy(); K22 = z["k22"][i].astype(float); K31 = z["k31"][i].astype(float)
    else:
        m, v = st["host_m"][i].astype(float), st["host_v"][i].astype(float)
        k3, k4 = st["host_k3"][i].astype(float), st["host_k4"][i].astype(float)
        C = z["cov"][i].astype(float); K21 = z["k21"][i].astype(float).copy(); K22 = z["k22"][i].astype(float); K31 = z["k31"][i].astype(float)
    np.fill_diagonal(K21, k3)
    Wd = np.asarray(W[l + 1], dtype=float)
    kt = T["mc_pre_k3"][l + 1].astype(float); vt = T["mc_pre_v"][l + 1].astype(float)
    floor = np.linalg.norm(np.sqrt(6 * vt ** 3 / P)) / np.linalg.norm(kt)
    if src in ("cardplus", "truthplus"):
        # increment mode: the card's own k3 at l+1 plus the born PATH words of bond order >= 3 (the card carries (1,1))
        Tinc, comp = k3_next(m, v, C, k3, k4, K21, K22, K31, Wd, nb=NB, parts=("T",), nb_min=3)
        base_k3 = st["host_k3"][l].astype(float)
        full = base_k3 + Tinc
        comp = {"D": base_k3, "P": np.zeros_like(base_k3), "T": Tinc}
    else:
        full, comp = k3_next(m, v, C, k3, k4, K21, K22, K31, Wd, nb=NB)
    D = comp["D"]; DP = comp["D"] + comp["P"]
    rel = lambda x: round(float(np.linalg.norm(x - kt) / np.linalg.norm(kt)), 4)
    row = {"floor": round(float(floor), 4), "D": rel(D), "D+P": rel(DP), "D+P+Tpath": rel(full),
           "T_share_of_k3": round(float(np.linalg.norm(comp["T"]) / np.linalg.norm(kt)), 4),
           "card": rel(st["host_k3"][l].astype(float)),
           "engine_dep": (rel(eng[f"k3u_{l + 1}"].astype(float)) if f"k3u_{l + 1}" in eng.files else None)}
    rows[l + 1] = row; K3U[f"k3u_{l + 1}"] = full.astype(np.float32)
    print(l + 1, json.dumps(row), round(time.time() - t0, 1), "s", flush=True)
tag = f"_{src}_nb{NB}"
(HERE / f"K3GEN_LAYERS_rank{rank:04d}{tag}.json").write_text(json.dumps(rows, indent=1) + "\n")
np.savez(G19 / f"tensor_planes_rank{rank:04d}_k3gen{tag}.npz", **K3U)
print("saved", f"tensor_planes_rank{rank:04d}_k3gen{tag}.npz")
