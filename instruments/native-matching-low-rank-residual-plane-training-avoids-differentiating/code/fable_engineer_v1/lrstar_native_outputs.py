#!/usr/bin/env python3
"""TAG fable-engineer (E-62c): native half of the torch-hook certificate. For every captured star plane A and w = net 37's layer i+1
weights, evaluate the native S18 fragment's _flms_star and _flms_star_diagonal (lrstar2_k48_q1_r32.py in process through flopscope,
_mm = plain matmul) and save them. usage: lrstar_native_outputs.py <planes.npz> <out.npz>"""
import sys, json
from pathlib import Path
import numpy as np
import flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
out = Path(sys.argv[2])
if out.exists():
    raise SystemExit(f"refusing to overwrite {out}")
flops.configure(symmetry_warnings=False)
planes = np.load(sys.argv[1])["planes"]
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W = np.load(STORE / Path(man[37]["path"]).parent / "weights.npy").astype(np.float32)
ns = {"fnp": fnp, "_mm": lambda a, b: a @ b}
exec(compile((HERE / "lrstar2_k48_q1_r32.py").read_text(), "lrstar2_k48_q1_r32.py", "exec"), ns)
stars, diags, ws = [], [], []
for i, A in enumerate(planes):
    w = W[min(i + 1, 15)]
    with flops.BudgetContext(flop_budget=2**46, quiet=True):
        stars.append(np.asarray(ns["_flms_star"](fnp.asarray(w), fnp.asarray(A)), dtype=np.float64))
        diags.append(np.asarray(ns["_flms_star_diagonal"](fnp.asarray(w), fnp.asarray(A)), dtype=np.float64))
    ws.append(w)
np.savez(out, planes=planes, w=np.stack(ws), star=np.stack(stars), diag=np.stack(diags))
print("saved", len(planes), "native star terms to", out)
