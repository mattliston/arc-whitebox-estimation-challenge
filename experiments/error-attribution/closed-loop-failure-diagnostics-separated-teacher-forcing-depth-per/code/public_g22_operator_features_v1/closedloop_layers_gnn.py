#!/usr/bin/env python3
"""T-116c: per-layer carried-error profile of the LEARNED CLOSURE run closed-loop (payload = GNN npz; "openloop" = the deployed card, no loop). on one net: carried (corrected) means vs the one-step (pre) means vs truth.
Usage: closedloop_layers.py <rank> <payload.npz> <pre_rows_dir>"""
import importlib.util, json, os, sys
from pathlib import Path
import numpy as np, flopscope as flops
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
rank = int(sys.argv[1]); os.environ.update({"RETRO_CLOSED_LOOP": "1", "RETRO_CL_MECH": "1", "RETRO_CL_MECH_FNP": "1", "RETRO_DUMP_FEAT": "1", "RETRO_MEANLEAK_OFF": "1", "RETRO_GNN_PAYLOAD": sys.argv[2]}) if sys.argv[2] != "openloop" else None; os.environ.pop("RETRO_CLOSED_LOOP_ROWS", None)
man = json.loads((STORE / "manifest.json").read_text()); e = man["roles"]["observer_fit"]["row_manifests"][rank]; rmp = STORE / e["path"]; row = json.loads(rmp.read_text())
W = np.load(rmp.parent / "weights.npy", mmap_mode="r"); mlp = MLP(width=1024, depth=16, weights=[W[l] for l in range(16)], seed=int(derive_estimator_seed(int(row["mlp_seed"]))), name=f"cl-{rank}")
spec = importlib.util.spec_from_file_location("retro_cl", P2 / "public_g20_emission_retrofit_v1/estimator_retro.py"); m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
with threadpool_limits(limits=3):
    with flops.BudgetContext(flop_budget=2**42, wall_time_limit_s=900.0, quiet=True):
        host, raw, sig = m.predict_with_native_tape(mlp)
corr = np.asarray(host, dtype=np.float64); pre = np.load(Path(sys.argv[3]) / f"row_{rank:04d}/host.npy").astype(np.float64) if sys.argv[3] != "none" else None
t = np.load(HERE / f"truthpost/truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
print("layer: one-step error -> after the operator's correction (removal)")
print("carried-error profile: " + " ".join(f"L{L} {np.mean((corr[L] - t[L]) ** 2):.2e}" for L in range(1, 16)))
if pre is not None:
    for L in range(2, 16):
        a = np.mean((pre[L] - t[L]) ** 2); b = np.mean((corr[L] - t[L]) ** 2); print(f"  L{L}: {a:.3e} -> {b:.3e} (removal {100*(1-b/a):.0f}%)")
