#!/usr/bin/env python3
"""TAG g22-cand-vs-instrument (T-116k): the packaged candidate scored 2.857e-8 on r35 where the instrument's loop gave 2.537e-8 (same
payload). Record the candidate's per-layer applied corrections by wrapping its _gnn_step and compare with the instrument's dumped
cl_delta_l (featdump_retro-35.npz from the HF1 run). Usage: cand_vs_instrument.py <cand.py> <rank> <instrument_dump.npz>"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, flopscope as flops
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, rank, dump = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
man = json.loads((STORE / "manifest.json").read_text()); e = man["roles"]["observer_fit"]["row_manifests"][rank]; rmp = STORE / e["path"]; row = json.loads(rmp.read_text())
W = np.load(rmp.parent / "weights.npy", mmap_mode="r"); mlp = MLP(width=1024, depth=16, weights=[W[l] for l in range(16)], seed=int(derive_estimator_seed(int(row["mlp_seed"]))), name=f"cvi-{rank}")
spec = importlib.util.spec_from_file_location("cand", cand); m = importlib.util.module_from_spec(spec); sys.modules["cand"] = m; spec.loader.exec_module(m)
rec = {}; orig = m._gnn_step; mech_prev = {}
def wrapped(layer_index, raw_row, sigma_row, layer_weights):
    mech_prev[layer_index] = {k: np.asarray(m._MECH[k], dtype=np.float64).copy() for k in ('post_var', 'post_k3', 'post_k4', 'gate', 'pre_v', 'pre_m', 'pre_k3', 'cov_post_diag', 'cov_post_row') if k in m._MECH}   # T-130: the previous layer's post state as the candidate stashed it
    out = orig(layer_index, raw_row, sigma_row, layer_weights); rec[layer_index] = (np.asarray(raw_row, dtype=np.float64).copy(), np.asarray(out, dtype=np.float64).copy()); return out
m._gnn_step = wrapped
with threadpool_limits(limits=4):
    with flops.BudgetContext(flop_budget=2**41, wall_time_limit_s=1200.0, quiet=True) as ctx:
        out = m._predict_corrected(mlp)
z = np.load(dump)
print("layer: corr(cand delta, instrument delta), rms ratio | input max rel diff (cols)")
for l in sorted(rec):
    x, d = rec[l]
    if f"cl_delta_{l}" in z.files:
        di = z[f"cl_delta_{l}"].astype(np.float64); xi = z[f"cl_x_{l}"].astype(np.float64) if f"cl_x_{l}" in z.files else None
        s = f"L{l}: corr {np.corrcoef(d, di)[0,1]:+.4f} rms ratio {np.sqrt(np.mean(d**2))/max(np.sqrt(np.mean(di**2)),1e-30):.4f}"
        if xi is not None and xi.shape == x.shape:
            rel = np.abs(x - xi).max(0) / (np.abs(xi).max(0) + 1e-12); bad = [int(c) for c in np.where(rel > 1e-4)[0]]
            s += f" | input cols differing > 1e-4 rel: {bad} (max {rel.max():.1e})"
        print(s)
    else:
        print(f"L{l}: candidate applied a step (rms {np.sqrt(np.mean(d**2)):.2e}) but the instrument dump has no cl_delta_{l}")
t = np.load(HERE / "truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
o = np.asarray(out, dtype=np.float64); np.save(HERE / f"cvi_out_{cand.stem}_{rank}.npy", o); np.savez(HERE / f"cvi_x_{cand.stem}_{rank}.npz", **{f"x_{l}": rec[l][0] for l in rec}, **{f"d_{l}": rec[l][1] for l in rec}, **{f"{k}_prev_{l}": v for l in mech_prev for k, v in mech_prev[l].items()})   # T-130: the candidate's step inputs
of = o[15] if o.ndim == 2 else o
print(f"candidate final row vs 1e9 truth: {np.mean((of - t[15])**2):.3e}; saved cvi_out_*.npy")
