#!/usr/bin/env python3
"""Parity of the fnp in-loop GNN step (estimator_retro._gnn_step) against the torch forward, in two stages because torch and
flopscope live in different interpreters. Stage torch (anaconda python): gnn_parity.py torch <MODEL.pt> <rank> -> parity_torch_<rank>.npz
Stage fnp (.venv python): gnn_parity.py fnp <payload.npz> <rank>  -> prints per-layer max |fnp - torch| / rms and the billed flops."""
import importlib.util, json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
stage, arg, rank = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = HERE / os.environ.get("PARITY_ROWS", "rows_meanleak_mech") / f"row_{rank:04d}"   # T-116g: rows dir
x = np.concatenate([np.load(d / "raw20.npy").astype(np.float32), np.load(d / "mech.npy").astype(np.float32)], axis=2); sigma = np.load(d / "sigma.npy").astype(np.float32)
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]; W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy", mmap_mode="r")
if stage == "torch":
    import torch
    sys.argv = [sys.argv[0], str(HERE / os.environ.get("PARITY_ROWS", "rows_meanleak_mech")), "parity", "1", "32"]
    spec = importlib.util.spec_from_file_location("tg", HERE / "train_gnn_oracle.py"); tg = importlib.util.module_from_spec(spec); sys.modules["tg"] = tg; spec.loader.exec_module(tg)
    sd = torch.load(arg, map_location="cpu"); C = sd["head"].shape[1]; m = tg.GNN(x.shape[2], C); m.load_state_dict(sd, strict=False); m.eval()
    with torch.no_grad():
        dt = m(torch.tensor(x), torch.tensor(sigma), [torch.tensor(np.asarray(W[l], dtype=np.float32)) for l in range(16)]).numpy()
    np.savez(HERE / f"parity_torch_{rank}.npz", delta=dt, rms=sd["rms"].numpy()); print("torch deltas saved")
else:
    os.environ["RETRO_GNN_PAYLOAD"] = arg
    spec2 = importlib.util.spec_from_file_location("rc", P2 / "public_g20_emission_retrofit_v1/estimator_retro.py"); rc = importlib.util.module_from_spec(spec2); sys.modules["rc"] = rc; spec2.loader.exec_module(rc)
    import flopscope as flops
    with flops.BudgetContext(flop_budget=2**42, wall_time_limit_s=600.0, quiet=True) as ctx:
        df = np.stack([np.asarray(rc._gnn_step(l, rc.fnp.asarray(x[l]), rc.fnp.asarray(sigma[l]), rc.fnp.asarray(np.asarray(W[l], dtype=np.float32))), dtype=np.float64) for l in range(16)])
    z = np.load(HERE / f"parity_torch_{rank}.npz"); dt = z["delta"]; rms = z["rms"]
    print("per-layer max |fnp - torch| / rms:", " ".join(f"L{l} {np.max(np.abs(df[l] - dt[l])) / max(float(rms[l]), 1e-30):.1e}" for l in (2, 5, 8, 12, 15)), "| billed flops of the 16 steps:", int(ctx.flops_used), f"({ctx.flops_used / 2**41:.4f} of the budget)")
