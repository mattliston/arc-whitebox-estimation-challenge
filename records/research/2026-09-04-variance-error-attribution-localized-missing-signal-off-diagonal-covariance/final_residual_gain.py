#!/usr/bin/env python3
"""TAG g22-final-residual-gain (T-115f): is the final-layer residual a per-NET scale the model cannot see? Per-net slope of
target on prediction at L15/L14 on valid nets, and the removal after an oracle per-net gain (upper bound for net-level context)."""
import importlib.util, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.argv = [sys.argv[0], str(HERE / "rows_meanleak_mech15"), "anat", "1", "64"]
spec = importlib.util.spec_from_file_location("tg", HERE / "train_gnn_oracle.py"); tg = importlib.util.module_from_spec(spec); sys.modules["tg"] = tg; spec.loader.exec_module(tg)
tg.dev = torch.device("cpu")
nets = [n for n in tg.load_rows() if n["valid"]][:12]
sd = torch.load(HERE / "GNN_GNN_M15_C64_E40_MODEL.pt", map_location="cpu"); m = tg.GNN(nets[0]["x"].shape[2], 64); m.load_state_dict(sd, strict=False); m.eval()
P = {}
for n in nets:
    W = np.load(n["wpath"], mmap_mode="r")
    with torch.no_grad():
        P[n["rank"]] = m(torch.tensor(n["x"]), torch.tensor(n["sigma"]), [torch.tensor(np.asarray(W[l], dtype=np.float32)) for l in range(16)]).numpy()
for L in (15, 14, 12):
    e0 = e1 = e2 = t2 = 0.0; slopes = []
    for n in nets:
        p = P[n["rank"]][L].astype(np.float64); t = n["target"][L].astype(np.float64)
        g = (p @ t) / (p @ p); slopes.append(g)
        # oracle per-net gain, and oracle per-net gain + the model's own layer-14 correction rms as a second regressor
        A = np.stack([p, p * np.abs(P[n["rank"]][L - 1])], axis=1); w = np.linalg.lstsq(A, t, rcond=None)[0]
        e0 += np.mean((t - p) ** 2); e1 += np.mean((t - g * p) ** 2); e2 += np.mean((t - A @ w) ** 2); t2 += np.mean(t ** 2)
    print(f"L{L}: removal {100*(1-e0/t2):.1f}% | per-net slope of target on prediction: mean {np.mean(slopes):.3f} sd {np.std(slopes):.3f} range [{min(slopes):.2f},{max(slopes):.2f}] | oracle per-net gain -> {100*(1-e1/t2):.1f}% | + per-net gain x |own delta_{L-1}| -> {100*(1-e2/t2):.1f}%")
