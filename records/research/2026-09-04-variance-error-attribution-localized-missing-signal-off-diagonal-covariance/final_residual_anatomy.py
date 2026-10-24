#!/usr/bin/env python3
"""TAG g22-final-residual-anatomy (T-115e): where does the learned closure's final-layer residual live? Runs the definitive
model on valid nets (CPU), then: share of the L15 residual energy by decile of |residual|; correlation of |residual| with the gate,
sigma, m/sigma, |target|, and each mech column; the same at L14 as a control."""
import importlib.util, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.argv = [sys.argv[0], str(HERE / "rows_meanleak_mech15"), "anat", "1", "64"]
spec = importlib.util.spec_from_file_location("tg", HERE / "train_gnn_oracle.py"); tg = importlib.util.module_from_spec(spec); sys.modules["tg"] = tg; spec.loader.exec_module(tg)
tg.dev = torch.device("cpu")
nets = [n for n in tg.load_rows() if n["valid"]][:8]
sd = torch.load(HERE / "GNN_GNN_M15_C64_E40_MODEL.pt", map_location="cpu"); m = tg.GNN(nets[0]["x"].shape[2], 64); m.load_state_dict(sd, strict=False); m.eval()
names = ["m/s", "k3/v^1.5", "v", "log win", "cp3_diag", "age0", "age1", "age2", "age3", "older", "star_diag", "energy", "k21_row", "k21_col", "cp4_d", "cp4_k31_row", "cp4_k22_row", "pre_k4", "cp4_inh", "cp4_birth", "cp4_drop", "k22_row", "k31_row", "win", "gW", "gW2", "g2W2", "W3pk3", "W4pk4", "W2pv", "W3pk3g"]
for L in (15, 14):
    R, T, X, S = [], [], [], []
    for n in nets:
        W = np.load(n["wpath"], mmap_mode="r")
        with torch.no_grad():
            p = m(torch.tensor(n["x"]), torch.tensor(n["sigma"]), [torch.tensor(np.asarray(W[l], dtype=np.float32)) for l in range(16)]).numpy()
        R.append(n["target"][L] - p[L]); T.append(n["target"][L]); X.append(n["x"][L]); S.append(n["sigma"][L])
    r = np.concatenate(R); t = np.concatenate(T); x = np.concatenate(X); s = np.concatenate(S)
    e = r ** 2; order = np.argsort(-e); cum = np.cumsum(e[order]) / e.sum()
    print(f"L{L}: removal {100*(1-e.mean()/np.mean(t**2)):.1f}% | residual energy share: top 1% units {100*cum[len(e)//100]:.0f}%, top 10% {100*cum[len(e)//10]:.0f}%, top 50% {100*cum[len(e)//2]:.0f}%")
    def c(a, b): return np.corrcoef(a, b)[0, 1]
    print(f"   corr(|res|, .): |target| {c(np.abs(r), np.abs(t)):+.2f}  sigma {c(np.abs(r), s):+.2f}  gate {c(np.abs(r), x[:, 5]):+.2f}  m/s {c(np.abs(r), x[:, 20]):+.2f}  |m/s| {c(np.abs(r), np.abs(x[:, 20])):+.2f}")
    cc = sorted([(abs(c(np.abs(r), x[:, 20 + i])), names[i], c(np.abs(r), x[:, 20 + i])) for i in range(31) if np.std(x[:, 20 + i]) > 0], reverse=True)[:6]
    print("   strongest mech correlates of |res|:", ", ".join(f"{nm} {v:+.2f}" for _, nm, v in cc))
    # is the residual a smooth function of the pre-state? fit |res| quantile buckets by gate
    g = x[:, 5]; qs = np.quantile(g, [0, .1, .3, .5, .7, .9, 1])
    print("   residual rms by gate decile-ish buckets:", " ".join(f"[{qs[i]:.2f},{qs[i+1]:.2f}] {np.sqrt(np.mean(e[(g>=qs[i])&(g<=qs[i+1])])):.1e}" for i in range(6)))
