#!/usr/bin/env python3
"""Per-layer k3 rel error vs the 32M truth for a list of dumped arms. Usage: k3_table.py <rank> arm1 arm2 ..."""
import sys, os, numpy as np
R = int(sys.argv[1]); arms = [a for a in sys.argv[2:] if os.path.exists(f"k4dump_{a}_retro-{R}.npz")]
G = "../public_g19_attribution_v1/"
T = np.load(G + f"mps_truth_rank{R:04d}_P32M.npz"); st = np.load(G + f"states_rank{R:04d}.npz"); eng = np.load(G + f"tensor_planes_rank{R:04d}_dep.npz")
D = {a: np.load(f"k4dump_{a}_retro-{R}.npz") for a in arms}
print(f"r{R} per-layer k3 rel err vs 32M truth: L | card | engine | " + " | ".join(arms))
for l in range(2, 15):
    kt = T["mc_pre_k3"][l].astype(float); kn = np.linalg.norm(kt); e = lambda x: np.linalg.norm(x - kt) / kn
    row = f"{l:2d} | {e(st['host_k3'][l-1].astype(float)):.4f} | " + (f"{e(eng[f'k3u_{l}'].astype(float)):.4f}" if f"k3u_{l}" in eng.files else "  --  ")
    for a in arms: row += f" | {e(D[a][f'k3_{l}']):.4f}"
    print(row)
