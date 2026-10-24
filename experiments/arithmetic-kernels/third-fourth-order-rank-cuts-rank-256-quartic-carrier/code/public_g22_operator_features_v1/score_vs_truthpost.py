#!/usr/bin/env python3
"""TAG g22-score-vs-truthpost (T-116k): score saved instrument outputs (public_g20_emission_retrofit_v1/retro_out_<TAG>_rank<r>.npy,
row 0 = the closed-loop / f2j-off output) against the organizers' 1e9 means (noise ~1.6e-10 per unit) for near-noiseless ranking of rounds.
Usage: score_vs_truthpost.py <TAG> [<TAG> ...]   (nets 35 and 100)"""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; R = HERE.parent / "public_g20_emission_retrofit_v1"
for tag in sys.argv[1:]:
    out = []
    for r in (35, 100):
        f = R / f"retro_out_{tag}_rank{r:04d}.npy"
        if not f.exists(): out.append(f"r{r} missing"); continue
        o = np.load(f).astype(float); o = o[0] if o.ndim == 2 else o
        t = np.load(HERE / "truthpost" / f"truthpost_rank{r:04d}.npz")["mc_post"][15].astype(float)
        out.append(f"r{r} {np.mean((o - t) ** 2):.3e}")
    print(f"{tag}: " + " | ".join(out))
