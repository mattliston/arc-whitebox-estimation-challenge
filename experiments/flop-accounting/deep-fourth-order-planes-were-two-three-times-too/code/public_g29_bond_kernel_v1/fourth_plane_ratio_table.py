#!/usr/bin/env python3
"""TAG g29-fourth-plane-ratio-table: per-layer scale of the CARD's fourth-order pre-planes
(K22, K31 from host_cov_rank*_base.npz, index l-1 <-> pre-layer l) against the 8M truth planes
(mps_planes_rank*.npz K22_l / K31_l), off-diagonal entries: rms ratio card/truth, corr, and the
best single scale beta = <card, truth>/<card, card>; two nets side by side (is the ratio universal
like the univariate k4's 0.99 -> 0.40?).  Usage: fourth_plane_ratio_table.py 35 100"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
G19 = HERE.parent / "public_g19_attribution_v1"
ranks = [int(a) for a in sys.argv[1:]] or [35, 100]
out = {}
off = ~np.eye(1024, dtype=bool)
print("plane L   " + "   ".join(f"r{r}: ratio  corr  beta(truth on card)" for r in ranks))
for name in ("K22", "K31"):
    for L in (2, 4, 6, 8, 10, 12, 14):
        row = []
        for r in ranks:
            z = np.load(G19 / f"host_cov_rank{r:04d}_base.npz", mmap_mode="r")
            Pl = np.load(G19 / f"mps_planes_rank{r:04d}.npz", mmap_mode="r")
            card = np.asarray(z[name.lower()][L - 1], dtype=np.float64)[off]
            truth = np.asarray(Pl[f"{name}_{L}"], dtype=np.float64)[off]
            ratio = float(np.sqrt(np.mean(card ** 2) / np.mean(truth ** 2)))
            corr = float(card @ truth / np.sqrt((card @ card) * (truth @ truth)))
            beta = float(card @ truth / (card @ card))
            row.append((ratio, corr, beta))
            out[f"{name}_{L}_r{r}"] = {"ratio": ratio, "corr": corr, "beta": beta}
        print(f"{name} {L:>2}    " + "   ".join(f"{a:.3f}  {b:.3f}  {c:.3f}" for (a, b, c) in row))
(HERE / "FOURTH_PLANE_RATIOS.json").write_text(json.dumps(out, indent=1))
