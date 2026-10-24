#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-LR4): E-62 tolerance screen summary vs the baseline arm. Per arm: per-net final-MSE ratio vs baseline, pooled
raw change (mean of per-net ratios - 1), and the native bill-cut estimate on S21 (census r100, 337.07 G):
  oldtier: rank-dependent old-tier products (reconstruct 11.19% + inner 7.03% + old pushes 4.67% + join tiles 1.84% + W5 shared_basis 0.82%
           + old-plane read 0.61% = 26.16%) scale with the stored rank: cut <= (1 - r/384) x 26.16%, BEFORE the truncation's own cost;
  star_s1: dense s1 (1.07 G per layer at 0.497 x 2n^3, 14 layers) -> range finder 8 n^2 k + thin 4 n^2 r + QR/SVD ~0.04 G (plain rate):
           r32 -2.06%, r64 -0.1%, r128 +1.9% (worse).
Bar (lead): raw <= +0.5% at bill <= -2%.  usage: lr_summary.py"""
import json, glob
from pathlib import Path
H = Path(__file__).resolve().parent
base = {r["rank"]: r for r in json.load(open(H / "lr_e62_S21_none_r0.json"))}
CUT = {("oldtier", 384): 0.0, ("oldtier", 128): -(1 - 128 / 384) * 26.16, ("oldtier", 64): -(1 - 64 / 384) * 26.16, ("oldtier", 32): -(1 - 32 / 384) * 26.16,
       ("star_s1", 32): -2.06, ("star_s1", 64): -0.1, ("star_s1", 128): +1.9}
print("baseline: " + ", ".join(f"r{k} {v['mse']:.4e}" for k, v in base.items()))
for f in sorted(glob.glob(str(H / "lr_e62_S21_*.json"))):
    if f.endswith("none_r0.json"): continue
    rows = json.load(open(f)); c, r = rows[0]["cls"], rows[0]["r"]
    ratios = [x["mse"] / base[x["rank"]]["mse"] for x in rows if x["rank"] in base]
    raw = 100 * (sum(ratios) / len(ratios) - 1); cut = CUT.get((c, r))
    ok = cut is not None and raw <= 0.5 and cut <= -2.0
    print(f"{c:8s} r{r:<4d} nets {len(ratios)}: raw pooled {raw:+.3f}% (per net {', '.join(f'{100*(q-1):+.2f}' for q in ratios)}); "
          f"native bill cut {'n/a' if cut is None else f'{cut:+.1f}%'}; replaced {sum(x['replaced'] for x in rows)}, missing P {sum(x['missing_P'] for x in rows)} "
          f"-> {'PASSES the bar' if ok else 'fails the bar'}")
