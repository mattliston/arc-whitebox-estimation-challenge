#!/usr/bin/env python3
"""TAG g29-engine-feed-ablation-table: per-layer k3 relative error vs the 32M truth
for the card (host_k3), the engine reference (tensor_planes_rank{r}_dep_g29ref.npz)
and the engine without the plane-to-distinct feed (_dep_nofeed).  Engine k3u_l is
aligned to mc_pre_k3[l+1] (checked: the alignment giving the smaller error is
printed for both candidates at one layer).  Also prints the energy of the
card->engine gap that the feed accounts for:  1 - |engine_ref - engine_nofeed|^2 ...
Usage: engine_feed_ablation_table.py <rank>"""
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
G19 = HERE.parent / "public_g19_attribution_v1"
rank = int(sys.argv[1]) if len(sys.argv) > 1 else 35
T = np.load(G19 / f"mps_truth_rank{rank:04d}_P32M.npz")
ST = np.load(G19 / f"states_rank{rank:04d}.npz")
ref = np.load(G19 / f"tensor_planes_rank{rank:04d}_dep_g29ref.npz")
nof = np.load(G19 / f"tensor_planes_rank{rank:04d}_dep_nofeed.npz")
tk3 = T["mc_pre_k3"].astype(np.float64)
rel = lambda x, t: float(np.linalg.norm(x - t) / np.linalg.norm(t))
# alignment check at layer 8
for off in (0, 1):
    print(f"alignment k3u_l <-> mc_pre_k3[l+{off}] at l=8: rel {rel(ref['k3u_8'].astype(np.float64), tk3[8 + off]):.4f}")
rows = []
print("L   card    eng_ref  eng_nofeed  |ref-nofeed|/|truth|  corr(ref-nofeed, truth-nofeed)  gap_energy_explained")
for L in range(2, 16):
    kl = f"k3u_{L}"                      # alignment check above: k3u_l <-> mc_pre_k3[l]
    if kl not in ref.files or kl not in nof.files:
        continue
    t = tk3[L]
    card = ST["host_k3"][L - 1].astype(np.float64)
    e_ref = ref[kl].astype(np.float64)
    e_nof = nof[kl].astype(np.float64)
    feed_eff = e_ref - e_nof
    resid_nof = t - e_nof
    c = float(feed_eff @ resid_nof) / np.sqrt(float(feed_eff @ feed_eff) * float(resid_nof @ resid_nof))
    expl = 1.0 - float((resid_nof - feed_eff) @ (resid_nof - feed_eff)) / float(resid_nof @ resid_nof)
    rows.append({"L": L, "card": rel(card, t), "eng_ref": rel(e_ref, t), "eng_nofeed": rel(e_nof, t),
                 "feed_effect_rel": rel(e_ref, e_nof) * np.linalg.norm(e_nof) / np.linalg.norm(t),
                 "corr_feed_vs_nofeed_resid": c, "gap_energy_explained": expl})
    r = rows[-1]
    print(f"{L:>2}  {r['card']:.4f}  {r['eng_ref']:.4f}  {r['eng_nofeed']:.4f}   {r['feed_effect_rel']:.4f}"
          f"              {c:+.3f}                       {expl:+.3f}")
(HERE / f"ENGINE_FEED_ABLATION_rank{rank:04d}.json").write_text(json.dumps(rows, indent=1))
