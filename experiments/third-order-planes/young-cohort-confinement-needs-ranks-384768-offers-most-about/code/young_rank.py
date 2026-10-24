#!/usr/bin/env python3
"""TAG fable-oldtier-economics (OT-8): rank profile of the YOUNG cohorts' columns by age (from the full bank dump): energy captured at
ranks 256/384/512/640/768/896 for cohorts of age 1..4 at layers 8..14, i.e. how early a cohort could be confined to a shared frame and
at what rank (the young push + young reads are ~29% of the bill).  usage: young_rank.py <bank_dir>"""
import sys, json
from pathlib import Path
import numpy as np
D = Path(sys.argv[1]); out = {}
for l in (8, 11, 14):
    z = np.load(D / f"layer_{l:02d}.npz"); P = z["pp"].astype(np.float64); oc = int(z["old_columns"]); coh = z["cohorts"]
    for b, (st, wd) in enumerate(coh):
        if st < oc or wd == 0: continue
        age = l - b; Pb = P[:, st:st + wd]
        s = np.linalg.svd(Pb, compute_uv=False); e = np.cumsum(s ** 2) / (s ** 2).sum()
        rec = {r: round(float(e[r - 1]), 5) for r in (128, 256, 384, 512, 640, 768, 896) if r <= len(e)}
        rec99 = int(np.searchsorted(e, 0.99) + 1); rec999 = int(np.searchsorted(e, 0.999) + 1)
        print(json.dumps(dict(layer=l, cohort=b, age=age, cols=int(wd), rank99=rec99, rank999=rec999, energy=rec)), flush=True)
        out[f"{l}:{b}"] = dict(age=age, cols=int(wd), rank99=rec99, rank999=rec999, energy=rec)
json.dump(out, open(D / "young_rank.json", "w"), indent=1)
