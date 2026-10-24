#!/usr/bin/env python3
"""TAG fable-engineer (E-62c): torch half of the function-level certificate of lrstar_torch_hook.py against the native S18 fragment.
Loads lrstar_native_outputs.py's file (star planes, the consuming w, the native _flms_star terms and _flms_star_diagonal values) and
evaluates the torch hook's functions on the same inputs. Printed per plane: relative Frobenius differences and the singular-value gap
at the rank cutoff (s_32 / s_33 - 1), since a small gap makes the kept subspace sensitive to rounding.
usage (python with torch): lrstar_factor_cert.py <native_outputs.npz>"""
import sys, json
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lrstar_torch_hook as lh
d = np.load(sys.argv[1])
class Stub:
    def __init__(self): self.env = {}
card = lh.install(Stub(), torch)
worst = [0.0, 0.0]
for i in range(len(d["planes"])):
    A, w, sn, dn = d["planes"][i], d["w"][i], d["star"][i], d["diag"][i]
    st = card.env["_flms_star"](torch.from_numpy(w), torch.from_numpy(A)).numpy()
    dt = card.env["_flms_star_diagonal"](torch.from_numpy(w), torch.from_numpy(A)).numpy()
    rs = float(np.linalg.norm(st - sn) / np.linalg.norm(sn)); rd = float(np.linalg.norm(dt - dn) / np.linalg.norm(dn))
    worst = [max(worst[0], rs), max(worst[1], rd)]
    s = np.linalg.svd(A.astype(np.float64), compute_uv=False)
    print(json.dumps(dict(plane=i, star_rel_diff=float(f"{rs:.3e}"), diag_rel_diff=float(f"{rd:.3e}"), gap_at_32=round(float(s[31] / s[32] - 1), 4))), flush=True)
print("MAX rel diff: star %.3e, diagonal %.3e" % tuple(worst))
