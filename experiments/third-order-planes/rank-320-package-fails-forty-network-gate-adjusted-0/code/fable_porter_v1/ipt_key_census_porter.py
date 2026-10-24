#!/usr/bin/env python3
"""TAG fable-porter (PT-53: versioned copy of fable_engineer_v1/ipt_key_census.py, unchanged except this line; its output lands in
fable_porter_v1/ because HERE is this file's directory). Original: TAG fable-engineer (E-61): which tree-kernel cache entries does each net bind? The alt-basis tree kernel keys its bound views by
shape: (kind, L, rows, cols) for single products and (kind, L, P, rows, cols) for batched ones. A cold net binds ~219 entries (~22
requests each, ~19 ms of cold residual locally); caches warm over ~3 nets. This census runs nets in ONE in-process session and logs,
per net, every key used and whether it missed (membership test only: no LRU reordering), to price prebinding the recurring shapes in
setup(). usage: ipt_key_census.py <candidate.py> <ranks> <tag> -> fable_engineer_v1/ipt_key_census_<tag>.json"""
import sys, json, types, time, collections
from pathlib import Path
import numpy as np
import flopscope as flops
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); ranks = [int(x) for x in sys.argv[2].split(",")]; tag = sys.argv[3]
out = HERE / f"ipt_key_census_{tag}.json"
if out.exists():
    raise SystemExit(f"refusing to overwrite {out}")
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("ikc_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, w, nm): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, w, 0, nm
log = {"used": collections.Counter(), "miss": collections.Counter()}
_side, _side_p = mod._ipt_side, mod._ipt_side_p
def side(kind, L, rows, cols):
    key = (kind, L, rows, cols)
    k = f"s{kind}|{L}|{rows}|{cols}"
    log["used"][k] += 1
    if key not in (mod._IPT_A, mod._IPT_B, mod._IPT_C)[kind]:
        log["miss"][k] += 1
    return _side(kind, L, rows, cols)
def side_p(kind, L, P, rows, cols):
    key = (L, P, rows, cols)
    k = f"p{kind}|{L}|{P}|{rows}|{cols}"
    log["used"][k] += 1
    if key not in (mod._IPT_BA, mod._IPT_BB, mod._IPT_BC)[kind]:
        log["miss"][k] += 1
    return _side_p(kind, L, P, rows, cols)
mod._ipt_side = side; mod._ipt_side_p = side_p
est = mod.Estimator(); res = []
for rank in ranks:
    log["used"].clear(); log["miss"].clear()
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    t0 = time.time()
    with flops.BudgetContext(flop_budget=2**41, quiet=True, namespace=f"ikc-{rank}"):
        est.predict(MLP([W[l] for l in range(16)], f"ikc-{rank}"), 2**41)
    rec = dict(rank=rank, wall=round(time.time() - t0, 1), used=dict(log["used"]), miss=dict(log["miss"]))
    res.append(rec)
    print(json.dumps(dict(rank=rank, wall=rec["wall"], keys=len(rec["used"]), calls=sum(rec["used"].values()), misses=len(rec["miss"]),
                          sizes=[len(mod._IPT_A), len(mod._IPT_B), len(mod._IPT_C)])), flush=True)
json.dump(dict(tag=tag, cand=str(cand), ranks=ranks, nets=res), open(out, "w"), indent=1)
print("WROTE", out)
