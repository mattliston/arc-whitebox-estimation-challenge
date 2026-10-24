#!/usr/bin/env python3
"""TAG fable-engineer (E-31): cross-net REUSE of the kernel's frame-cache entries. Every frame cache dict is replaced by a tracking
dict that records, per key, the nets in which it was created or hit; each key's cached arrays (= server handles) are counted.
Reports per cache and per arena layout: keys and arrays used in one net only vs in several nets.
usage: reuse_census.py <candidate.py> <ranks> <tag>   -> fable_engineer_v1/reuse_census_<tag>.json"""
import sys, json, types, collections
from pathlib import Path
import numpy as np
import flopscope as flops
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); ranks = [int(x) for x in sys.argv[2].split(",")]; tag = sys.argv[3]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("reuse_census_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
NET = [0]
class Track(dict):
    def __init__(self, *a):
        dict.__init__(self, *a); self.use = collections.defaultdict(set)
    def get(self, key, default=None):
        v = dict.get(self, key, default)
        if v is not default: self.use[key].add(NET[0])
        return v
    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value); self.use[key].add(NET[0])
NAMES = ("_A_FRAME_CACHE", "_B_FRAME_CACHE", "_O_FRAME_CACHE", "_PACKED_FRAME_CACHE", "_BH_O", "_BH_F", "_BH_X", "_READ5_FRAME_CACHE", "_READ5_CHILD_CACHE")
for nm in NAMES:
    if nm in vars(mod): setattr(mod, nm, Track(getattr(mod, nm)))
def arrays(obj, acc, seen):
    if isinstance(obj, np.ndarray): acc.add(id(obj)); return
    if id(obj) in seen: return
    if isinstance(obj, dict):
        seen.add(id(obj)); [arrays(v, acc, seen) for v in obj.values()]
    elif isinstance(obj, (list, tuple)):
        seen.add(id(obj)); [arrays(v, acc, seen) for v in obj]
def layout(views_id):
    S = mod._SCRATCH
    if S is not None:
        for i, nm in ((1, "PUSH"), (2, "READ"), (3, "SQ3"), (4, "SQ4"), (5, "SQ5"), (6, "SQ6")):
            if len(S) > i and S[i] is not None and id(S[i]) == views_id: return nm
    for nm, g in (("PUSH6", "_SCRATCH_PUSH6"), ("PUSH5", "_SCRATCH_PUSH5")):
        if id(getattr(mod, g, None)) == views_id: return nm
    return "other"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
est = mod.Estimator()
for i, rank in enumerate(ranks):
    NET[0] = i
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        est.predict(MLP([W[l] for l in range(16)], f"rc-{rank}"), 2**41)
    print(f"net {i} r{rank} util {ctx.summary_dict()['flops_used'] / 2**41:.5f}", flush=True)
report = {}
tot = collections.Counter()
for nm in NAMES:
    c = getattr(mod, nm, None)
    if not isinstance(c, Track): continue
    agg = collections.Counter()
    for key, nets in c.use.items():
        if key not in c: continue
        acc = set(); arrays(dict.get(c, key), acc, set())
        lay = layout(key[0]) if isinstance(key, tuple) and key and isinstance(key[0], int) else "other"
        cls = "single" if len(nets) == 1 else ("every" if len(nets) == len(ranks) else "some")
        agg[(lay, cls, "keys")] += 1; agg[(lay, cls, "arrays")] += len(acc)
        tot[(cls, "arrays")] += len(acc); tot[(lay, cls, "arrays")] += len(acc)
    report[nm] = {f"{a}/{b}/{c_}": v for (a, b, c_), v in sorted(agg.items())}
print(json.dumps({"per_cache": report}, indent=0)[:6000])
print("TOTAL arrays by reuse class (overlapping caches counted per cache):", {f"{k[0]}": v for k, v in tot.items() if len(k) == 2})
print("by layout:", {f"{k[0]}/{k[1]}": v for k, v in sorted(tot.items()) if len(k) == 3})
json.dump({"per_cache": report}, open(HERE / f"reuse_census_{tag}.json", "w"), indent=1)
