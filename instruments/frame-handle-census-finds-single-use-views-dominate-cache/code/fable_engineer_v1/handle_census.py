#!/usr/bin/env python3
"""TAG fable-engineer (E-31): how many arrays (= server handles on the platform) the candidate's module-level caches keep alive after
each net, per cache (global name). In-process flopscope; counts distinct array objects reachable from every module global that is
a dict / list / tuple / set (recursively, through dicts, lists, tuples, sets and plain objects' __dict__), after each prediction.
usage: handle_census.py <candidate.py> <ranks e.g. 35,100,37,39> <tag>   -> fable_engineer_v1/handle_census_<tag>.json"""
import sys, json, types, collections
from pathlib import Path
import numpy as np
import flopscope as flops
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); ranks = [int(x) for x in sys.argv[2].split(",")]; tag = sys.argv[3]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("handle_census_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
def arrays_in(obj, seen_arr, seen_obj, depth=0):
    if depth > 12: return
    oid = id(obj)
    if isinstance(obj, np.ndarray):
        seen_arr.add(oid); return
    if oid in seen_obj: return
    if isinstance(obj, dict):
        seen_obj.add(oid)
        for v in obj.values(): arrays_in(v, seen_arr, seen_obj, depth + 1)
        for k in obj.keys():
            if isinstance(k, (tuple, np.ndarray)): arrays_in(k, seen_arr, seen_obj, depth + 1)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        seen_obj.add(oid)
        for v in obj: arrays_in(v, seen_arr, seen_obj, depth + 1)
    elif hasattr(obj, "__dict__") and not isinstance(obj, (types.ModuleType, type, types.FunctionType)):
        seen_obj.add(oid)
        for v in vars(obj).values(): arrays_in(v, seen_arr, seen_obj, depth + 1)
def census():
    per = {}; allarr = set(); seen_obj = set()
    for name, val in list(vars(mod).items()):
        if name.startswith("__") or isinstance(val, (types.ModuleType, type, types.FunctionType)): continue
        s = set(); arrays_in(val, s, set())
        if s: per[name] = len(s); allarr |= s
    return len(allarr), per
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
est = mod.Estimator(); out = []; prev = {}
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        pred = np.asarray(est.predict(MLP([W[l] for l in range(16)], f"hc-{rank}"), 2**41), dtype=np.float64)
    np.save(HERE / f"handle_census_{tag}_pred_r{rank}.npy", pred)
    total, per = census()
    delta = {k: v - prev.get(k, 0) for k, v in per.items() if v != prev.get(k, 0)}
    prev = per
    entries = {nm: len(getattr(mod, nm)) for nm in ("_A_FRAME_CACHE", "_B_FRAME_CACHE", "_O_FRAME_CACHE", "_PACKED_FRAME_CACHE", "_BH_O",
               "_BH_F", "_BH_X", "_READ5_FRAME_CACHE", "_READ5_CHILD_CACHE") if isinstance(getattr(mod, nm, None), dict)}
    rec = dict(rank=rank, total=total, util=round(ctx.summary_dict()["flops_used"] / 2**41, 5), per=per, delta=delta,
               keys=len(getattr(mod, "_EVL_LAST", {})), entries=entries)
    out.append(rec)
    print(json.dumps(dict(rank=rank, total=total, util=rec["util"], keys=rec["keys"], entries=entries, delta=dict(sorted(delta.items(), key=lambda kv: -kv[1])))), flush=True)
json.dump(out, open(HERE / f"handle_census_{tag}.json", "w"), indent=1)
