#!/usr/bin/env python3
"""TAG fable-engineer (E-50): user CPU and call counts by numpy op, per net, in process (OpenBLAS env, 1 thread). Every raw numpy
call that flopscope makes (_call_numpy) is timed with thread CPU time and filed under the numpy function's name; the flopscope op
count (len of the op log) is the request count of the client-server path apart from free trips. For pricing copy and binding
changes in wall terms. usage: op_cpu_profile.py <candidate.py> <ranks> <tag> -> fable_engineer_v1/op_cpu_<tag>.json"""
import sys, json, types, time, collections
from pathlib import Path
import numpy as np
import flopscope as flops, flopscope._budget as fb
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); ranks = [int(x) for x in sys.argv[2].split(",")]; tag = sys.argv[3]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("op_cpu_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
acc = collections.defaultdict(lambda: [0, 0.0])
orig = fb._call_numpy
def timed(fn, *args, **kwargs):
    t0 = time.thread_time()
    try:
        return orig(fn, *args, **kwargs)
    finally:
        v = acc[getattr(fn, "__name__", str(fn))]; v[0] += 1; v[1] += time.thread_time() - t0
for name in list(sys.modules):
    if name.startswith("flopscope") and getattr(sys.modules[name], "_call_numpy", None) is orig:
        sys.modules[name]._call_numpy = timed
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, w, nm): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, w, 0, nm
est = mod.Estimator(); out = []
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy"); acc.clear(); c0 = time.thread_time()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        est.predict(MLP([W[l] for l in range(16)], f"opc-{rank}"), 2**41)
    rec = dict(rank=rank, bill=int(ctx.summary_dict()["flops_used"]), ops=len(ctx._op_log), thread_cpu=round(time.thread_time() - c0, 2),
               by_fn={k: [v[0], round(v[1], 4)] for k, v in sorted(acc.items(), key=lambda kv: -kv[1][1])})
    out.append(rec)
    top = list(rec["by_fn"].items())[:10]
    print(json.dumps(dict(rank=rank, ops=rec["ops"], thread_cpu=rec["thread_cpu"], numpy_cpu=round(sum(v[1] for v in acc.values()), 2), top=top)), flush=True)
    json.dump(out, open(HERE / f"op_cpu_{tag}.json", "w"))
