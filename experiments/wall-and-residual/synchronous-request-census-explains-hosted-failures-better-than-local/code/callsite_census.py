#!/usr/bin/env python3
"""TAG fable-wall-repair (W-F): census of every REMOTE REQUEST the numpy-free client sends during one prediction, attributed to the
candidate's call site (the innermost frame inside the candidate file) and to the request op. Run in the numpy-free client python with
FLOPSCOPE_SERVER_URL set (server: OpenBLAS flopscope_server).  usage: callsite_census.py <candidate.py abs> <rank> <tag>
-> <tag>_callsites.json {total, by_site{line func -> count}, by_func, by_op}; prints the totals, the top 40 sites and per-function sums."""
import collections, importlib.util, json, sys, types
from pathlib import Path
import flopscope as flops, flopscope.numpy as fnp
from flopscope import _connection
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
assert importlib.util.find_spec("numpy") is None, "must run numpy-free"
cand, rank, tag = Path(sys.argv[1]).resolve(), int(sys.argv[2]), sys.argv[3]
stub = types.ModuleType("whestbench")
class BaseEstimator: pass
stub.BaseEstimator = BaseEstimator; sys.modules["whestbench"] = stub
spec = importlib.util.spec_from_file_location("census_candidate", cand); mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
by_site = collections.Counter(); by_func = collections.Counter(); by_op = collections.Counter(); total = [0]; active = [False]
cand_name = str(cand)
_orig = _connection.Connection.send_recv
def counted(self, raw_request):
    if active[0]:
        total[0] += 1
        f = sys._getframe(1); site = None
        while f is not None:
            if f.f_code.co_filename == cand_name:
                site = (f.f_lineno, f.f_code.co_name); break
            f = f.f_back
        by_site[site] += 1; by_func[site[1] if site else None] += 1
        try:
            by_op[raw_request[:24].split(b"\x00")[0][:24].decode("latin1")] += 1
        except Exception:
            pass
    return _orig(self, raw_request)
_connection.Connection.send_recv = counted
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
fixture = STORE / Path(man[rank]["path"]).parent / "weights.npy"
est = mod.Estimator()
with flops.BudgetContext(flop_budget=2**41, quiet=True, namespace=f"{tag}-census") as ctx:
    blob = fnp.load(str(fixture)); weights = [blob[l] for l in range(16)]
    active[0] = True
    out = est.predict(MLP(weights, f"census-{rank}"), 2**41)
    active[0] = False
s = ctx.summary_dict()
print(f"{tag}: rank {rank} remote requests during predict {total[0]} | flops {int(s['flops_used']):.4e} util {int(s['flops_used'])/2**41:.4f}")
print("per function:", ", ".join(f"{k}={v}" for k, v in by_func.most_common(24)))
print("top sites:")
for (site, c) in by_site.most_common(40):
    print(f"  {c:6d}  line {site[0] if site else '-'}  {site[1] if site else '-'}")
json.dump(dict(total=total[0], by_site={f"{k[0]} {k[1]}" if k else "-": v for k, v in by_site.items()}, by_func={str(k): v for k, v in by_func.items()},
               by_op=dict(by_op), flops=int(s["flops_used"])), open(HERE / f"{tag}_callsites.json", "w"), indent=1)
