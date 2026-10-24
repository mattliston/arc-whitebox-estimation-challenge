#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-ABS): IN-SITU shadow of the alternative-basis tree kernel (altbasis/ab_fragment.py) vs the native in-place tree
kernel on real single-product calls (_ipt_product) of a candidate (default S13), in process (OpenBLAS env, 1 thread, nets back to back, first
cold). Each call runs native and alt-basis on the same operands in random order; only the native output goes on. Per execution: bill, ops,
user/system CPU (getrusage), backend; max relative output difference. -> altbasis/ab_shadow_<tag>.json + per-depth summary.
usage: ab_shadow.py <cand.py> <ranks> <tag> [--seed=0]"""
import sys, json, types, time, random, collections, resource
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--")); pos = [a for a in sys.argv[1:] if not a.startswith("--")]
cand = Path(pos[0]).resolve(); ranks = [int(x) for x in pos[1].split(",")]; tag = pos[2]; rng = random.Random(int(opts.get("seed", "0")))
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("ab_shadow_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
exec(compile((HERE / "ab_fragment.py").read_text(), str(HERE / "ab_fragment.py"), "exec"), mod.__dict__)
native = mod._ipt_product; ctx_ref = [None]; log = []
def run(fn):
    c = ctx_ref[0]; f0, q0, t0 = c._flops_used, len(c._op_log), c.flopscope_backend_time_s; r0 = resource.getrusage(resource.RUSAGE_SELF)
    o = fn(); r1 = resource.getrusage(resource.RUSAGE_SELF)
    return o, dict(bill=int(c._flops_used - f0), ops=len(c._op_log) - q0, backend=round(c.flopscope_backend_time_s - t0, 5),
                   user=round(r1.ru_utime - r0.ru_utime, 5), sys=round(r1.ru_stime - r0.ru_stime, 5))
def wrap(a, b, out, L):
    if ctx_ref[0] is None:
        return native(a, b, out, L)
    m, k, n = int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1]); order = ["native", "ab"]; rng.shuffle(order)
    rec = dict(L=L, shape=[m, k, n], order=order); res = {}
    for w in order:
        if w == "native":
            res[w], rec[w] = run(lambda: native(a, b, out, L))
        else:
            o2 = fnp.empty((m, n), dtype=fnp.float32)
            res[w], rec[w] = run(lambda: mod._ipt_product_ab(a, b, o2, L))
    x = np.asarray(res["native"], dtype=np.float64); y = np.asarray(res["ab"], dtype=np.float64)
    rec["maxrel"] = float(np.abs(x - y).max() / max(np.abs(x).max(), 1e-30)); log.append(rec)
    return res["native"]
mod._ipt_product = wrap
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
est = mod.Estimator(); out = []
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy"); log.clear(); t0 = time.time()
    with flops.BudgetContext(flop_budget=2**44, quiet=True) as ctx:
        ctx_ref[0] = ctx; est.predict(MLP([W[l] for l in range(16)], f"ab-{rank}"), 2**41); ctx_ref[0] = None
    out.append(dict(rank=rank, calls=list(log), wall=round(time.time() - t0, 1)))
    byL = collections.defaultdict(list)
    for r in log: byL[r["L"]].append(r)
    print(f"net r{rank}: {len(log)} single tree products, wall {time.time()-t0:.0f} s", flush=True)
    for L, rs in sorted(byL.items()):
        s = lambda w, f: sum(r[w][f] for r in rs)
        print(f"   L{L} x{len(rs)}: bill {s('native','bill')/1e9:.3f} -> {s('ab','bill')/1e9:.3f} Gflop ({100*(s('ab','bill')/max(s('native','bill'),1)-1):+.2f}%), "
              f"user {s('native','user'):.2f} -> {s('ab','user'):.2f} s, ops {s('native','ops')} -> {s('ab','ops')}, maxrel {max(r['maxrel'] for r in rs):.2e}", flush=True)
json.dump(out, open(HERE / f"ab_shadow_{tag}.json", "w"))
print("DONE", tag)
