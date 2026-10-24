#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-AVI): independent check of the engineer's --ipab against engineer2's alternative-basis variant on the same
single-product harness as ab_shadow.py. S13 in process (OpenBLAS env, 1 thread); engineer2's ab_fragment.py and the engineer's
ipab_fragment.py are both loaded, then S13's native dispatchers are restored so the network runs exactly S13. Every single tree product
(_ipt_product) runs native, engineer2-ab (_ipt_product_ab) and engineer-ipab (_ipab_product) on the same operands in random order; the
native output goes on. Per execution: bill, ops (requests), user/system CPU, backend; max relative difference vs native.
usage: ab_vs_ipab.py <S13.py> <ranks> <tag>"""
import sys, json, types, time, random, collections, resource
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, ranks, tag = Path(sys.argv[1]).resolve(), [int(x) for x in sys.argv[2].split(",")], sys.argv[3]; rng = random.Random(0)
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("avi_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
keep = {n: getattr(mod, n) for n in ("_ipt_product", "_cp_d6_shared", "_bb_product", "_w5_port_product_body", "_packed_product_cached", "_packed_read5")}
exec(compile((HERE / "ab_fragment.py").read_text(), "ab_fragment", "exec"), mod.__dict__)
ipab_path = P2 / "fable_engineer_v1" / "ipab_fragment.py"
exec(compile(ipab_path.read_text(), str(ipab_path), "exec"), mod.__dict__)
for n, f in keep.items(): setattr(mod, n, f)            # network = exactly S13
native, mine, ipab = keep["_ipt_product"], mod._ipt_product_ab, mod._ipab_product
ctx_ref = [None]; log = []
def run(fn):
    c = ctx_ref[0]; f0, q0, t0 = c._flops_used, len(c._op_log), c.flopscope_backend_time_s; r0 = resource.getrusage(resource.RUSAGE_SELF)
    o = fn(); r1 = resource.getrusage(resource.RUSAGE_SELF)
    return o, dict(bill=int(c._flops_used - f0), ops=len(c._op_log) - q0, backend=round(c.flopscope_backend_time_s - t0, 5),
                   user=round(r1.ru_utime - r0.ru_utime, 5), sys=round(r1.ru_stime - r0.ru_stime, 5))
def wrap(a, b, out, L):
    if ctx_ref[0] is None:
        return native(a, b, out, L)
    m, k, n = int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1]); order = ["native", "mine", "ipab"]; rng.shuffle(order)
    rec = dict(L=L, shape=[m, k, n]); res = {}
    for w in order:
        if w == "native":
            res[w], rec[w] = run(lambda: native(a, b, out, L))
        else:
            o2 = fnp.empty((m, n), dtype=fnp.float32); fn = mine if w == "mine" else ipab
            res[w], rec[w] = run(lambda: fn(a, b, o2, L))
    x = np.asarray(res["native"], dtype=np.float64)
    for w in ("mine", "ipab"):
        rec[w]["maxrel"] = float(np.abs(np.asarray(res[w], dtype=np.float64) - x).max() / max(np.abs(x).max(), 1e-30))
    log.append(rec)
    return res["native"]
mod._ipt_product = wrap
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
est = mod.Estimator(); out = []
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy"); log.clear(); t0 = time.time()
    with flops.BudgetContext(flop_budget=2**44, quiet=True) as ctx:
        ctx_ref[0] = ctx; est.predict(MLP([W[l] for l in range(16)], f"avi-{rank}"), 2**41); ctx_ref[0] = None
    out.append(dict(rank=rank, calls=list(log)))
    tot = lambda w, f: sum(r[w][f] for r in log)
    print(f"net r{rank}: {len(log)} single tree products, wall {time.time()-t0:.0f} s", flush=True)
    for w in ("native", "mine", "ipab"):
        extra = "" if w == "native" else f", maxrel {max(r[w]['maxrel'] for r in log):.2e}"
        print(f"   {w:6s}: bill {tot(w,'bill')/1e9:8.3f} Gflop, user {tot(w,'user'):6.2f} s, backend {tot(w,'backend'):6.2f} s, ops {tot(w,'ops'):6d}{extra}", flush=True)
json.dump(out, open(HERE / f"ab_vs_ipab_{tag}.json", "w"))
print("DONE", tag)
