#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-DBC): bill of a candidate's tree-kernel products by entry and Winograd depth (in process, one or more nets).
Wraps _packed_product_cached (levels arg), _packed_read5 (5), _w5_port_product_body (levels), _cp_d6_shared (6), _bb_product (L); records
calls, bill and the m*k*n volume per (entry, depth).  usage: depth_bill_census.py <cand.py> <ranks>"""
import sys, json, types, collections
from pathlib import Path
import numpy as np, flopscope as flops, flopscope.numpy as fnp
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, ranks = Path(sys.argv[1]).resolve(), [int(x) for x in sys.argv[2].split(",")]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("dbc_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
ctx_ref = [None]; acc = collections.defaultdict(lambda: [0, 0, 0]); nest = [0]
def wrap(name, fn, lev):
    def w(*args):
        c = ctx_ref[0]
        if c is None or nest[0]: return fn(*args)
        f0 = c._flops_used; nest[0] += 1
        try: r = fn(*args)
        finally: nest[0] -= 1
        a, b = args[0], args[1]
        if name == "_bb_product": m, k, n, P = int(a[0].shape[0]), int(a[0].shape[1]), int(b[0].shape[1]), len(a)
        else: m, k, n, P = int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1]), 1
        x = acc[(name, lev(args))]; x[0] += 1; x[1] += int(c._flops_used - f0); x[2] += P * m * k * n
        return r
    return w
for name, lev in (("_packed_product_cached", lambda a: a[4]), ("_packed_read5", lambda a: 5), ("_w5_port_product_body", lambda a: a[4]),
                  ("_cp_d6_shared", lambda a: 6), ("_bb_product", lambda a: a[3])):
    setattr(mod, name, wrap(name, getattr(mod, name), lev))
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, name): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, name
est = mod.Estimator()
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy"); acc.clear()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        ctx_ref[0] = ctx; est.predict(MLP([W[l] for l in range(16)], f"dbc-{rank}"), 2**41); ctx_ref[0] = None
    T = int(ctx.summary_dict()["flops_used"]); byd = collections.defaultdict(lambda: [0, 0, 0])
    print(f"net r{rank}: total bill {T/1e9:.1f} G")
    for (name, L), (n, b, v) in sorted(acc.items(), key=lambda kv: (kv[0][1], -kv[1][1])):
        byd[L][0] += n; byd[L][1] += b; byd[L][2] += v
        print(f"   depth {L}  {name:24s} calls {n:4d}  bill {b/1e9:7.2f} G ({100*b/T:5.2f}%)  bill / 2mkn {b/(2*v):.3f}")
    for L, (n, b, v) in sorted(byd.items()):
        print(f"   DEPTH {L}: calls {n:4d}  bill {b/1e9:7.2f} G ({100*b/T:5.2f}%)  2mkn {2*v/1e9:7.1f} G  bill / 2mkn {b/(2*v):.3f}")
    print(f"   tree products total {sum(x[1] for x in byd.values())/1e9:.1f} G ({100*sum(x[1] for x in byd.values())/T:.1f}%)", flush=True)
