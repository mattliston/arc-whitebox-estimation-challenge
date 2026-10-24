#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-BL v2: tree = rank >= 4 AND last dim < 512 (pair-map arena arrays are body); call sites for getitem/tanh/qr/sum/body multiply): where the bill is LEFT (lead: after IPT + alt basis). In-process census of a candidate (default the engineer's
candS14ab_mem.py = S14 + ipab): every charged op with its op name, the rank of its first operand and its bill (BudgetContext._charge_op hook).
Classes: tree leaf matmul (matmul with operand rank >= 4: batched over the 7^L chunk axes), other matmul/einsum (body products), tree
elementwise (add/subtract/copyto/multiply with operand rank >= 4: Winograd transforms, copies), body elementwise (rank <= 3), reshape
(frame/view binding, billed at numel), everything else by name. usage: bill_left_census.py <cand.py> <ranks> <tag>"""
import sys, json, types, time, collections
from pathlib import Path
import numpy as np, flopscope as flops
from flopscope import _budget
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, ranks, tag = Path(sys.argv[1]).resolve(), [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("bl_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
acc = collections.Counter(); cnt = collections.Counter(); site = collections.Counter(); on = [False]
_orig = _budget.BudgetContext._charge_op
def _charge(self, op_name, flop_cost, subscripts, shapes, **kw):
    b0 = self._flops_used; r = _orig(self, op_name, flop_cost, subscripts, shapes, **kw)
    if on[0]:
        d = self._flops_used - b0; sh = tuple(shapes[0]) if shapes and hasattr(shapes[0], "__len__") else (); rk = len(sh)
        tree = rk >= 4 and sh and int(sh[-1]) < 512          # tree arrays end in small leaf blocks; pair-map arena arrays end in n = 1024
        if op_name in ("matmul", "dot", "tensordot"): cls = "tree leaf matmul" if tree else "body matmul"
        elif op_name == "einsum": cls = "body einsum"
        elif op_name in ("add", "subtract", "copyto", "multiply", "negative"): cls = ("tree " if tree else "body ") + op_name
        elif op_name == "reshape": cls = "reshape (binding) " + ("tree" if (rk >= 4 and int(sh[-1]) < 512) or rk <= 1 else "other")
        else: cls = "other: " + op_name
        acc[cls] += d; cnt[cls] += 1
        if op_name in ("getitem", "tanh", "linalg.qr", "sum") or cls == "body multiply":
            f = sys._getframe(1)
            while f is not None and f.f_code.co_filename != str(cand): f = f.f_back
            if f is not None: site[(cls, f.f_code.co_name, f.f_lineno)] += d
    return r
_budget.BudgetContext._charge_op = _charge
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, nm): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, nm
est = mod.Estimator(); out = []
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy"); acc.clear(); cnt.clear(); t0 = time.time()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        on[0] = True; est.predict(MLP([W[l] for l in range(16)], f"bl-{rank}"), 2**41); on[0] = False
    tot = int(ctx._flops_used)
    out.append(dict(rank=rank, total=tot, classes={k: [int(v), int(cnt[k])] for k, v in acc.items()}))
    print(f"net r{rank}: bill {tot/1e9:.1f} Gflop (util {tot/2**41:.5f}), wall {time.time()-t0:.0f} s", flush=True)
    for k, v in sorted(acc.items(), key=lambda x: -x[1])[:18]:
        print(f"   {k:28s} {v/1e9:8.2f} Gflop {100*v/tot:6.2f}%  ({cnt[k]} ops)", flush=True)
    print("   top call sites (getitem / tanh / qr / sum / body multiply):", flush=True)
    for (c, fn, ln), v in sorted(site.items(), key=lambda x: -x[1])[:10]:
        print(f"      {c:18s} {fn}:{ln} {v/1e9:.2f} Gflop ({100*v/tot:.2f}%)", flush=True)
    site.clear()
json.dump(out, open(HERE / f"bill_left_{tag}.json", "w"))
