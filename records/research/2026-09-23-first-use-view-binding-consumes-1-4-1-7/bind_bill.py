#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-BB): FIRST-TIME BINDING BILL per network in one warm process (in-process flopscope with the server's
numpy; run with the openblas env python). Nets run back to back like one platform worker; for each net it splits the bill into
(a) reshape charges (view binding), (b) every charge made inside a view/frame/cache-binding function of the candidate, and
(c) the rest, so the per-MLP bill paid for shapes a worker has not seen yet is visible. Measurement only.
usage: bind_bill.py <cand.py> <ranks e.g. 35,100,37> <tag>  -> fable_engineer2_v1/bind_bill_<tag>.json"""
import sys, json, types, collections
from pathlib import Path
import numpy as np, flopscope as flops
from flopscope import _budget
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, ranks, tag = Path(sys.argv[1]).resolve(), [int(x) for x in sys.argv[2].split(',')], sys.argv[3]; name = str(cand)
BIND = {'_cp3_bind_arena_views', '_packed_frame', '_exact_view', '_a_frame', '_b_frame', '_o_frame', '_bh_o_frame', '_bh_fwd',
        '_read5_view', '_read5_child_frame', '_read5_packet_frames', '_bind_push6', '_split_views_3', '_split_views_4',
        '_split_views_5', '_split_views_6', '_e2_pair_state', '_e2_gnn_consts', '_e2_gnn_layer', '_e2_edge_consts', '_e2_bd_state',
        '_unpack_static_fp32', '_bh_leafx', '_e2_feature_columns', '_message_edge_parameters', '_bdnh_parameters'}
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("bind_bill_mod"); mod.__file__ = name; sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), name, "exec"), mod.__dict__)
acc = collections.Counter(); fn_acc = collections.Counter()
_orig = _budget.BudgetContext._charge_op
def charged(self, op_name, flop_cost, subscripts, shapes, **kw):
    before = self._flops_used; r = _orig(self, op_name, flop_cost, subscripts, shapes, **kw); cost = self._flops_used - before
    f = sys._getframe(1); binder = None
    while f is not None:
        if f.f_code.co_filename == name and f.f_code.co_name in BIND:
            binder = f.f_code.co_name; break
        f = f.f_back
    if op_name == 'reshape': acc['reshape'] += cost
    if binder: acc['binding_fn'] += cost; fn_acc[binder] += cost
    acc['total'] += cost
    return r
_budget.BudgetContext._charge_op = charged
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, nm): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, nm
est = mod.Estimator(); out = []
for i, rank in enumerate(ranks):
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy"); acc.clear(); fn_acc.clear()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        est.predict(MLP([W[l] for l in range(16)], f"bindbill-{rank}"), 2**41)
    tot = ctx.summary_dict()["flops_used"]
    rec = dict(i=i, rank=rank, util=round(tot / 2**41, 6), reshape_pct=round(100 * acc['reshape'] / tot, 3),
               binding_pct=round(100 * acc['binding_fn'] / tot, 3), top=[(k, round(100 * v / tot, 3)) for k, v in fn_acc.most_common(6)])
    out.append(rec); print(json.dumps(rec), flush=True)
json.dump(out, open(HERE / f"bind_bill_{tag}.json", "w"), indent=1)
