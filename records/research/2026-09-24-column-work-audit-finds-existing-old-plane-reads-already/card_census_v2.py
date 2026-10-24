#!/usr/bin/env python3
"""TAG fable-engineer (E-48 v2 of card_census.py: + the op name in every key, to split a column's price into leaf products,
transform adds, frame binding (reshape) and copies): native BILL and COLUMN census of one candidate (S11 = acf67b9d) per layer, the price base for the
lean-card candidates. One process, nets back to back (the first is cold, later ones warm). Every charged op is attributed to
(layer, function, source line): layer = the loop variable l of the innermost _predict_binary frame (None = outside the loop);
function = the innermost frame of the candidate file that is not a kernel helper (the analyst's bill_by_layer_v2 rule, with the
kernel list extended for the S5-S11 kernels: batched products, push5/push6 paths, used-slot frames, LRU caches); line = that
frame's current source line. Products of the four-site W-left port (_w5_port_product_body) are labelled port:<role> (role from
the live grant), and the layer-0 Gram (_resident_gram512) keeps its own name. Columns are recorded at every push
(_shared_old384_push: roster K, old frame columns before/after the join, active rows, births) and read (response rows, live rows,
young columns). No arithmetic changes: the wrappers only record shapes.
usage: card_census_v2.py <candidate.py> <ranks> <tag>   -> fable_engineer_v1/card_census_v2_<tag>.json (rows gain the op name before the flops)"""
import sys, json, types, collections
from pathlib import Path
import numpy as np
import flopscope as flops
from flopscope import _budget
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand = Path(sys.argv[1]).resolve(); ranks = [int(x) for x in sys.argv[2].split(",")]; tag = sys.argv[3]; name = str(cand)
KERNEL = {'_packed_product_cached', '_packed_product_3', '_packed_product_4', '_packed_product_5', '_packed_product_6', '_square_product',
          '_mm', '_shared_old384_mm', '_exact_view', '_a_frame', '_b_frame', '_o_frame', '_packed_frame', '_cp3_bind_arena_views', '<genexpr>',
          '_p1_d6_832_product', '_cp3_winograd_arena_depth4', '_cache_put', '_packed_read5', '_read5_child_frame',
          '_read5_leaf_packets', '_read5_packet_frames', '_read5_view', '_split_views_3', '_split_views_4', '_split_views_5', '_split_views_6',
          '<listcomp>', '<dictcomp>', '<lambda>', '_join_rows_mm', '_square_schedule', '_square_schedule_reference', '_rule_depth', 'frame',
          '_own_leaf', 'close', '_mm_batch', '_push5_product', '_push6_padded', '_dr_value', 'touch', 'get', '__call__', '__getitem__', '__init__'}
PREFIX = ('_bh_', '_read5', '_split_views', '_packed_product', '_cp3_', '_p1_d6_832', '_bb_', '_ul_', '_w5_port')
PORT = ('_w5_port_product_body', '_w5_port_product')
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("card_census_v2_mod"); mod.__file__ = name; sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), name, "exec"), mod.__dict__)
src = cand.read_text().splitlines()

bill = collections.Counter()              # (layer, label, line, op) -> flops
def charged(self, *args, **kwargs):
    op = args[0] if args else kwargs.get("op_name")
    before = self._flops_used
    r = _orig(self, *args, **kwargs)
    cost = self._flops_used - before
    if not cost:
        return r
    f = sys._getframe(1); fn = None; line = None; layer = None; port = False; gram = False
    while f is not None:
        if f.f_code.co_filename == name:
            co = f.f_code.co_name
            if co in PORT:
                port = True
            elif co == '_resident_gram512':
                gram = True
            elif fn is None and not (co in KERNEL or co.startswith(PREFIX)):
                fn, line = co, f.f_lineno
            if co == '_predict_binary':
                layer = f.f_locals.get('l')
                break
        f = f.f_back
    if port:
        scope = mod._W5_PORT_SCOPE; grant = scope.get('grant') if scope is not None else None
        fn = 'port:' + (grant['role'] if grant else '?')
    elif gram:
        fn = '_resident_gram512'
    bill[(layer, fn, line, op)] += cost
    return r
_orig = _budget.BudgetContext._charge_op
_budget.BudgetContext._charge_op = charged

pushes, reads = [], []
orig_push = mod._shared_old384_push
def push_wrap(layer, w32, p, previous, newborn, lam, state):
    old_before = int(state['columns'])
    out = orig_push(layer, w32, p, previous, newborn, lam, state)
    pushes.append(dict(layer=int(layer), K=int(lam.shape[0]), old_before=old_before, old=int(state['columns']), prev=int(previous.shape[0]),
                       rows=int(w32.shape[1]), births=(int(len(newborn[1])) if newborn is not None else 0),
                       inherited=(int(newborn[0]) if newborn is not None else int(p.shape[1])), out_cols=int(out.shape[1])))
    return out
mod._shared_old384_push = push_wrap
orig_read = mod._e2_shared_old384_read_young
def read_wrap(left, young_right, lam, live_indices, state):
    reads.append(dict(resp=int(left.shape[0]), live=int(live_indices.shape[0]), K=int(lam.shape[0]), old=int(state['columns']),
                      young=int(young_right.shape[1])))
    return orig_read(left, young_right, lam, live_indices, state)
mod._e2_shared_old384_read_young = read_wrap

man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
class MLP:
    def __init__(self, weights, nm): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, nm
est = mod.Estimator(); out = []
for rank in ranks:
    W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    bill.clear(); pushes.clear(); reads.clear()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        pred = est.predict(MLP([W[l] for l in range(16)], f"census-{rank}"), 2**41)
    total = int(ctx.summary_dict()["flops_used"])
    rows = [[k[0], k[1], k[2], (src[k[2] - 1].strip()[:90] if k[2] else ''), k[3], int(v)] for k, v in sorted(bill.items(), key=lambda kv: (str(kv[0][0]), -kv[1]))]
    rec = dict(rank=rank, cand=cand.name, total=total, util=total / 2**41, attributed=int(sum(bill.values())), rows=rows, pushes=list(pushes), reads=list(reads))
    out.append(rec)
    np.save(HERE / f"card_census_v2_{tag}_pred_r{rank}.npy", np.asarray(pred))
    print(json.dumps(dict(rank=rank, util=round(rec["util"], 6), attributed_minus_total=rec["attributed"] - total, rows=len(rows), pushes=len(pushes), reads=len(reads))), flush=True)
    json.dump(out, open(HERE / f"card_census_v2_{tag}.json", "w"))
print("DONE", tag, flush=True)
