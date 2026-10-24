#!/usr/bin/env python3
"""TAG fable-analyst (AN-32): PER-LAYER bill census of one native prediction. Same hook as fable_porter_v1/bill_by_function.py (callers mode:
matrix-product kernel frames skipped, cost lands on the calling function of the candidate file), plus the layer: the value of the loop
variable l in the innermost _predict_binary frame on the stack (None = outside the loop). Prints the bill per layer and per (layer, function),
writes bill_by_layer_<cand>_r<rank>.json. usage: bill_by_layer.py <cand.py> <rank>"""
import sys, json, types, collections
from pathlib import Path
import numpy as np, flopscope as flops
from flopscope import _budget
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, rank = Path(sys.argv[1]).resolve(), int(sys.argv[2]); name = str(cand)
KERNEL = {'_packed_product_cached', '_packed_product_3', '_packed_product_4', '_packed_product_5', '_packed_product_6', '_square_product',
          '_mm', '_shared_old384_mm', '_exact_view', '_a_frame', '_b_frame', '_o_frame', '_packed_frame', '_cp3_bind_arena_views', '<genexpr>',
          '_p1_d6_832_product', '_resident_gram512', '_cp3_winograd_arena_depth4', '_cache_put', '_packed_read5', '_read5_child_frame',
          '_read5_leaf_packets', '_read5_packet_frames', '_read5_view', '_split_views_3', '_split_views_4', '_split_views_5', '_split_views_6',
          '_w5_port_call', '_w5_port_product', '_w5_port_product_body', '<listcomp>', '<dictcomp>', '<lambda>'}
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType("bill_layer_mod"); mod.__file__ = name; sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), name, "exec"), mod.__dict__)
fl = collections.Counter(); per_layer = collections.Counter()
_orig = _budget.BudgetContext._charge_op
def charged(self, *args, **kwargs):
    before = self._flops_used; r = _orig(self, *args, **kwargs); cost = self._flops_used - before
    f = sys._getframe(1); fn = None; layer = None
    while f is not None:
        if f.f_code.co_filename == name:
            if fn is None and f.f_code.co_name not in KERNEL: fn = f.f_code.co_name
            if f.f_code.co_name == '_predict_binary':
                layer = f.f_locals.get('l'); break
        f = f.f_back
    fl[(layer, fn)] += cost; per_layer[layer] += cost
    return r
_budget.BudgetContext._charge_op = charged
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
class MLP:
    def __init__(self, weights): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, f"billlayer-{rank}"
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    mod.Estimator().predict(MLP([W[l] for l in range(16)]), 2**41)
tot = ctx.summary_dict()["flops_used"]
print(json.dumps(dict(cand=cand.name, rank=rank, flops=tot, util=tot / 2**41)))
for l in sorted(per_layer, key=lambda x: (x is None, x if x is not None else 0)):
    print(f"layer {l}: {100 * per_layer[l] / tot:6.2f}%")
out = dict(total=tot, per_layer={str(k): v for k, v in per_layer.items()}, per_layer_fn={f'{k[0]}|{k[1]}': v for k, v in fl.items()})
json.dump(out, open(HERE / f'bill_by_layer_{cand.stem}_r{rank}.json', 'w'), indent=1)
