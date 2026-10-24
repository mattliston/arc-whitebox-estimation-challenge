#!/usr/bin/env python3
"""TAG fable-porter (PT-57b: versioned copy of ks_shape_check.py for k-split v4, which has no _KS_SHAPES: its chunk keys come from the
re-derived _PB_SINGLE, so the check reports, per net, how many of the split reads' chunk keys (0,6,M,q), (1,6,q,N), (2,6,M,N) are in _PB_SINGLE
(bound and fused in setup) vs bound lazily in predict; with _KS_SHAPES present (v2/v3) it also reports shapes outside it). Original: TAG fable-porter (PT-57, engineer 03:3x "after composing S26 run a product census on one net: every split shape must be in _KS_SHAPES"):
in process (.venv), Estimator.setup(context) first (as the platform), then one predict; every read the k-split actually splits (a call of
_ks_product, wrapped by global name) is recorded as (m, k, n). PASS iff every split shape is in the candidate's _KS_SHAPES (a split shape
outside it would bind its chunk keys lazily in predict: billed and unfused). usage: ks_shape_check.py <candidate.py> <rank>"""
import sys, json, types, collections
from pathlib import Path
import numpy as np, flopscope as flops
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
cand, rank = Path(sys.argv[1]).resolve(), int(sys.argv[2])
mod = types.ModuleType("ksc_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
class MLP:
    def __init__(self, w, nm): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, w, 0, nm
est = mod.Estimator()
sctx = types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0)
with flops.BudgetContext(flop_budget=2**41, quiet=True):
    est.setup(sctx)
split = collections.Counter(); plans = {}; _orig = mod._ks_product
def _rec(a, b, out, m, k, n, plan):
    split[(m, k, n)] += 1; plans[(m, k, n)] = plan
    return _orig(a, b, out, m, k, n, plan)
mod._ks_product = _rec
W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    est.predict(MLP([W[l] for l in range(16)], f"ksc-{rank}"), 2**41)
shapes = set(tuple(s) for s in getattr(mod, '_KS_SHAPES', ())); outside = sorted(s for s in split if shapes and s not in shapes)
pbs = set(tuple(k) for k in mod._PB_SINGLE); ck = set()
for (m, k, n), (M, N, sizes) in plans.items():
    for q in set(sizes): ck |= {(0, 6, M, q), (1, 6, q, N)}
    ck.add((2, 6, M, N))
lazy = sorted(ck - pbs)
print(json.dumps(dict(rank=rank, setup_log=[list(x) for x in getattr(mod, '_PB_LOG', [])], split_shapes={f'{m}x{k}x{n}': c for (m, k, n), c in sorted(split.items())},
                      n_split_calls=sum(split.values()), ks_stats=dict(mod._KS_STATS), outside_KS_SHAPES=[list(s) for s in outside],
                      unused_KS_SHAPES=[list(s) for s in sorted(shapes - set(split))], chunk_keys=len(ck), chunk_keys_in_PB_SINGLE=len(ck & pbs), chunk_keys_lazy=[list(x) for x in lazy], flops=ctx.summary_dict()["flops_used"])))
print(f'KS CHUNK-KEY COVERAGE: {len(ck & pbs)}/{len(ck)} chunk keys of the {sum(split.values())} split reads are in _PB_SINGLE ({len(lazy)} bind lazily)' + (f'; {len(outside)} split shape(s) outside _KS_SHAPES' if shapes else ''))
