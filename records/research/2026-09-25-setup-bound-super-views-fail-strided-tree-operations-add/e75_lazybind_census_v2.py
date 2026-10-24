#!/usr/bin/env python3
"""TAG fable-engineer (E-75 v2 = v1 + a sha256 of each net's predictions, so two candidates' runs prove bit-identity; v1: E-75, podium push: price the lazy tree-key binds): in process (.venv), Estimator.setup first, then the given nets back
to back in ONE session (a platform worker's order). Every tree-view bind that happens INSIDE a predict (a cache miss of _ipt_side,
_ipt_side_p or the k-split's _ks_acc) is timed on the live budget counter (ctx.flops_used before/after), so its billed flops are exact.
A miss on a key already bound earlier in the session is an eviction rebind. Per net: binds, their bill (and % of the net's bill),
by depth and kind, rebinds, cache sizes. usage: .venv/bin/python e75_lazybind_census.py <cand.py> <ranks> <tag>"""
import sys, json, types, time, collections, hashlib
from pathlib import Path
import numpy as np, flopscope as flops
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
cand, ranks, tag = Path(sys.argv[1]).resolve(), [int(x) for x in sys.argv[2].split(',')], sys.argv[3]
out = HERE / f'e75v2_lazybind_{tag}.json'; assert not out.exists(), out
mod = types.ModuleType("lb_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
class MLP:
    def __init__(self, w, nm): self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, w, 0, nm
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
CTX = [None]; LOG = []
def wrap(name, cache_of, keyfn):
    orig = getattr(mod, name)
    def w(*a):
        cache, key, ek = cache_of(*a), keyfn(*a), (name,) + tuple(a)
        miss = CTX[0] is not None and cache is not None and dict.get(cache, key) is None
        f0 = CTX[0].flops_used if miss else 0
        r = orig(*a)
        if miss:
            LOG.append(dict(fn=name, key=list(a), bill=CTX[0].flops_used - f0, rebind=ek in SEEN)); SEEN.add(ek)
        elif cache is not None and dict.get(cache, key) is not None:
            SEEN.add(ek)
        return r
    setattr(mod, name, w)
SEEN = set()
for kind, c in enumerate((mod._IPT_A, mod._IPT_B, mod._IPT_C)):
    for kk in dict.keys(c):
        SEEN.add(('_ipt_side', kind) + tuple(kk))
wrap('_ipt_side', lambda kind, L, r, c: (mod._IPT_A, mod._IPT_B, mod._IPT_C)[kind], lambda kind, L, r, c: (L, r, c))
if hasattr(mod, '_ipt_side_p'):
    wrap('_ipt_side_p', lambda kind, L, P, r, c: (mod._IPT_BA, mod._IPT_BB, mod._IPT_BC)[kind], lambda kind, L, P, r, c: (L, P, r, c))
if hasattr(mod, '_ks_acc'):
    wrap('_ks_acc', lambda M, N: mod._KS_ACC, lambda M, N: (M, N))
nets = []
for r in ranks:
    LOG.clear(); W = np.load(STORE / Path(man[r]["path"]).parent / "weights.npy")
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        CTX[0] = ctx
        y = est.predict(MLP([W[l] for l in range(16)], f"lb-{r}"), 2**41)
        CTX[0] = None
    tot = ctx.summary_dict()["flops_used"]; bill = sum(x['bill'] for x in LOG)
    byd = collections.Counter(); 
    for x in LOG: byd[f"{x['fn']}|L{x['key'][1] if x['fn'] != '_ks_acc' else 6}"] += x['bill']
    yh = hashlib.sha256(np.ascontiguousarray(np.asarray(y, dtype=np.float64)).tobytes()).hexdigest()[:16]
    rec = dict(rank=r, flops=tot, y_sha=yh, binds=len(LOG), bind_bill=bill, bind_pct=round(100 * bill / tot, 3), rebinds=sum(1 for x in LOG if x['rebind']),
               rebind_bill=sum(x['bill'] for x in LOG if x['rebind']), by=dict(byd), caches=[len(mod._IPT_A), len(mod._IPT_B), len(mod._IPT_C)])
    nets.append(dict(rec, log=list(LOG)))
    print(json.dumps(rec), flush=True)
json.dump(dict(tag=tag, cand=str(cand), ranks=ranks, nets=nets), open(out, 'w'))
print('E75_LAZYBIND_DONE', flush=True)
