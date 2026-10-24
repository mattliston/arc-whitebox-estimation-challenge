"""TAG opus_engineer a51 probe (2026-10-10; lead 17:xxZ go): which operands get their forward transforms (A side and B side of the tree
kernel) computed more than once within a layer (same array object, same (L, rows, cols)), i.e. what a broader transform cache would save.
In process, one network; prediction unchanged (the wrappers only record). Prints per (layer, side, key, object): times computed and the
transform's flops, then a summary of the repeats' flops.
usage: probe_trreuse.py <candidate.py> <rank>"""
from pathlib import Path
import collections
import json
import sys
import types
import numpy as np
import flopscope as flops

P2 = Path(__file__).resolve().parents[3]
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
candidate, rank = Path(sys.argv[1]).resolve(), int(sys.argv[2])
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_probe_tr'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
seen = collections.defaultdict(lambda: [0, 0, None])
CTX = [None]


def wrap(name, side):
    f = getattr(mod, name)

    def w(x, L, r, c, *rest, **kw):
        ctx = CTX[0]
        f0 = int(ctx.summary_dict()['flops_used']) if ctx is not None else 0
        out = f(x, L, r, c, *rest, **kw)
        f1 = int(ctx.summary_dict()['flops_used']) if ctx is not None else 0
        key = (-1 if mod._PHYS_LAYER[0] is None else int(mod._PHYS_LAYER[0]), side, (L, r, c), id(x))
        s = seen[key]
        s[0] += 1; s[1] += f1 - f0; s[2] = tuple(int(v) for v in x.shape)
        return out
    setattr(mod, name, w)


wrap('_TR_OLD_FWD_A', 'A')            # the A transforms the reuse wrapper did NOT skip
wrap('_ipt_forward_b_ab', 'B')
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
manifest = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
weights = np.load(STORE/Path(manifest[rank]['path']).parent/'weights.npy', allow_pickle=False)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[weights[l] for l in range(16)], seed=0, name=f'phase2-{rank}')
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    CTX[0] = ctx
    y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
    bill = int(ctx.summary_dict()['flops_used'])
rep = [(k, v) for k, v in seen.items() if v[0] > 1]
saved = sum(v[1] * (v[0] - 1) / v[0] for k, v in rep)
for k, v in sorted(rep, key=lambda kv: -kv[1][1])[:25]:
    print(json.dumps(dict(layer=k[0], side=k[1], key=k[2], shape=v[2], times=v[0], flops=v[1])))
print(json.dumps(dict(rank=rank, bill=bill, transforms=len(seen), repeated=len(rep), repeat_flops=saved, repeat_share=saved / bill)))
