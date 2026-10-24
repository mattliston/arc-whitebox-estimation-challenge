#!/usr/bin/env python3
"""TAG fable-porter (PT-55a v2: v1 computed plans at import, before the arena exists, so _IPT_REGION[6] was unset and every plan was None;
v2 runs Estimator.setup(context) first, in a BudgetContext, as the platform and parity_layers_setup.py do, then reads the plans). Original: TAG fable-porter (PT-55a): the tree keys the engineer's k-split v2 phase 3 binds FUSED in setup, computed from the candidate's own _KS_SHAPES,
_ks_plan, _IPT_REGION and _ipt_fits (loaded as the census and parity tools load it; nothing runs): for each planned shape, (0, 6, M, q) and
(1, 6, q, N) per chunk size q, and (2, 6, M, N). usage (.venv): ks_phase3_keys.py <candidate.py> <out.json>"""
import sys, json, types
from pathlib import Path
import flopscope as flops
stub = types.ModuleType("whestbench"); stub.BaseEstimator = type("BaseEstimator", (), {}); sys.modules["whestbench"] = stub
flops.configure(symmetry_warnings=False)
cand, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
assert not out.exists(), f'refusing to overwrite {out}'
mod = types.ModuleType("ksp_mod"); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), "exec"), mod.__dict__)
import types as _ty
sctx = _ty.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0)
with flops.BudgetContext(flop_budget=2**41, quiet=True):
    mod.Estimator().setup(sctx)
print('setup log:', [list(x) for x in getattr(mod, '_PB_LOG', [])])
keys, plans = [], {}
for m, k, n in mod._KS_SHAPES:
    plan = mod._ks_plan(m, k, n); plans[f'{m}x{k}x{n}'] = plan
    if plan is None: continue
    M, N, sizes = plan
    for q in sorted(set(sizes)):
        keys += [(0, 6, M, q), (1, 6, q, N)]
    keys.append((2, 6, M, N))
keys = sorted(set(keys))
json.dump(dict(cand=str(cand), keys=keys, plans={k: (list(v[:2]) + [list(v[2])] if v else None) for k, v in plans.items()}), open(out, 'w'), indent=1)
print(f'{len(keys)} phase-3 keys from {sum(p is not None for p in plans.values())}/{len(plans)} planned shapes -> {out.name}')
