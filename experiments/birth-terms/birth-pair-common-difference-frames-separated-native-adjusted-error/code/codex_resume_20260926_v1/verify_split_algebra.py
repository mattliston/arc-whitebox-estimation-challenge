"""Compare the materialized and lazy structured reads to dense pair algebra."""
import json
from pathlib import Path
import sys
import types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

HERE = Path(__file__).resolve().parent
stub = types.ModuleType('whestbench')
stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
flops.configure(symmetry_warnings=False)
mod = types.ModuleType('split_test')
mod.__file__ = str(HERE / 'candidate_split_v1.py')
sys.modules[mod.__name__] = mod
exec(compile(Path(mod.__file__).read_text(), mod.__file__, 'exec'), mod.__dict__)
rng = np.random.default_rng(9262026)
arr = lambda shape: rng.normal(size=shape).astype(np.float32)
results = []
for lazy in (False, True):
    with flops.BudgetContext(flop_budget=2**35, quiet=True):
        state = dict(columns=20, cohorts=[(0, 8), (8, 12)])
        dense_parts = []
        parts = []
        for rank in (4, 6):
            if lazy:
                base, join, rot = arr((8, 4)), arr((rank, 6)), arr((rank, 8))
                F = np.concatenate((rot @ base, join), axis=1)
                part = dict(rank=rank, F=None, lazy=dict(base=fnp.asarray(base), join=fnp.asarray(join), map=fnp.asarray(rot)))
            else:
                F = arr((rank, 10))
                part = dict(rank=rank, F=fnp.asarray(F), lazy=None)
            parts.append(part)
            dense_parts.append(F)
        state['uv2'] = dict(u=parts[0], v=parts[1])
        U = arr((16, 10))
        pu, pv = U[:, :4] @ dense_parts[0], U[:, 4:] @ dense_parts[1]
        dense = np.concatenate((pu[:, :4]+pv[:, :4], pu[:, :4]-pv[:, :4],
                                pu[:, 4:]+pv[:, 4:], pu[:, 4:]-pv[:, 4:]), axis=1)
        got = np.asarray(mod._uv2_reconstruct(fnp.asarray(U), state))
        np.testing.assert_allclose(got, dense, rtol=2e-5, atol=2e-5)
        lam = arr((20,))  # arbitrary signed coefficients, stronger than paired.
        L2 = arr((9, 20))**2
        inner = np.asarray(mod._cp_slr_inner_sq(fnp.asarray(L2), fnp.asarray(lam), state))
        plane = inner @ U.T
        ref = (L2 * lam) @ dense.T
        np.testing.assert_allclose(plane, ref, rtol=3e-5, atol=2e-4)
        results.append(dict(lazy=lazy, reconstruction_max_abs=float(np.max(np.abs(got-dense))),
                            read_max_abs=float(np.max(np.abs(plane-ref)))))
path = HERE / 'split_algebra_result.json'
with path.open('x') as f:
    json.dump(dict(passed=True, cases=results), f, indent=2)
print(json.dumps(dict(passed=True, cases=results)))
