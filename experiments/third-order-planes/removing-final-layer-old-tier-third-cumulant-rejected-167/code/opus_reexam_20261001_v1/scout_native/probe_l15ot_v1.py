"""TAG scout-native probe_l15ot_v1 (2026-10-12): measurement only (no file of any lane changed). Lever L15-OT: at layer 15 only,
replace the old tier's reconstruction on the row-cut rows by zeros and repair the k3 diagonal with the native first-chaos trace
(the OTT estimate 3 rho U_i . F (lam o |F|^2), every row), as the old-tier tails already do at 7-14. The L15 K4L / head columns then
see no old-tier columns (their old part is dropped, not estimated). One prediction per process; mode 'base' = the file unchanged.
Writes final + all-layer predictions, bill, MSE and the old-tier share of the L15 k3 diagonal (exact vs trace) to <out>.json/.npz.
usage: probe_l15ot_v1.py <candidate.py> <seed> <base|ot> <out.json>"""
import hashlib
import json
import sys
import time
import types
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

O = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(O / 'analyst/fresh'))
import loader

candidate, seed, mode, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
assert mode in ('base', 'ot') and not out.exists()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_l15ot'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))

INFO = {}
REPAIR = [None]
parent_recon = mod.__dict__['_slr_reconstruct']
parent_l15 = mod.__dict__['_l15_rowcut_k3']


def trace_delta(U, state, lam_all):
    # the OTT estimate on EVERY row of U (rows x rank), as _slr_reconstruct's tail branch computes it on the tail rows
    columns = int(state['columns'])
    lam = lam_all[:columns]
    group = state['lazy']
    rows = fnp.arange(int(U.shape[0]))
    if group is None:
        return mod._ott_trace(U, state['F'], lam, rows)
    mapped = mod._shared_old384_mm(U, group['map'])
    split = int(group['prefix'])
    return mod._ott_trace(mapped, group['base'], lam[:split], rows) + mod._ott_trace(U, group['join'], lam[split:], rows)


def np_trace(U, F, lam):
    rho = float((U * U).sum()) / (U.shape[0] * U.shape[1])
    return (U @ (F @ (lam * (F * F).sum(axis=0)))) * (3 * rho)


def recon(U, state):
    if mod._UVP_CUR[0] != 15 or INFO.get('lam') is None:
        return parent_recon(U, state)
    if mode == 'base':
        exact = parent_recon(U, state)
        cols = int(state['columns'])
        lam = np.asarray(INFO['lam'], dtype=np.float64)[:cols]
        X = np.asarray(exact, dtype=np.float64)
        INFO['k3_old'] = (X ** 3 * lam).sum(axis=1)
        Un = np.asarray(U, dtype=np.float64)
        group = state['lazy']
        if group is None:
            INFO['k3_trace'] = np_trace(Un, np.asarray(state['F'], dtype=np.float64), lam)
        else:
            split = int(group['prefix'])
            mapped = Un @ np.asarray(group['map'], dtype=np.float64)
            INFO['k3_trace'] = (np_trace(mapped, np.asarray(group['base'], dtype=np.float64), lam[:split])
                                + np_trace(Un, np.asarray(group['join'], dtype=np.float64), lam[split:]))
        return exact
    REPAIR[0] = trace_delta(U, state, INFO['lam'])
    return fnp.zeros((int(U.shape[0]), int(state['columns'])), dtype=fnp.float32)


def l15(layer, mu, v, w32, p, previous, newborn, lam, state, star):
    INFO['lam'] = lam
    k3 = parent_l15(layer, mu, v, w32, p, previous, newborn, lam, state, star)
    INFO['k3_total'] = np.asarray(k3, dtype=np.float64)
    if mode == 'ot' and REPAIR[0] is not None:
        z = fnp.abs(mu / fnp.sqrt(fnp.maximum(v, 1e-10)))
        rows, _m = mod.select(mod._L15_ROWCUT_Z - z, 0.0)
        k3 = fnp.add(k3, 0.0)
        fnp.put(k3, rows, k3[rows] + REPAIR[0])
        REPAIR[0] = None
    return k3


mod.__dict__['_slr_reconstruct'] = recon
mod.__dict__['_l15_rowcut_k3'] = l15
w = loader.weights(seed)
truth = loader.truth(seed, check_weights=False)['mc_post']
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{seed}')
t0 = time.monotonic()
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
bill = int(ctx.summary_dict()['flops_used'])
mse = ((y - truth) ** 2).mean(axis=1)
res = dict(candidate=str(candidate), seed=seed, mode=mode, bill=bill, mse_final=float(mse[15]), mse_layers=[float(x) for x in mse],
           wall_s=round(time.monotonic() - t0, 1), pred_sha=hashlib.sha256(y.astype(np.float32).tobytes()).hexdigest()[:16])
if 'k3_old' in INFO:
    ko, kt = INFO['k3_old'], INFO['k3_trace']
    res.update(k3_old_rms=float(np.sqrt((ko ** 2).mean())), k3_trace_rms=float(np.sqrt((kt ** 2).mean())),
               k3_old_trace_corr=float(np.corrcoef(ko, kt)[0, 1]), k3_old_minus_trace_rms=float(np.sqrt(((ko - kt) ** 2).mean())),
               k3_total_rms=float(np.sqrt((INFO['k3_total'] ** 2).mean())))
np.savez(str(out).replace('.json', '.npz'), y=y.astype(np.float32))
with out.open('x') as f:
    json.dump(res, f)
print(json.dumps(res), flush=True)
