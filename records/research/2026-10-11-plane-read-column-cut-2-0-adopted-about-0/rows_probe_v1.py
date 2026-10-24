"""TAG opus_engineer a62 rows probe (lead 10-11 RESTART, the young READ's rows): at every bank read of one in-process prediction,
the read's row roster against |a| = |mu / sigma| and the covariance-channel weight of each row.

usage: rows_probe_v1.py <candidate.py> <comma ranks> <out.json>
Per layer: response rows (the read's rows), live columns, young / old columns, rows with a >= -3.09 (the dead-side push cut);
for z in ZS the fraction of response rows with |a| > z, and their share of the covariance-channel energy of the read's plane,
E_i = (pdf_i / (2 s_i))^2 sum_j (plane_ij g_j)^2 (the e21 term's leading coefficient pdf_i g_j / (2 s_i) on K21_ij; the
plane = young + old read, before the diagonal). Measurement only: the wrapped read returns the parent's answer unchanged.
"""
from pathlib import Path
import json
import sys
import types
import numpy as np
import flopscope as flops

P2 = Path(__file__).resolve().parents[3]
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
candidate = Path(sys.argv[1]).resolve()
ranks = [int(x) for x in sys.argv[2].split(',')]
out = Path(sys.argv[3]).resolve()
assert not out.exists()
ZS = (1.6, 1.8, 2.0, 2.1, 2.2)
manifest = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_rows'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2',
                                scratch_dir=None, submission_dir=None, seed=0))
parent = mod._cp_read_young_sq
records = []


def probe(left, young_right, lam, live_indices, state):
    plane = parent(left, young_right, lam, live_indices, state)
    layer = int(sys._getframe(1).f_locals['l'])
    L = mod._E2_LAYER
    a = np.asarray(L['a'], dtype=np.float64); s = np.asarray(L['s'], dtype=np.float64); g = np.asarray(L['g'], dtype=np.float64)
    rows = np.asarray(L['ecut_rows']); cols = np.asarray(live_indices)
    P = np.asarray(plane, dtype=np.float64)
    pdf = np.exp(-0.5 * a[rows] ** 2) / np.sqrt(2 * np.pi)
    energy = (pdf / (2 * s[rows])) ** 2 * ((P * g[cols][None, :]) ** 2).sum(axis=1)
    rec = dict(layer=layer, resp=int(rows.shape[0]), live=int(cols.shape[0]), young=int(young_right.shape[1]),
               old=int(state['columns']), push_rows=int((a >= -3.0902).sum()), absa_max=float(np.abs(a[rows]).max()))
    for z in ZS:
        cut = np.abs(a[rows]) > z
        rec['rows_gt_%.1f' % z] = float(cut.mean())
        rec['energy_gt_%.1f' % z] = float(energy[cut].sum() / energy.sum())
    records.append(rec)
    return plane


mod._cp_read_young_sq = probe
nets = []
for rank in ranks:
    weights = np.load(STORE/Path(manifest[rank]['path']).parent/'weights.npy', allow_pickle=False)
    mlp = types.SimpleNamespace(width=1024, depth=16, weights=[weights[l] for l in range(16)], seed=0, name=f'phase2-{rank}')
    start = len(records)
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
    for r in records[start:]:
        r['rank'] = rank
        print(json.dumps(r), flush=True)
    nets.append(dict(rank=rank, flops=int(ctx.summary_dict()['flops_used'])))
with out.open('x') as stream:
    json.dump(dict(candidate=str(candidate), nets=nets, records=records), stream)
print(json.dumps(dict(done=True, nets=nets)), flush=True)
