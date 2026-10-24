import argparse
import importlib.util
import resource
import sys
import time
import types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--ranks', required=True)
ap.add_argument('--disabled', action='store_true')
args = ap.parse_args()
seal_check(); done('unit'); assert read(HERE / 'unit.json')['passed']
assert all(t['num_threads'] == 1 for t in threadpool_info())
meta = {r['rank']: r for r in read(HERE / 'inputs.json')['rows']}
build = read(HERE / 'build.json'); check(build['candidate'])
out = HERE / args.out; out.mkdir(exist_ok=False)
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
spec = importlib.util.spec_from_file_location('cubic_support_candidate', build['candidate']['path'])
mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
records, traces, cohorts = [], [], []
for index, rank in enumerate(map(int, args.ranks.split(','))):
    row = meta[rank]; check(row['weights'])
    w = np.load(row['weights']['path'], allow_pickle=False)
    assert w.shape == (16, 1024, 1024) and w.dtype == np.float32
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ingress:
        ww = [fnp.asarray(x) for x in w]
    assert ingress.summary_dict()['flops_used'] == 0
    mlp = types.SimpleNamespace(width=1024, depth=16, weights=ww, seed=0, name=f'rank{rank}')
    traces.clear(); cohorts.clear(); start = time.monotonic()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        if index == 0:
            spec.loader.exec_module(mod); mod._FULL_CUBIC_SUPPORT = not args.disabled
            original_select = mod._cubic_support_select
            original_commit = mod._shared_old384_commit
            def observe_select(score, floor):
                answer = original_select(score, floor)
                traces.append(answer)
                return answer
            def observe_commit(layer, gate, alive, selected, state):
                answer = original_commit(layer, gate, alive, selected, state)
                cohorts.append(dict(layer=int(layer), birth_count=int(selected.shape[0]),
                    old_columns=int(state['columns']), total_columns=int(state['next_column'])))
                return answer
            mod._cubic_support_select = observe_select
            mod._shared_old384_commit = observe_commit
            estimator = mod.Estimator()
        result = estimator.predict(mlp, 2**41)
    wall = time.monotonic() - start
    bill = int(ctx.summary_dict()['flops_used'])
    assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
    pred = np.asarray(result).copy()
    assert pred.shape == (16, 1024) and pred.dtype == np.float32
    assert np.isfinite(pred).all() and (pred >= 0).all()
    check(row['prediction']); parent = np.load(row['prediction']['path'], allow_pickle=False)
    assert pred[0].tobytes() == parent[0].tobytes()
    if args.disabled:
        assert pred.tobytes() == parent.tobytes() and bill == row['parent']['flops']
    assert len(traces) == 30 and [r['layer'] for r in cohorts] == list(range(15))
    roster = []
    for slot, (indices, mask) in enumerate(traces):
        ii, mm = np.asarray(indices), np.asarray(mask)
        assert ii.ndim == 1 and mm.shape == (1024,) and mm.dtype == bool
        expected_mask = np.zeros(1024, bool); expected_mask[ii] = True
        assert np.array_equal(mm, expected_mask) and len(np.unique(ii)) == len(ii)
        if not args.disabled:
            assert len(ii) == 1024 and mm.all() and np.array_equal(np.sort(ii), np.arange(1024))
        roster.append(dict(layer=slot//2, role=['live', 'response'][slot%2], count=int(len(ii)),
            indices_sha256=__import__('hashlib').sha256(ii.tobytes()).hexdigest(),
            mask_sha256=__import__('hashlib').sha256(mm.tobytes()).hexdigest()))
    path = out / f'call{index:02d}_rank{rank:04d}.npy'; np.save(path, pred)
    record = dict(index=index, rank=rank, prediction=pin(path), weights=row['weights'],
        flops=bill, multiplier=max(.1, bill/2**41), wall_s=wall, includes_initialization=index == 0,
        input_flops=0, numeric_cache_retired=True, first_parent_row_exact=True,
        full_cubic_support=not args.disabled, rosters=roster, cohorts=list(cohorts),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(out / f'call{index:02d}.json', record); records.append(record)
    print(dict(rank=rank, index=index, flops=bill, wall_s=wall,
        full_cubic_support=not args.disabled), flush=True)
    traces.clear(); cohorts.clear()
    del w, ww, mlp, result, pred, parent, indices, mask, ii, mm
write(out / 'result.json', dict(records=records, completed=True,
    candidate=build['candidate'], targets_decoded=False))

