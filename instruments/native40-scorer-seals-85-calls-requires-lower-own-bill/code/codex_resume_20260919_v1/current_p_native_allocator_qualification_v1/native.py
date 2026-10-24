"""Complete actual estimator bills; only passive domain/selector references."""
import argparse
import gc
import importlib.util
import resource
import time
import types
from endpoint_common import *
import weakref
from lifecycle import reset_saved_history

ap = argparse.ArgumentParser()
ap.add_argument('role', choices=('baseline', 'relief'))
args = ap.parse_args()
endpoint_verify()
done('build')
build = read(HERE / 'build.json')
assert build['passed'] and build['only_gnn_literal_changed']
inputs = read(HERE / 'inputs.json')
ranks, source = [37, 39, 42, 37], build['candidate']
if args.role == 'relief':
    done('qualify_baseline')
    assert read(HERE/'baseline_qualified.json')['passed']
from retirement import Allocator, rss
allocator = Allocator()
check(source)
out = HERE / f'native_{args.role}'
out.mkdir(exist_ok=False)
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
assert np.__version__ == '2.4.6' and all(t['num_threads'] == 1 for t in threadpool_info())
stub = types.ModuleType('whestbench')
stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
spec = importlib.util.spec_from_file_location('current_p_fit_native', source['path'])
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
variances, selectors, records = [], [], []


def array_hash(x):
    return hashlib.sha256(x.tobytes()).hexdigest()


for index, rank in enumerate(ranks):
    wp = inputs['weights'][str(rank)]
    check(wp)
    weights = np.load(wp['path'], allow_pickle=False)
    assert weights.shape == (16, 1024, 1024) and weights.dtype == np.float32
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ingress:
        ww = [fnp.asarray(w) for w in weights]
    assert ingress.summary_dict()['flops_used'] == 0
    input_bill_path = out / f'call_{index:02d}_input_bill.json'
    write(input_bill_path, dict(flops=0, charged_to_participant=False, context_closed=True,
        input_preparation_unchanged=True, weights=wp))
    input_ref = weakref.ref(ingress)
    input_reset = reset_saved_history(flops, ingress, input_bill_path)
    del ingress
    gc.collect()
    input_reset.update(complete=True, context_released=input_ref() is None, index=index, rank=rank)
    input_reset_path = out / f'call_{index:02d}_input_reset.json'
    write(input_reset_path, input_reset)
    assert input_reset['closed_context_bill_unchanged_by_reset'] and input_reset['context_released']
    mlp = types.SimpleNamespace(width=1024, depth=16, weights=ww, seed=0, name=f'rank{rank}')
    variances.clear()
    selectors.clear()
    start = time.monotonic()
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        if index == 0:
            spec.loader.exec_module(mod)
            original_gaussian, original_select = mod.gaussian_raw, mod.select
            def gaussian(mu, v):
                variances.append(v)
                return original_gaussian(mu, v)
            def select(score, floor):
                answer = original_select(score, floor)
                selectors.append((score, floor, answer))
                return answer
            mod.gaussian_raw, mod.select = gaussian, select
            estimator = mod.Estimator()
        prediction = estimator.predict(mlp, 2**41)
    wall = time.monotonic() - start
    flops_used = int(ctx.summary_dict()['flops_used'])
    assert 0 < flops_used <= 2**41 and wall < 120
    assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
    assert len(variances) == 16 and len(selectors) == 45
    domains, cells = {}, {}
    for layer, v in enumerate(variances):
        x = np.asarray(v)
        assert x.shape == (1024,) and np.isfinite(x).all() and x.min() >= 1e-10
        domains[str(layer)] = dict(minimum=float(x.min()), finite=True, sha256=array_hash(x))
    for k, (score, floor, (ids, mask)) in enumerate(selectors):
        s, ix, live = np.asarray(score), np.asarray(ids), np.asarray(mask)
        assert s.shape == live.shape == (1024,) and np.isfinite(s).all() and floor == .01
        count = min(1024, ((int(np.count_nonzero(s >= floor)) + 31) // 32) * 32)
        expected_ix = np.argsort(-s, kind='stable')[:count]
        expected_live = np.zeros(1024, dtype=bool)
        expected_live[expected_ix] = True
        assert ix.shape == (count,) and np.array_equal(ix, expected_ix) and np.array_equal(live, expected_live)
        cells[f'{k // 3}:{k % 3}'] = dict(count=count, floor=floor,
            indices_sha256=array_hash(ix), mask_sha256=array_hash(live), score_sha256=array_hash(s))
    value = np.asarray(prediction).copy()
    assert value.shape == (16, 1024) and value.dtype == np.float32 and np.isfinite(value).all() and (value >= 0).all()
    output_path = out / f'call_{index:02d}_rank{rank:04d}.npy'
    np.save(output_path, value, allow_pickle=False)
    bill_path = out / f'call_{index:02d}_bill.json'
    write(bill_path, dict(flops=flops_used, multiplier=max(.1, flops_used / 2**41),
        includes_initialization=index == 0, input_flops_separate=0, wall_s=wall,
        complete_counted_predict=True, initialization_subtracted=False, extra_participant_array_operations=0))
    context_ref = weakref.ref(ctx)
    lifecycle = reset_saved_history(flops, ctx, bill_path)
    del ctx
    gc.collect()
    lifecycle.update(complete=True, context_released=context_ref() is None, index=index, rank=rank)
    lifecycle_path = out / f'call_{index:02d}_lifecycle.json'
    write(lifecycle_path, lifecycle)
    assert lifecycle['closed_context_bill_unchanged_by_reset'] and lifecycle['context_released']
    record = dict(index=index, rank=rank, prediction=pin(output_path), bill=pin(bill_path), weights=wp,
        input_bill=pin(input_bill_path), input_reset=pin(input_reset_path), lifecycle=pin(lifecycle_path),
        flops=flops_used, multiplier=max(.1, flops_used / 2**41), includes_initialization=index == 0,
        input_flops_separate=0, wall_s=wall, domain=domains, selectors=cells,
        numeric_cache_retired=True, cache=mod._W5_PORT_LAST, passive_extra_fnp_ops=0,
        truth_accessed=False, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if index == len(ranks) - 1:
        assert rank == ranks[0] and record['prediction']['sha256'] == records[0]['prediction']['sha256']
        assert domains == records[0]['domain'] and cells == records[0]['selectors']
    write(out / f'call_{index:02d}.json', record)
    records.append(record)
    print(dict(role=args.role, index=index, rank=rank, flops=flops_used, wall_s=wall), flush=True)
    tracked = [weakref.ref(weights), weakref.ref(prediction)] + [weakref.ref(w) for w in ww]
    variances.clear()
    selectors.clear()
    del prediction, value, weights, ww, mlp, v, x, score, ids, mask, s, ix, live, expected_ix, expected_live
    gc.collect()
    alive = [i for i,ref in enumerate(tracked) if ref() is not None]
    before_rss = rss()
    assert not alive, ('call objects retained', index, alive)
    saved_bill = pin(bill_path)
    released = allocator.relieve() if args.role == 'relief' else None
    after_rss = rss()
    assert pin(bill_path) == saved_bill
    write(out / f'call_{index:02d}_retirement.json', dict(complete=True, index=index, rank=rank,
        all_tracked_call_objects_released=True, tracked_objects=18, live_objects=alive,
        after_closed_saved_bill=True, saved_bill=saved_bill, saved_bill_unchanged=True,
        allocator_called=args.role == 'relief', allocator_reported_released_bytes=released,
        before_rss_bytes=before_rss, after_rss_bytes=after_rss,
        allocator_statistics_called=False, candidate_state_reset=False))
mod.gaussian_raw, mod.select = original_gaussian, original_select
check(source)
write(out / 'result.json', dict(completed=True, records=records, source=source, role=args.role,
    full_estimator_calls=len(records), truth_accessed=False, passive_extra_fnp_ops=0,
    all_16_domains_and_45_selectors_qualified=True, passive_observers_restored=True,
    repeat_output_domain_selector_bytes_exact=True))
