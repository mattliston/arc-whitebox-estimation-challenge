"""Meter a fixed companion feature while preserving the exact Lazy output."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
import types

HERE = Path(__file__).resolve().parent
D = HERE.parent
P2 = D.parent
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
PARENT = D/'lazy_rotation_objective_v1/candidate.py'
PARENT_SHA = 'b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--ranks', required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    from feature_fnp import OddLineageTap, observed_push
    assert all(t['num_threads'] == 1 for t in threadpool_info())
    assert sha(PARENT) == PARENT_SHA
    observed, observed_proof = observed_push(PARENT.read_text())
    seal = json.loads((HERE/'sources_sealed.json').read_text())
    for name, value in seal['artifacts_sha256'].items():
        assert sha(HERE/name) == value, name
    stub = types.ModuleType('whestbench')
    stub.BaseEstimator = type('BaseEstimator', (), {})
    sys.modules['whestbench'] = stub
    spec = importlib.util.spec_from_file_location('shared_fourth_odd_parent', PARENT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    manifest = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
    ranks = list(map(int, args.ranks.split(',')))
    assert ranks in ([35, 100, 35], [100])
    baseline_path = D/'lazy_rotation_v1'/('dev' if len(ranks) == 3 else 'cold100')/'result.json'
    baseline = json.loads(baseline_path.read_text())['records']
    records, previous = [], {}
    for call, rank in enumerate(ranks):
        wp = (STORE/manifest[rank]['path']).parent/'weights.npy'
        weights = np.load(wp)
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ingress:
            ww = [fnp.asarray(weights[l]) for l in range(16)]
        mlp = types.SimpleNamespace(width=1024, depth=16, weights=ww, seed=0, name=f'rank{rank}')
        started = time.monotonic()
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            if call == 0:
                spec.loader.exec_module(mod)
                estimator = mod.Estimator()
            tap = OddLineageTap(mod, ctx, observed)
            tap.attach()
            try:
                prediction = estimator.predict(mlp, 2**41)
            finally:
                tap.detach()
        elapsed = time.monotonic()-started
        total_flops = ctx.summary_dict()['flops_used']
        parent_flops = total_flops-tap.extra_flops
        assert set(tap.held) == {'delta_k3', 'sigma', 'feature'} and tap.pending is None
        assert tap.saved is None and tap.u is None
        arrays = {k: np.asarray(v).copy() for k, v in tap.held.items()}
        arrays['prediction'] = np.asarray(prediction, dtype=np.float32).copy()
        for key, array in arrays.items():
            expected = (16, 1024) if key == 'prediction' else (1024,)
            assert array.shape == expected and np.isfinite(array).all(), key
        assert (arrays['prediction'] >= 0).all() and (arrays['sigma'] > 0).all()
        if baseline is not None:
            parent = baseline[call]
            assert rank == parent['rank'] and sha(wp) == parent['weights_sha256']
            assert arrays['prediction'].tobytes() == np.load(parent['output']).tobytes()
            assert parent_flops == parent['flops'], (parent_flops, parent['flops'], tap.extra_flops)
        if rank in previous:
            assert all(a.dtype == previous[rank][k].dtype and a.tobytes() == previous[rank][k].tobytes() for k, a in arrays.items())
        previous[rank] = arrays
        assert mod._W5_PORT_SCOPE is None and mod._W5_PORT_LAST['completed'] and mod._W5_PORT_LAST['numeric_cache_retired']
        target = args.out/f'call_{call:02d}.npz'
        np.savez(target, **arrays)
        rec = dict(call=call, rank=rank, weights_sha256=sha(wp), parent_sha256=PARENT_SHA,
                   capture=str(target.resolve()), capture_sha256=sha(target), parent_flops=parent_flops,
                   feature_flops=tap.extra_flops, total_flops=total_flops,
                   parent_multiplier=max(.1, parent_flops/2**41), includes_initialization=call == 0,
                   capture_wall_s=elapsed, wall_includes_feature=True, input_flops=ingress.summary_dict()['flops_used'],
                   truth_accessed=False, bindings_restored=True, numeric_cache_retired=True,
                   feature_rms=float(np.sqrt(np.mean(arrays['feature']**2))),
                   feature_maxabs=float(np.max(np.abs(arrays['feature']))),
                   cohort_sizes=[int(g.shape[0]) for _, g in tap.cohorts],
                   max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        records.append(rec)
        print(json.dumps(rec), flush=True)
        tap.held.clear()
        tap.cohorts.clear()
        del tap, prediction, weights, ww, mlp, arrays
    result = dict(completed=True, source_sha256=sha(__file__), protocol_sha256=sha(HERE/'PROTOCOL.md'),
                  helper_sha256=sha(HERE/'feature_fnp.py'), parent_sha256=PARENT_SHA, records=records,
                  repeat_byte_exact=ranks == [35, 100, 35], parent_all16_and_flops_exact=baseline is not None,
                  parent_source_file_unchanged=sha(PARENT) == PARENT_SHA, truth_accessed=False,
                  observed_source=observed_proof)
    with (args.out/'completed.json').open('x') as stream:
        json.dump(result, stream, indent=2)


if __name__ == '__main__':
    main()
