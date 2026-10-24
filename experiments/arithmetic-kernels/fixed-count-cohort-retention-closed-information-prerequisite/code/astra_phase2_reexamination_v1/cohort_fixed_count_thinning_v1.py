"""Coherent final-consumer variance of fixed-count complete-pair retention."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent


def statistics(effects):
    n, count = effects.shape
    assert count > 1
    total_norm = float(np.sum(effects * effects) / n)
    effect_sum = effects.sum(axis=1)
    common_norm = float(np.mean(effect_sum * effect_sum) / count)
    centered = effects - effects.mean(axis=1, keepdims=True)
    within = float(np.sum(centered * centered) / n)
    assert abs(within - (total_norm - common_norm)) <= 2e-12 * max(total_norm, 1e-30)
    coefficient = count * count * within / (count - 1)
    return dict(count=count, total_squared_rms=total_norm,
                common_squared_rms=common_norm, centered_squared_rms=within,
                coefficient=coefficient)


def allocate(coefficients, counts, total):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    assert len(counts) < total < counts.sum()
    roots = np.sqrt(coefficients)
    assert np.all(roots > 0)
    lo, hi = 0., max(float(roots.max()), 1e-30)
    for _ in range(100):
        tau = (lo + hi) / 2
        if np.clip(roots / tau, 1., counts).sum() > total:
            lo = tau
        else:
            hi = tau
    retained = np.clip(roots / hi, 1., counts)
    variance = float(np.sum(coefficients * (1 / retained - 1 / counts)))
    interior = (retained > 1 + 1e-9) & (retained < counts - 1e-9)
    derivative = coefficients / retained**2
    kkt = float(np.max(np.abs(derivative[interior] / hi**2 - 1))) if interior.any() else 0.
    assert abs(retained.sum() - total) <= 1e-8
    assert kkt <= 1e-10
    return retained, variance, dict(tau=hi, relative_interior_kkt_error=kkt)


def tiny_certificate():
    rng = np.random.Generator(np.random.PCG64DXSM(930397))
    cohorts = [rng.normal(size=(5, n)) + 10 * rng.normal(size=(5, 1)) for n in (4, 6)]
    kept = (2, 3)
    rows = [statistics(v) for v in cohorts]
    predicted = sum(r['coefficient'] * (1 / m - 1 / r['count']) for r, m in zip(rows, kept))
    choices = [list(itertools.combinations(range(v.shape[1]), m)) for v, m in zip(cohorts, kept)]
    exact_sum = sum(v.sum(axis=1) for v in cohorts)
    enumerated = 0.
    patterns = 0
    for pattern in itertools.product(*choices):
        estimate = sum(v[:, ids].sum(axis=1) * v.shape[1] / m
                       for v, ids, m in zip(cohorts, pattern, kept))
        enumerated += float(np.mean((estimate - exact_sum)**2))
        patterns += 1
    enumerated /= patterns
    relative = abs(predicted - enumerated) / predicted
    alloc, optimized, kkt = allocate([r['coefficient'] for r in rows],
                                    [r['count'] for r in rows], sum(kept))
    passed = relative <= 1e-11 and optimized <= predicted * (1 + 1e-12)
    assert passed
    return dict(passed=passed, patterns=patterns, formula=predicted,
                enumeration=enumerated, relative_error=relative,
                continuous_retention=alloc.tolist(), optimized_variance=optimized,
                kkt=kkt)


def wide():
    # Import only after the root has allocated the MPS slot to this worker.
    import torch
    from paired_cp_thinning_bound_v1 import capture, final_jacobian, relative
    assert 'threadpoolctl' not in sys.modules
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0'
    torch.set_num_threads(3)
    records, arrays, max_gpu = [], {}, 0
    for rank in (35, 100):
        tape, fidelity, gpu = capture(rank)
        max_gpu = max(max_gpu, gpu)
        direct, left, right, jacobian = final_jacobian(tape, rank)
        rows, offset, total_cp = [], 0, np.zeros(1024)
        for layer, count in enumerate(tape['counts']):
            plus = tape['pp'][:, offset:offset + count]
            minus = tape['pp'][:, offset + count:offset + 2 * count]
            weight = tape['lam'][offset:offset + count]
            assert np.array_equal(weight, -tape['lam'][offset + count:offset + 2 * count])
            common, difference = (plus + minus) / 2, plus - minus
            atoms = weight * (3 * common * common * difference + .25 * difference**3)
            total_cp += atoms.sum(axis=1)
            effects = direct[:, None] * atoms + left[:, None] * (right @ atoms)[None, :]
            row = dict(layer=layer, **statistics(effects))
            rows.append(row)
            arrays[f'cohort_effect_sum_{rank}_{layer}'] = effects.sum(axis=1)
            offset += 2 * count
        assert offset == len(tape['lam'])
        original_cp = (tape['pp']**3 * tape['lam']).sum(axis=1)
        identity_error = relative(total_cp, original_cp)
        assert identity_error <= 1e-10
        counts = np.array([r['count'] for r in rows])
        coefficients = np.array([r['coefficient'] for r in rows])
        retained, optimized, kkt = allocate(coefficients, counts, counts.sum() / 2)
        uniform = float(np.sum(coefficients / counts))
        independent_uniform = float(sum(r['total_squared_rms'] for r in rows))
        record = dict(rank=rank, pair_count=int(counts.sum()), retained_total=float(retained.sum()),
                      optimized_continuous_local_added_mse=optimized,
                      uniform_half_per_cohort_local_added_mse=uniform,
                      coherent_independent_uniform_half_local_added_mse=independent_uniform,
                      common_fraction=sum(r['common_squared_rms'] for r in rows) / independent_uniform,
                      continuous_retained_by_cohort=retained.tolist(), cohorts=rows,
                      threshold=1e-9, passed=bool(optimized <= 1e-9),
                      signed_pair_identity_relative_error=identity_error,
                      forward_fidelity=fidelity, jacobian_certificate=jacobian, kkt=kkt)
        records.append(record)
        for name, values in dict(counts=counts, coefficients=coefficients,
                                 retained=retained, direct=direct, left=left, right=right).items():
            arrays[f'{name}_{rank}'] = values
        print(json.dumps({k: v for k, v in record.items() if k != 'cohorts'}), flush=True)
        del tape, plus, minus, atoms, effects, common, difference
        gc.collect()
    return records, arrays, max_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=('tiny', 'wide'))
    args = parser.parse_args()
    output = HERE / f'cohort_fixed_count_thinning_{args.mode}_v1.json'
    assert not output.exists()
    assert not output.with_suffix('.npz').exists()
    started = time.time()
    tiny = tiny_certificate()
    result = dict(tiny_certificate=tiny, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  target_access=False, sampling_executed=False)
    if args.mode == 'wide':
        records, arrays, gpu = wide()
        np.savez(output.with_suffix('.npz'), **arrays)
        result.update(records=records, passed=all(r['passed'] for r in records),
                      largest_observed_gpu_driver_bytes=gpu,
                      arrays_sha256=hashlib.sha256(output.with_suffix('.npz').read_bytes()).hexdigest(),
                      qualification='One coherent current-bank capture per net. Exact local final-consumer variance for uniform fixed-count sampling within each cohort; continuous oracle cohort allocation is a lower bound on integer allocation. All earlier-layer errors, acquisition, feedback, costs and runtime are free. No carrier, finite perturbation, fit or score claim. Process/GPU memory observations are not independent additive peaks.')
    else:
        result['passed'] = tiny['passed']
    result.update(utc_finished=datetime.now(timezone.utc).isoformat(),
                  wall_seconds=time.time() - started,
                  peak_process_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
