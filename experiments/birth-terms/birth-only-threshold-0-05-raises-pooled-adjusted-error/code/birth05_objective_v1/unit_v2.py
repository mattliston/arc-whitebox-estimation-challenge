"""Actual selector checks and independent dense current-plane ownership."""
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def star_tensor(star):
    n = len(star)
    answer = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                answer[i, j, k] = ((star[i, j] if j == k else 0.) +
                                   (star[j, i] if i == k else 0.) +
                                   (star[k, i] if i == j else 0.)) / 3
    return answer


def main():
    build = json.loads((HERE / 'build.json').read_text())
    assert sha(HERE / 'candidate.py') == build['candidate_sha256']
    stub = types.ModuleType('whestbench')
    stub.BaseEstimator = type('BaseEstimator', (), {})
    sys.modules['whestbench'] = stub
    spec = importlib.util.spec_from_file_location('birth05_unit_candidate', HERE / 'candidate.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    selector_rows = []
    with flops.BudgetContext(flop_budget=2**36, quiet=True) as ctx:
        spec.loader.exec_module(mod)
        fixtures = [np.linspace(0, 1, 256), np.full(256, .5), np.zeros(256),
                    np.tile(np.array([0., .01, .049, .05, .051, .5, .949, .95,
                                      .951, .99, 1.]), 32)]
        for index, values in enumerate(fixtures):
            g = fnp.asarray(values, dtype=fnp.float64)
            indices, mask = mod._birth05_select(g)
            again, again_mask = mod._birth05_select(g)
            score = np.minimum(values, 1 - values)
            count = int(min(len(values), ((np.count_nonzero(score >= .05) + 31) // 32) * 32))
            expected_indices = np.argsort(-score, kind='stable')[:count]
            expected_mask = np.zeros(len(values), dtype=bool)
            expected_mask[expected_indices] = True
            assert np.array_equal(np.asarray(indices), expected_indices)
            assert np.array_equal(np.asarray(mask), expected_mask)
            assert np.array_equal(np.asarray(again), expected_indices)
            assert np.array_equal(np.asarray(again_mask), expected_mask)
            assert np.array_equal(np.asarray(g), values)
            parent_indices, parent_mask = mod.select(fnp.minimum(g, 1 - g), .01)
            assert np.all(~np.asarray(mask) | np.asarray(parent_mask))
            if index == 1:
                assert np.array_equal(np.asarray(indices), np.asarray(parent_indices))
            selector_rows.append(dict(fixture=index, size=len(values), selected=count,
                                      parent_selected=len(np.asarray(parent_indices))))
    selector_flops = ctx.summary_dict()['flops_used']

    rng = np.random.Generator(np.random.PCG64(2026092013))
    n = 7
    base = rng.normal(size=(n, n)) * .2
    np.fill_diagonal(base, 0)
    t, s = rng.normal(size=(2, n))
    coefficients = rng.normal(size=n)
    target = rng.normal(size=(n, n)) * .3
    inherited = rng.normal(size=(n, n)) * .2
    trace = rng.normal(size=(n, n)) * .1
    outputs = []
    scaled_errors, native_errors = [], []
    with flops.BudgetContext(flop_budget=2**36, quiet=True) as star_ctx:
        for selected in [np.ones(n, dtype=bool), np.array([1, 0, 1, 0, 0, 1, 0], dtype=bool)]:
            tensor = np.zeros((n, n, n))
            for i in np.flatnonzero(selected):
                plus, minus = base[:, i].copy(), base[:, i].copy()
                plus[i] += t[i]
                minus[i] += s[i]
                tensor += coefficients[i] * (np.einsum('i,j,k->ijk', plus, plus, plus) -
                                              np.einsum('i,j,k->ijk', minus, minus, minus))
            born_plane = np.einsum('iij->ij', tensor)
            rest = target - inherited - born_plane - trace
            star = 3 * rest.T - 2 * np.diag(np.diag(rest))
            total_plane = inherited + trace + np.einsum('iij->ij', tensor + star_tensor(star))
            error = float(np.linalg.norm(total_plane - target) / max(1., np.linalg.norm(target)))
            assert error <= 1e-11
            rr = fnp.asarray(rest, dtype=fnp.float64)
            native_star = (3 * rr.T - 2 * fnp.diag(fnp.diag(rr))).astype(fnp.float32)
            native_plane = inherited + trace + np.einsum('iij->ij', tensor + star_tensor(np.asarray(native_star).astype(np.float64)))
            native_error = float(np.linalg.norm(native_plane - target) / max(1., np.linalg.norm(target)))
            assert native_error <= 1e-6
            scaled_errors.append(error)
            native_errors.append(native_error)
            outputs.append(tensor + star_tensor(star))
    difference = outputs[0] - outputs[1]
    repeated_error = float(np.max(np.abs(np.einsum('iij->ij', difference))))
    assert repeated_error <= 1e-11
    distinct = np.array([difference[i, j, k] for i in range(n) for j in range(n)
                         for k in range(n) if len({i, j, k}) == 3])
    assert np.linalg.norm(distinct) > 1e-6
    assert mod._W5_PORT_SCOPE is None and mod._SCRATCH is None
    result = dict(passed=True, selector_rows=selector_rows, selector_flops=selector_flops,
                  star_fixture_flops=star_ctx.summary_dict()['flops_used'],
                  repeated_plane_scaled_errors=scaled_errors,
                  actual_fp32_star_scaled_errors=native_errors,
                  between_masks_repeated_plane_maxabs=repeated_error,
                  changed_all_distinct_tensor_norm=float(np.linalg.norm(distinct)),
                  whole_tensor_preserved=False, exact_repeat_and_input_preservation=True,
                  candidate_sha256=build['candidate_sha256'], source_sha256=sha(__file__),
                  truth_accessed=False, full_estimator_executed=False,
                  max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out = HERE / 'unit_v2'
    out.mkdir(exist_ok=False)
    with (out / 'result.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
