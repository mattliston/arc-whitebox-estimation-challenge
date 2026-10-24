"""Inert until --execute: fixed tiny explicit-tensor/FNP certificate, no targets."""
import argparse
import gc
import json
import signal
import time
import traceback
from types import SimpleNamespace
import common as h


def dense_birth(np, base, d, gamma):
    # Deliberately literal index construction, independent of J@B contraction.
    n = len(d)
    tensor = np.zeros((n, n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for a in range(n):
                    tensor[i, j, k] += d[a] * (
                        (i == a) * base[j, a] * base[k, a]
                        + (j == a) * base[i, a] * base[k, a]
                        + (k == a) * base[i, a] * base[j, a])
                if i == j == k:
                    tensor[i, j, k] += gamma[i]
    return tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        parser.print_help()
        return 0
    environment, bindings = h.runtime(), h.sources()
    out = h.HERE / 'small'
    out.mkdir()
    start = time.monotonic()
    result = dict(passed=False, completed=False, source_bindings=bindings,
                  environment=environment, fixture=dict(seed=930915, width=5, depth=4),
                  truth_accessed=False, current_network_accessed=False,
                  candidate_sha256=h.sha(h.HERE / 'estimator.py'), started_utc=h.utc())
    h.write(out / 'started.json', result)
    h.alarm(60)
    try:
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from threadpoolctl import threadpool_info, threadpool_limits
        from whestbench.scoring import materialise_predictions
        flops.configure(symmetry_warnings=False)
        core = h.load_estimator()
        with threadpool_limits(limits=1):
            result['threadpools'] = threadpool_info()
            h.require(all(x['num_threads'] == 1 for x in result['threadpools']
                          if x.get('user_api') == 'blas'), 'BLAS1')
            rng = np.random.default_rng(930915)
            raw = (rng.normal(size=(4, 5, 5)) * np.sqrt(2 / 5)).astype(np.float32)
            with flops.BudgetContext(flop_budget=h.BUDGET, wall_time_limit_s=60, quiet=True) as ctx:
                weights = [fnp.array(w, dtype=fnp.float32) for w in raw]
                state = core.forecast(weights, True)
                predicted_k3 = core.terminal_skew(weights, state)
                mlp = SimpleNamespace(weights=weights, depth=4, width=5)
                baseline = core.GaussianEstimator()
                candidate = core.Estimator()
                y0 = materialise_predictions(baseline.predict(mlp, h.BUDGET), depth=4, width=5)
                y1 = materialise_predictions(candidate.predict(mlp, h.BUDGET), depth=4, width=5)
            result['bill'] = h.metrics(ctx, start)
            h.write(out / 'bill.json', result['bill'])
            ref, lifecycle = h.reset_closed(flops, ctx, out / 'bill.json')
            del ctx
            gc.collect()
            lifecycle['context_released'] = ref() is None
            result['lifecycle'] = lifecycle
            h.require(lifecycle['context_released'], 'Closed context released')
            births = [(np.asarray(b).astype(np.float64), np.asarray(d).copy(), np.asarray(g).copy())
                      for b, d, g in state['births']]
            gates = [np.asarray(g).astype(np.float64) for g in state['gates']]
            tensor = np.zeros((5, 5, 5), dtype=np.float64)
            diagonal_error = 0.0
            for layer in range(3):
                if layer:
                    a = gates[layer][:, None] * raw[layer].astype(np.float64).T
                    tensor = np.einsum('ai,bj,ck,ijk->abc', a, a, a, tensor, optimize=True)
                b, d, gamma = births[layer]
                born = dense_birth(np, b, d, gamma)
                actual_diag = np.array([born[i, i, i] for i in range(5)])
                expected_diag = 3 * d * np.diag(b) ** 2 + gamma
                diagonal_error = max(diagonal_error, float(np.max(np.abs(actual_diag - expected_diag))))
                tensor += born
            wlast = raw[-1].astype(np.float64).T
            final_tensor = np.einsum('ai,bj,ck,ijk->abc', wlast, wlast, wlast, tensor, optimize=True)
            reference = np.array([final_tensor[i, i, i] for i in range(5)])
            j = wlast
            symbolic = np.zeros(5)
            for layer in range(2, -1, -1):
                b, d, gamma = births[layer]
                u = j @ b
                symbolic += np.sum(3 * j * u * u * d[None, :] + j**3 * gamma[None, :], axis=1)
                if layer:
                    j = (j * gates[layer][None, :]) @ raw[layer].astype(np.float64).T
            native = np.asarray(predicted_k3)
            y0, y1 = np.asarray(y0), np.asarray(y1)
            error = native - reference
            result['checks'] = dict(
                column_birth_diagonal_max_error=diagonal_error,
                dense_vs_symbolic_max_error=float(np.max(np.abs(reference - symbolic))),
                fnp_vs_dense_max_error=float(np.max(np.abs(error))),
                fnp_vs_dense_relative=float(np.linalg.norm(error) / max(np.linalg.norm(reference), 1e-30)),
                reference_l2=float(np.linalg.norm(reference)),
                reference_maxabs=float(np.max(np.abs(reference))),
                early_rows_byte_exact=np.asarray(y0)[:-1].tobytes() == np.asarray(y1)[:-1].tobytes(),
                outputs_finite_nonnegative=bool(np.isfinite(y0).all() and np.isfinite(y1).all()
                                               and (y0 >= 0).all() and (y1 >= 0).all()),
                gaussian_products=baseline.last_diagnostics['fp32_matrix_products'],
                skew_products=candidate.last_diagnostics['fp32_matrix_products'])
            c = result['checks']
            h.require(c['column_birth_diagonal_max_error'] <= 1e-11 and c['dense_vs_symbolic_max_error'] <= 1e-11,
                      'Independent dense algebra')
            h.require(c['fnp_vs_dense_relative'] <= 2e-5 and c['fnp_vs_dense_max_error'] <= 2e-5,
                      'Fixed FP32 circuit accuracy')
            h.require(c['early_rows_byte_exact'] and c['outputs_finite_nonnegative'], 'Output contract/control')
            h.require(c['gaussian_products'] == 6 and c['skew_products'] == 11, 'Literal matrix-count gate')
        result['wall_s'] = time.monotonic() - start
        h.guard(result['bill'], wall=60, memory=1024**3)
        h.require(result['wall_s'] < 60 and h.rss() < 1024**3, 'Final resources')
        h.verify(bindings)
        result.update(passed=True, completed=True, sources_unchanged=True)
    except BaseException:
        result['error'] = traceback.format_exc()
    signal.setitimer(signal.ITIMER_REAL, 0)
    h.finish(out, result)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
