"""Inert OFFPY-only FNP scalar certificate; authenticated fixed reference only."""
import argparse
import gc
import json
import math
import signal
import time
import traceback

import common as h
from scalar_reference import GRID


def _sealed(directory):
    path, seal = directory / 'result.json', directory / 'completion.json'
    completion = json.loads(seal.read_text())
    h.require(completion['passed'] and completion['completed']
              and completion['result_sha256'] == h.sha(path), 'Completed passing seal')
    result = json.loads(path.read_text())
    h.require(result['passed'] and result['completed'] and result['sources_unchanged'],
              'Passing immutable prerequisite')
    return result, {str(path): h.sha(path), str(seal): h.sha(seal)}


def verify_reference(bindings):
    result, files = _sealed(h.HERE / 'reference')
    h.require(result['source_bindings'] == bindings, 'Exact reference source lineage')
    h.verify(result['source_bindings'])
    h.require(result['grid'] == list(GRID)
              and [r['alpha'] for r in result['rows']] == list(GRID),
              'Exactly the fixed independent scalar grid')
    h.require(result['target_access'] is False and result['estimator_imported'] is False,
              'Reference is scalar-only, target-free')
    return result, files


def verify_scalar(bindings):
    """Metadata-only prerequisite used by the v2 tiny/native adapters."""
    reference, reference_files = verify_reference(bindings)
    result, files = _sealed(h.HERE / 'scalar')
    h.require(result['source_bindings'] == bindings, 'Exact scalar source lineage')
    h.require(result['reference_bindings'] == reference_files, 'Reference seal lineage')
    h.require(result['candidate_sha256'] == h.sha(h.HERE / 'estimator.py'),
              'Exact candidate numerical helper binding')
    h.require(result['grid'] == reference['grid'], 'Scalar prerequisite fixed grid')
    h.verify(result['source_bindings'])
    h.verify(result['artifacts'])
    return {**reference_files, **files, **result['artifacts']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('INERT: pass --execute only after source review.')
        return 0
    runtime = h.runtime()
    bindings = h.sources()
    reference, reference_files = verify_reference(bindings)
    out = h.HERE / 'scalar'
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    result = dict(schema='gaussian_terminal_skew_v2_scalar_qualification', passed=False,
                  completed=False, source_bindings=bindings,
                  candidate_sha256=h.sha(h.HERE / 'estimator.py'),
                  reference_bindings=reference_files, runtime=runtime, grid=list(GRID),
                  rows=[], artifacts={}, target_access=False,
                  diagnostic_scope='Input + scalar helper + duplicate lower-tail check, not a card bill')
    h.write(out / 'started.json', dict(started_utc=h.utc(), **result))
    h.alarm(60)
    try:
        import numpy as np
        import flopscope as flops
        import flopscope.numpy as fnp
        from threadpoolctl import threadpool_limits, threadpool_info
        import stable_gaussian as stable
        with threadpool_limits(limits=1):
            pools = threadpool_info()
            h.require(all(p['num_threads'] == 1 for p in pools), 'All reported pools =1')
            result['threadpools'] = pools
            call_start = time.monotonic()
            with flops.BudgetContext(flop_budget=h.BUDGET, wall_time_limit_s=60, quiet=True) as ctx:
                alpha = fnp.array(GRID, dtype=fnp.float64)
                variance = fnp.ones(len(GRID), dtype=fnp.float64)
                values = stable.gaussian_raw(alpha, variance)
                lower_tail = stable.lower_tail(fnp.abs(alpha))
            bill = h.metrics(ctx, call_start)
            result['bill'] = bill
            bill_path = out / 'bill.json'
            h.write(bill_path, bill)
            result['artifacts'][str(bill_path)] = h.sha(bill_path)
            ref, reset = h.reset_closed(flops, ctx, bill_path)
            del ctx
            gc.collect()
            reset['context_released'] = ref() is None
            result['diagnostic_reset'] = reset
            h.require(reset['context_released'], 'Closed diagnostic context released')
            h.guard(bill, wall=60, memory=1024**3)
            arrays = {key: np.asarray(value) for key, value in values.items()}
            arrays['lower_tail'] = np.asarray(lower_tail)
            h.require(all(a.shape == (len(GRID),) and a.dtype == np.dtype('float64')
                          and np.isfinite(a).all() for a in arrays.values()),
                      'All scalar endpoints finite FP64 and fixed shape')
            for i, expected in enumerate(reference['rows']):
                actual = dict(lower_tail=float(arrays['lower_tail'][i]),
                              cdf=float(arrays['g'][i]), mean=float(arrays['m'][i]),
                              variance=float(arrays['post_v'][i]), k3=float(arrays['k3'][i]))
                relative = {key: abs(actual[key] - expected[key]) / abs(expected[key])
                            for key in ('lower_tail', 'mean', 'variance', 'k3')}
                cdf_error = abs(actual['cdf'] - expected['cdf'])
                row = dict(alpha=GRID[i], actual=actual,
                           reference={key: expected[key] for key in actual},
                           relative_errors=relative, cdf_absolute_error=cdf_error)
                result['rows'].append(row)
                row['passed'] = (all(math.isfinite(x) and x > 0 for x in
                                     (actual['lower_tail'], actual['mean'],
                                      actual['variance'], actual['k3']))
                                 and relative['lower_tail'] <= 1e-12
                                 and cdf_error <= 2e-14
                                 and all(relative[k] <= 1e-8 for k in ('mean', 'variance', 'k3')))
                h.require(row['passed'], 'Fixed scalar gate at alpha=' + str(GRID[i]))
        result['wall_s'] = time.monotonic() - start
        h.require(result['wall_s'] < 60 and h.rss() < 1024**3, 'Final scalar resource gate')
        h.verify(bindings)
        h.verify(reference_files)
        h.verify(result['artifacts'])
        result.update(passed=True, completed=True, sources_unchanged=True)
    except BaseException as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc(),
                      wall_s=time.monotonic() - start)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        h.finish(out, result)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
