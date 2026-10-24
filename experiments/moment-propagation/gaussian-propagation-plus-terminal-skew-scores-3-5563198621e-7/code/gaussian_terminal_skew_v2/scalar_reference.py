"""Inert fixed 13-point independent SciPy reference; no estimator or truth."""
import argparse
import math
import os
import signal
import sys
import time
import traceback

import common as h

GRID = (-16., -12., -9., -8., -7., -6., -5., -2., 0., 2., 5., 9., 16.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('INERT: pass --execute only after source review.')
        return 0
    env = {k: os.environ.get(k) for k in
           ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}
    h.require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'nice >=10 required')
    h.require(all(v == '1' for v in env.values()), 'All BLAS environment caps =1')
    bindings = h.sources()
    out = h.HERE / 'reference'
    out.mkdir(exist_ok=False)
    start = time.monotonic()
    result = dict(schema='gaussian_terminal_skew_v2_scalar_reference', passed=False,
                  completed=False, source_bindings=bindings, grid=list(GRID), rows=[],
                  executable=sys.executable, python=sys.version, env=env,
                  nice=os.getpriority(os.PRIO_PROCESS, 0), target_access=False,
                  estimator_imported=False, epsabs=1e-100, epsrel=2e-13, limit=200)
    h.write(out / 'started.json', dict(started_utc=h.utc(), **result))
    h.alarm(60)
    try:
        import scipy
        from scipy.integrate import quad
        result['scipy_version'] = scipy.__version__
        root_two_pi = math.sqrt(2 * math.pi)
        for alpha in GRID:
            a = abs(alpha)
            q = .5 * math.erfc(a / math.sqrt(2))
            moments, errors = [], []
            for degree in (1, 2, 3):
                value, error = quad(
                    lambda u: u**degree * math.exp(-.5 * (a + u)**2) / root_two_pi,
                    0., math.inf, epsabs=1e-100, epsrel=2e-13, limit=200)
                moments.append(float(value))
                errors.append(float(error))
            one, two, three = moments
            if alpha < 0:
                mean = one
                variance = two - one * one
                third = three - 3 * one * two + 2 * one**3
            else:
                loss2 = 2 * a * one + two
                mean = a + one
                variance = 1 - loss2 - one * one
                third = (3 * a * a * one + 3 * a * two + three
                         - 3 * one * (1 - loss2) + 2 * one**3)
            row = dict(alpha=alpha, lower_tail=q, cdf=q if alpha < 0 else 1 - q,
                       mean=mean, variance=variance, k3=third,
                       tail_raw_moments=moments, quadrature_error_estimates=errors)
            result['rows'].append(row)
            h.require(all(math.isfinite(x) and x > 0
                          for x in (q, mean, variance, third, *moments)),
                      'Positive finite independent scalar references')
            h.require(all(math.isfinite(e) and 0 <= e <= 1e-10 * v
                          for e, v in zip(errors, moments)),
                      'Fixed quadrature estimated relative error <=1e-10')
        result['wall_s'] = time.monotonic() - start
        h.require(result['wall_s'] < 60 and h.rss() < 1024**3, 'Reference resource gate')
        h.verify(bindings)
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
