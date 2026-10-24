"""Independent tiny split/merge algebra certificate; no wide model or labels."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE/'positive_gmm_merge_algebra_certificate_v1.json'


def sym3(d, a):
    return (np.einsum('i,jk->ijk', d, a) + np.einsum('j,ik->ijk', d, a)
            + np.einsum('k,ij->ijk', d, a))


def cube(d):
    return np.einsum('i,j,k->ijk', d, d, d)


def relative(actual, reference):
    return float(np.linalg.norm(actual-reference)/max(1., float(np.linalg.norm(reference))))


def third(mean, covariance, center):
    d = mean-center
    return cube(d)+sym3(d, covariance)


def lost_third_squared(pa, pb, ma, mb, ca, cb):
    total = pa+pb
    alpha, beta = pa/total, pb/total
    d = ma-mb
    b = ca-cb+(beta-alpha)*np.outer(d, d)/3
    # Nonnegative sum of squares, no cancellation of a signed cubic cross term.
    return (total*alpha*beta)**2 * (
        3*np.dot(d, d)*np.sum(b*b) + 6*np.dot(b@d, b@d))


def main():
    if os.getpriority(os.PRIO_PROCESS, 0) < 10 or '/anaconda3/' in sys.executable:
        raise RuntimeError('Use nice10 NumPy-only repo interpreter')
    if OUT.exists():
        raise FileExistsError('Preserve existing certificate')
    source_path = Path(__file__)
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    started = time.time()
    rng = np.random.default_rng(930126)
    fixtures = []
    for index in range(8):
        n = 5
        f, g = rng.normal(size=(2, n, n))
        ca, cb = f@f.T+.5*np.eye(n), g@g.T+.5*np.eye(n)
        ma, mb, center, q = rng.normal(size=(4, n))
        pa, pb = [(1/6, 2/3), (.5, .5), (.001, .999), (.35, .2)][index % 4]
        if index == 4:
            mb = ma.copy()
        if index == 5:
            cb = ca.copy()
        if index == 6:
            cb, mb = ca.copy(), -ma
        total = pa+pb
        alpha, beta = pa/total, pb/total
        mean = alpha*ma+beta*mb
        d = ma-mb
        cov = alpha*ca+beta*cb+alpha*beta*np.outer(d, d)
        original_mean = pa*ma+pb*mb
        original_second = pa*(ca+np.outer(ma, ma))+pb*(cb+np.outer(mb, mb))
        actual_lost = pa*third(ma, ca, center)+pb*third(mb, cb, center)-total*third(mean, cov, center)
        expected_lost = total*alpha*beta*(sym3(d, ca-cb)+(beta-alpha)*cube(d))
        norm_actual = float(np.sum(actual_lost*actual_lost))
        norm_formula = float(lost_third_squared(pa, pb, ma, mb, ca, cb))
        norm_swapped = float(lost_third_squared(pb, pa, mb, ma, cb, ca))
        a = ca@q/np.sqrt(2*np.dot(q, ca@q))
        conditional_cov = ca-np.outer(a, a)
        nodes = np.array([-np.sqrt(3.), 0., np.sqrt(3.)])
        weights = np.array([1/6, 2/3, 1/6])
        projection = rng.normal(size=n)
        v = float(projection@ca@projection)
        vc = float(projection@conditional_cov@projection)
        projected_offsets = nodes*float(projection@a)
        moment_errors = []
        for order in range(6):
            moment = 0.
            for power in range(0, order+1, 2):
                df = math.prod(range(1, power, 2))
                moment += math.comb(order, power)*df*vc**(power//2)*float(
                    np.dot(weights, projected_offsets**(order-power)))
            truth = (math.prod(range(1, order, 2))*v**(order//2)) if order % 2 == 0 else 0.
            moment_errors.append(abs(moment-truth)/max(1., abs(truth)))
        fixtures.append(dict(index=index, dimension=n,
            mean_error=relative(total*mean, original_mean),
            second_error=relative(total*(cov+np.outer(mean, mean)), original_second),
            full_tensor_error=relative(actual_lost, expected_lost),
            norm_error=abs(norm_actual-norm_formula)/max(1., norm_actual),
            swap_error=abs(norm_formula-norm_swapped)/max(1., norm_formula),
            split_moment_errors_orders_0_to_5=moment_errors,
            split_psd_half_bound_min_eigenvalue=float(np.linalg.eigvalsh(conditional_cov-.5*ca).min()),
            split_covariance_min_eigenvalue=float(np.linalg.eigvalsh(conditional_cov).min()),
            merged_covariance_min_eigenvalue=float(np.linalg.eigvalsh(cov).min())))
    error_keys = ('mean_error', 'second_error', 'full_tensor_error', 'norm_error', 'swap_error')
    worst = max(max(*(row[k] for k in error_keys), *row['split_moment_errors_orders_0_to_5']) for row in fixtures)
    passed = worst <= 1e-11 and all(row['split_psd_half_bound_min_eigenvalue'] >= -1e-11
        and row['split_covariance_min_eigenvalue'] > 0 and row['merged_covariance_min_eigenvalue'] > 0 for row in fixtures)
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != source_sha:
        raise RuntimeError('Frozen source changed')
    result = dict(passed=bool(passed), seed=930126, fixtures=fixtures, worst_scaled_error=worst,
                  fixed_tolerance=1e-11, source_sha256=source_sha, target_access=False,
                  wide_execution=False, interpretation='Algebra only; no mixture accuracy or metered-cost claim',
                  wall_seconds=time.time()-started, utc_finished=datetime.now(timezone.utc).isoformat())
    with OUT.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(passed=bool(passed), worst_scaled_error=worst, output=OUT.name)), flush=True)
    if not passed:
        raise RuntimeError('Fixed tiny algebra failed; preserve receipt')


if __name__ == '__main__':
    main()
