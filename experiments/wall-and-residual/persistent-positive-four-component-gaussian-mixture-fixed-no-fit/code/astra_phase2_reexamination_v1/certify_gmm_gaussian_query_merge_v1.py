"""Tiny independent Wick/query metric certificate; no model or truth arrays."""
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
from certify_positive_gmm_merge_algebra_v1 import sym3, cube

HERE = Path(__file__).resolve().parent
PARENT = HERE/'certify_positive_gmm_merge_algebra_v1.py'
PARENT_SHA = '7eae99bcc973d78b0def3ccd8bb72ea75e897c8ea5215355489ed1efe9050735'
OUT = HERE/'gmm_gaussian_query_merge_certificate_v1.json'


def main():
    if os.getpriority(os.PRIO_PROCESS, 0) < 10 or OUT.exists():
        raise RuntimeError('Use nice10; preserve any existing certificate')
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Frozen algebra parent changed')
    started = time.time()
    rng = np.random.default_rng(930138)
    n = 5
    nodes, weights = np.polynomial.hermite_e.hermegauss(4)
    weights /= np.sqrt(2*np.pi)
    indices = np.array(list(itertools.product(range(4), repeat=n)))
    points, joint_weights = nodes[indices], np.prod(weights[indices], axis=1)
    fixtures = []
    for index in range(7):
        d = rng.normal(size=n)
        draw = rng.normal(size=(n, n))
        b = (draw+draw.T)/2
        multiplier = [.03, .2, .5, 1.][index % 4]
        if index == 0:
            b = np.outer(d, d)/3  # Sym3(d,B)=d^3, sixth Gaussian moment is15||d||^6.
        if index == 1:
            d = np.zeros_like(d)
        tensor = multiplier*sym3(d, b)
        trace = np.einsum('iik->k', tensor)
        dense_formula = 6*np.sum(tensor*tensor)+9*np.dot(trace, trace)
        bd = b@d
        trace_formula = multiplier*(d*np.trace(b)+2*bd)
        frobenius_formula = multiplier**2*(3*np.dot(d, d)*np.sum(b*b)+6*np.dot(bd, bd))
        structured_formula = 6*frobenius_formula+9*np.dot(trace_formula, trace_formula)
        contracted = np.einsum('ijk,si,sj,sk->s', tensor, points, points, points)
        quadrature = float(np.dot(joint_weights, contracted*contracted))
        denominator = max(1., abs(quadrature))
        row = dict(index=index, dimension=n,
                   dense_wick_error=abs(dense_formula-quadrature)/denominator,
                   structured_metric_error=abs(structured_formula-quadrature)/denominator,
                   trace_identity_error=float(np.linalg.norm(trace-trace_formula)/max(1., np.linalg.norm(trace))),
                   query_energy=quadrature)
        if index == 0:
            expected = multiplier**2*15*np.dot(d, d)**3
            row['rank_one_sixth_moment_error'] = abs(expected-quadrature)/denominator
        fixtures.append(row)
    worst = max(max(row['dense_wick_error'], row['structured_metric_error'], row['trace_identity_error'],
                    row.get('rank_one_sixth_moment_error', 0.)) for row in fixtures)
    passed = worst <= 1e-11
    if (hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != source_sha
            or hashlib.sha256(PARENT.read_bytes()).hexdigest() != PARENT_SHA):
        raise RuntimeError('Frozen source changed')
    result = dict(passed=bool(passed), fixtures=fixtures, fixed_seed=930138,
                  quadrature_nodes_per_dimension=4, worst_scaled_error=worst, tolerance=1e-11,
                  source_sha256=source_sha, parent_sha256=PARENT_SHA,
                  target_access=False, wide_execution=False,
                  interpretation='Exact isotropic-Gaussian cubic-query MSE; not exact final-mean risk or known-W criterion',
                  wall_seconds=time.time()-started, utc_finished=datetime.now(timezone.utc).isoformat())
    with OUT.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(passed=bool(passed), worst_scaled_error=worst, output=OUT.name)), flush=True)
    if not passed:
        raise RuntimeError('Fixed query metric certificate failed; preserve receipt')


if __name__ == '__main__':
    main()
