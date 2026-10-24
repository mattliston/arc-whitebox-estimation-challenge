"""Tiny exact-response check against the unchanged full r=0..2 bank map."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from radial_mehler import make_nonlinear
from pair_memory_action_physics_v1 import offdiag, full_cubic_covariance_response

HERE = Path(__file__).resolve().parent
OUT = HERE/'pair_memory_response_certificate_v1.json'


def main():
    if OUT.exists():
        raise FileExistsError(f'Preserve existing certificate: {OUT}')
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError('Launch at nice >=10')
    started = time.time()
    rng = np.random.default_rng(930119)
    nonlinear = make_nonlinear(6, 2, True, True)
    rows = []
    for index in range(6):
        n = 7
        factor = rng.standard_normal((n, n))/np.sqrt(n)
        c0 = factor@factor.T + np.eye(n) + np.ones((n, n))*.6
        cov = c0/np.sqrt(np.outer(np.diag(c0), np.diag(c0)))
        scale = np.exp(rng.uniform(-1, 1, n))
        cov *= np.outer(scale, scale)
        mu = rng.standard_normal(n)*scale
        k = rng.standard_normal((n, n))*.003*scale[:, None]**2*scale[None, :]
        variance = np.diag(cov)
        k4 = .03*variance**2
        k22 = .03*(np.outer(variance, variance)+2*cov*cov)/3
        delta = offdiag(rng.standard_normal((n, n))*.003)
        base = nonlinear(mu, cov, k, k4, k22, 'covariance')
        aux = base[-1]
        response = full_cubic_covariance_response(delta, cov, aux['sigma'], aux['alpha'],
                                                 aux['pdf'], aux['gate'])
        physical = delta*aux['sigma'][:, None]**2*aux['sigma'][None, :]
        actual = nonlinear(mu, cov, k+physical, k4, k22, 'covariance')[1]-base[1]
        half = nonlinear(mu, cov, k+.5*physical, k4, k22, 'covariance')[1]-base[1]
        relative = float(np.linalg.norm(response-actual)/np.linalg.norm(actual))
        linearity = float(np.linalg.norm(2*half-actual)/np.linalg.norm(actual))
        row0 = .5*np.outer(aux['sigma'], aux['sigma'])*np.outer(aux['pdf'], aux['gate'])*delta
        truncated = row0+row0.T
        np.fill_diagonal(truncated, 0.)
        omitted = float(np.linalg.norm(truncated-actual)/np.linalg.norm(actual))
        rows.append(dict(fixture=index, full_response_relative_error=relative,
                         half_step_linearity_error=linearity,
                         r0_only_relative_error=omitted))
    worst = max(max(row['full_response_relative_error'], row['half_step_linearity_error']) for row in rows)
    result = dict(schema='pair-memory-response-certificate-v1', passed=worst <= 2e-11,
                  fixtures=rows, worst_relative_error=worst, target_access=False,
                  torch_imported=False, wide_execution=False,
                  source_hashes={name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                                 for name in (Path(__file__).name, 'pair_memory_action_physics_v1.py',
                                              'radial_mehler.py', 'k3_radial_numpy.py')},
                  wall_seconds=time.time()-started,
                  utc_finished=datetime.now(timezone.utc).isoformat(),
                  qualification='Exact frozen offdiagonal cubic covariance action only; '
                  'learned feedback, diagonal changes, higher-state recurrence and final raw are not differentiated.')
    with OUT.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result, indent=2), flush=True)
    if not result['passed']:
        raise RuntimeError('Fixed response identity failed')


if __name__ == '__main__':
    main()
