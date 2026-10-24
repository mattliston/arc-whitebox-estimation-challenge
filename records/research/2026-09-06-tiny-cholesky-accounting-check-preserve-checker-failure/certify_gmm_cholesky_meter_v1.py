"""Tiny official-version Cholesky operation/bill check, not an estimator port."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

HERE = Path(__file__).resolve().parent
OUT = HERE/'gmm_cholesky_meter_certificate_v1.json'


def main():
    if os.getpriority(os.PRIO_PROCESS, 0) < 10 or OUT.exists():
        raise RuntimeError('Use nice10 and preserve any existing receipt')
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    started = time.time()
    rows = []
    for n in (16, 32):
        original = np.eye(n, dtype=np.float64)+.01*np.ones((n, n), dtype=np.float64)
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ingress:
            a = fnp.asarray(original, dtype=fnp.float64)
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as operation:
            l = fnp.linalg.cholesky(a)
        actual = np.asarray(l)
        relative = float(np.linalg.norm(actual@actual.T-original)/np.linalg.norm(original))
        theoretical = 2*n**3/3
        rows.append(dict(dimension=n, dtype=str(actual.dtype), relative_reconstruction_error=relative,
                         ingress_flops=int(ingress.flops_used), cholesky_flops=int(operation.flops_used),
                         theoretical_float64_cholesky_flops=theoretical,
                         bill_matches_integer_rounding=abs(operation.flops_used-theoretical) <= 1))
    passed = all(row['relative_reconstruction_error'] <= 1e-12
                 and row['dtype'] == 'float64' and row['bill_matches_integer_rounding'] for row in rows)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != digest:
        raise RuntimeError('Frozen source changed')
    result = dict(passed=bool(passed), rows=rows, source_sha256=digest,
                  flopscope_version=importlib.metadata.version('flopscope'),
                  target_access=False, wide_execution=False, estimator_executed=False,
                  interpretation='Kernel-only installed official-version accounting; no whole-estimator bill or runtime guarantee',
                  wall_seconds=time.time()-started, utc_finished=datetime.now(timezone.utc).isoformat())
    with OUT.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result), flush=True)
    if not passed:
        raise RuntimeError('Tiny Cholesky certificate failed; preserve receipt')


if __name__ == '__main__':
    main()
