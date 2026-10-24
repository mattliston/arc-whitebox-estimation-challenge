"""Independent SPD/indefinite core inversion checks, no data or targets."""
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from fourth_trace_distinct_necessity_v2 import inverse_core, core_trace

HERE = Path(__file__).resolve().parent
OUT = HERE/'fourth_trace_distinct_necessity_small_v2.json'
assert not OUT.exists()
prior = json.loads((HERE/'fourth_trace_distinct_necessity_small_v1.json').read_text())
assert prior['passed']
rng = np.random.default_rng(930075)
records = []
with threadpool_limits(limits=2):
    for n in (9, 17):
        for indefinite in (False, True):
            Q = np.linalg.qr(rng.normal(size=(n, n)))[0]
            values = rng.uniform(.5, 2., n)
            if indefinite:
                values[:2] = (-.01, -.001)
            C = (Q*values)@Q.T
            A = rng.normal(size=(n, n))*.01
            A = (A+A.T)/2
            recovered = inverse_core(C, core_trace(A, C))
            error = float(np.max(np.abs(recovered-A)))
            records.append(dict(n=n, indefinite=indefinite, minimum_eigenvalue=float(values.min()),
                                minimum_sylvester_denominator=float(np.trace(C)+4*values.min()),
                                core_inverse_max=error, passed=error <= 1e-10))
result = dict(passed=all(r['passed'] for r in records), records=records,
              inherited_source_class_certificate='fourth_trace_distinct_necessity_small_v1.json',
              target_access=False,
              source_sha256=hashlib.sha256((HERE/'fourth_trace_distinct_necessity_v2.py').read_bytes()).hexdigest())
OUT.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2), flush=True)
assert result['passed']
