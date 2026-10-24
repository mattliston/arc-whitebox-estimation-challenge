"""Small independent inverse and source-class subtraction checks."""
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from fourth_trace_distinct_necessity_v1 import ZeroDistinctThird, core_trace, inverse_core
from fourth_trace_full_edge_reference_v1 import FullEdgeThird, full_edge_query
from fourth_trace_full_queries_v1 import complete_trace_query
from fourth_repeated_compiled_v1 import repeated
from thin_fourth_wide_helpers_v1 import PAIR

HERE = Path(__file__).resolve().parent
OUT = HERE/'fourth_trace_distinct_necessity_small_v1.json'
assert not OUT.exists()
assert json.loads((HERE/'fourth_full_edge_reference_certificate_v2.json').read_text())['passed']
rng = np.random.default_rng(930074)
records = []
with threadpool_limits(limits=2):
    for n in (9, 17):
        v = rng.normal(size=(n, n))/np.sqrt(n)
        C = v@v.T+np.eye(n)
        A = rng.normal(size=(n, n))*.01
        A = (A+A.T)/2
        recovered = inverse_core(C, core_trace(A, C))
        mu = rng.normal(size=n)
        p = rng.normal(size=(n, n+3))/np.sqrt(n)
        lam = rng.normal(size=n+3)*.005
        b = rng.normal(size=n)*.001
        star = rng.normal(size=(n, n))*.001
        ws = np.eye(n)
        K = (p*p)@(p*lam).T+(star.T+2*np.diag(np.diag(star)))/3
        K += np.diag(C)[:, None]*b[None, :]+2*C*b[:, None]
        radial = .02
        diagonal = np.diag(C)
        K22 = radial*(np.outer(diagonal, diagonal)+2*C*C)/3
        K31 = radial*diagonal[:, None]*C
        _, post_C, _, _, aux = PAIR(mu, C, K, radial*diagonal*diagonal, K22, 'covariance')
        h, wick = aux['h'], aux['wick']
        correction = repeated(C, K, K22, K31, radial*C, C, h, wick)
        third = FullEdgeThird(p, lam, star, ws, b, C, K, C)
        X = np.eye(n)
        kwargs = (C, K, K22, radial*C, C, h, wick, np.zeros((n, 0)))
        full = complete_trace_query(full_edge_query(*kwargs, third, correction, X), post_C, aux['post31'], aux['post22'], X)
        cheap = complete_trace_query(full_edge_query(*kwargs, ZeroDistinctThird(), correction, X), post_C, aux['post31'], aux['post22'], X)
        omitted = third.fourth_third_part(h[1], h[2], wick[1], wick[2], X)
        errors = dict(core_inverse=float(np.max(np.abs(recovered-A))),
                      source_class_difference=float(np.max(np.abs(full-cheap-omitted))))
        records.append(dict(n=n, errors=errors, passed=max(errors.values()) <= 1e-10))
result = dict(passed=all(r['passed'] for r in records), records=records, target_access=False,
              source_sha256=hashlib.sha256((HERE/'fourth_trace_distinct_necessity_v1.py').read_bytes()).hexdigest())
OUT.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2), flush=True)
assert result['passed']
