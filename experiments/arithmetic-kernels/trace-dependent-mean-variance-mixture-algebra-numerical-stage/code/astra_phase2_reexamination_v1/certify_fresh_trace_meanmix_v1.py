"""Tiny exact polynomial/Wick certificate; no network data or targets."""
from pathlib import Path
from itertools import product
import hashlib
import json
import resource
import time
import numpy as np
from fresh_trace_meanmix_v1 import central_raw, k3_tensor, k4_entry, planes, directional_cumulants

HERE = Path(__file__).resolve().parent
OUT = HERE / 'fresh_trace_meanmix_algebra_v1.json'
assert not OUT.exists()
started = time.time()
rng = np.random.default_rng(930098)
records = []
for q in (.01, .03, .10):
    n = 4
    A = rng.normal(size=(n, n)) / np.sqrt(n)
    B = A @ A.T + np.eye(n) * .5
    D = np.diag(rng.uniform(.05, .2, n))
    C = B + D
    direction = rng.normal(size=n)
    b = direction * np.sqrt(.2 * q / (direction @ np.linalg.solve(B, direction)))
    V = B - np.outer(b, b) / q
    third = k3_tensor(b, B, q)
    t21, q31, q22, q4 = planes(b, B, q)
    mean_error = max(abs(central_raw((i,), b, B, C, q)) for i in range(n))
    cov_error = max(abs(central_raw((i, j), b, B, C, q)-C[i, j]) for i, j in product(range(n), repeat=2))
    third_error = max(abs(central_raw(ix, b, B, C, q)-third[ix]) for ix in product(range(n), repeat=3))
    fourth_error = 0.0
    plane_error = 0.0
    for ix in product(range(n), repeat=4):
        i, j, k, l = ix
        actual = central_raw(ix, b, B, C, q) - C[i, j]*C[k, l]-C[i, k]*C[j, l]-C[i, l]*C[j, k]
        fourth_error = max(fourth_error, abs(actual-k4_entry(ix, b, B, q)))
    for i, j in product(range(n), repeat=2):
        plane_error = max(plane_error, abs(t21[i, j]-third[i, i, j]),
                          abs(q31[i, j]-k4_entry((i, i, i, j), b, B, q)),
                          abs(q22[i, j]-k4_entry((i, i, j, j), b, B, q)),
                          abs(q4[i]-k4_entry((i, i, i, i), b, B, q)))
    directional_errors = []
    for _ in range(12):
        u = rng.normal(size=n)
        u /= np.linalg.norm(u)
        bs = np.array([u @ b]); Bs = np.array([[u @ B @ u]]); Cs = np.array([[u @ C @ u]])
        moments = {k: central_raw((0,) * k, bs, Bs, Cs, q) for k in range(2, 7)}
        actual = {3: moments[3], 4: moments[4]-3*moments[2]**2,
                  5: moments[5]-10*moments[3]*moments[2],
                  6: moments[6]-15*moments[4]*moments[2]-10*moments[3]**2+30*moments[2]**3}
        expected = directional_cumulants(u, b, B, q)
        directional_errors.append(max(abs(actual[k]-expected[k])/(1e-13+abs(expected[k])) for k in actual))
    rec = dict(q=q, min_V_eigenvalue=float(np.linalg.eigvalsh(V).min()),
               mean_error=mean_error, covariance_error=cov_error,
               third_error=third_error, fourth_error=fourth_error,
               plane_error=plane_error, max_directional_relative_error=max(directional_errors))
    rec['passed'] = (max(mean_error, cov_error, third_error, fourth_error, plane_error) < 1e-11
                     and rec['max_directional_relative_error'] < 1e-11)
    records.append(rec)
result = dict(records=records, passed=all(r['passed'] for r in records),
              wall_s=time.time()-started,
              peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
              source_sha256=hashlib.sha256((HERE/'fresh_trace_meanmix_v1.py').read_bytes()).hexdigest(),
              qualification='Exact finite Gamma-polynomial/Wick expectations, n4, twelve fixed directional checks per q. No Monte Carlo or targets. This certifies probability-model identities, not their validity for the bank distribution or a useful correction.')
OUT.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2), flush=True)
