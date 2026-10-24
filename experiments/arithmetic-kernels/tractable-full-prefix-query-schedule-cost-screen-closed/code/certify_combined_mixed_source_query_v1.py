"""Small dense identity for the exact tractable source query compiler."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from combined_mixed_source_query_v1 import combined_factors, query
from exact_heavy_source_slices_v1 import dense as heavy_dense
from full_edge_gaussian_bond3_v1 import dense as gaussian_dense

HERE = Path(__file__).resolve().parent
OUT = HERE / 'combined_mixed_source_query_small_v1.json'
assert not OUT.exists()
rng = np.random.Generator(np.random.PCG64DXSM(930090))
records = []
for n, output in [(7, 5), (11, 8)]:
    M = rng.normal(size=(n, n)); C = M @ M.T / n + np.eye(n)
    g = rng.uniform(.1, .9, n); d = rng.uniform(.05, .4, n)
    eta = rng.normal(size=n); e = eta * d
    K21 = rng.normal(size=(n, n)) * .01
    K22 = rng.normal(size=(n, n)) * .01; K22 = (K22 + K22.T) / 2
    B, F = combined_factors(C, g, d, e, eta, K21, K22)
    base = np.einsum('ia,ja,ka->ijk', np.eye(n), B, F)
    T = sum(base.transpose(order) for order in itertools.permutations(range(3)))
    source_plane_error = float(np.max(np.abs(T[np.arange(n), np.arange(n), :] - 2 * B * F)))
    ii, jj, kk = np.indices(T.shape)
    T *= (ii != jj) & (ii != kk) & (jj != kk)
    off = C.copy(); np.fill_diagonal(off, 0)
    triangle = np.einsum('i,j,k,ij,jk,ki->ijk', d, d, d, off, off, off)
    existing = heavy_dense(C, g, d, eta, K21, K22) + gaussian_dense(C, g, d, e) - triangle
    source_error = float(np.max(np.abs(T - existing)))
    V = rng.normal(size=(n, output)) / np.sqrt(n)
    pushed = np.einsum('ia,jb,kc,ijk->abc', V, V, V, T, optimize=True)
    expected = pushed[np.arange(output), np.arange(output), :]
    query_error = float(np.max(np.abs(query(B, F, V) - expected)))
    passed = max(source_plane_error, source_error, query_error) <= 1e-12
    records.append(dict(width=n, output_width=output, passed=bool(passed), source_repeated_plane_max=source_plane_error,
                        existing_source_max=source_error, full_pushed_query_max=query_error))
result = dict(passed=all(r['passed'] for r in records), records=records, target_access=False,
              source_sha256=hashlib.sha256((HERE / 'combined_mixed_source_query_v1.py').read_bytes()).hexdigest())
OUT.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
assert result['passed']
