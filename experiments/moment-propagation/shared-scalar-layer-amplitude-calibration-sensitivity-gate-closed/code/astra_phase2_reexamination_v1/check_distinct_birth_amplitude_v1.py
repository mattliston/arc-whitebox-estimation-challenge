"""Synthetic exact repeated-plane and distinct-only action certificate."""
from pathlib import Path
import json
import numpy as np

here = Path(__file__).resolve().parent
out = here/'distinct_birth_amplitude_certificate_v1.json'
assert not out.exists()
rng = np.random.default_rng(961062)
n = 9
p = rng.normal(size=(n, n))
np.fill_diagonal(p, 0)
t, s = rng.normal(size=n), rng.normal(size=n)
delta = t-s
assert np.min(np.abs(delta)) > 1e-6
weight = rng.uniform(.1, 1, n)
p1, p2 = p.copy(), p.copy()
np.fill_diagonal(p1, t)
np.fill_diagonal(p2, s)
target = rng.normal(size=(n, n))


def tensor(amplitude):
    cp = np.einsum('ia,ja,ka,a->ijk', p1, p1, p1, amplitude*weight)
    cp -= np.einsum('ia,ja,ka,a->ijk', p2, p2, p2, amplitude*weight)
    plane = np.einsum('iij->ij', cp)
    rest = target-plane
    A = 3*rest.T-2*np.diag(np.diag(rest))
    star = np.zeros_like(cp)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                star[i, j, k] = (A[i, j]*(j == k)+A[j, i]*(i == k)+A[k, i]*(i == j))/3
    return cp+star, cp


base, cp0 = tensor(1.)
rows = []
index = np.indices((n, n, n))
distinct = (index[0] != index[1]) & (index[0] != index[2]) & (index[1] != index[2])
for amplitude in (.9, 1., 1.1):
    full, cp = tensor(amplitude)
    plane_error = float(np.max(np.abs(np.einsum('iij->ij', full)-target)))
    delta = full-base
    distinct_error = float(np.max(np.abs(delta-(cp-cp0)*distinct)))
    rows.append(dict(amplitude=amplitude, plane_max_abs_error=plane_error,
                     distinct_action_max_abs_error=distinct_error,
                     passed=max(plane_error, distinct_error)<1e-12))
receipt = dict(rows=rows, passed=all(r['passed'] for r in rows), qualification='Small exact-action certificate only; no MLP accuracy claim.')
out.write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))
assert receipt['passed'], receipt
