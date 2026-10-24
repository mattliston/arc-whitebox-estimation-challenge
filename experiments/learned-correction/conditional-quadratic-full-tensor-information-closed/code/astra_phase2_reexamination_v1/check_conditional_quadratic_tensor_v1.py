"""Dense synthetic tensor certificate, never wide arrays or labels."""
from pathlib import Path
import json
import numpy as np
from conditional_quadratic_tensor_v1 import tensor_contractions

here = Path(__file__).resolve().parent
out = here/'conditional_quadratic_tensor_certificate_v1.json'
assert not out.exists()
rows = []
for n, k, seed in ((8, 3, 95035), (16, 5, 95100)):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(n, n+3))
    lam = rng.normal(size=n+3)
    W = rng.normal(size=(n, n))/np.sqrt(n)
    star = rng.normal(size=(n, n))
    b = rng.normal(size=n)
    B = rng.normal(size=(n, n))
    B = (B+B.T)/2
    Q = rng.normal(size=(n, k))
    dense = np.einsum('ir,jr,kr,r->ijk', p, p, p, lam)
    ss = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for l in range(n):
                ss[i, j, l] = (star[i, j]*(j == l)+star[j, i]*(i == l)+star[l, i]*(i == j))/3
    dense += np.einsum('abc,ai,bj,ck->ijk', ss, W, W, W, optimize=True)
    dense += b[:, None, None]*B[None, :, :]+b[None, :, None]*B[:, None, :]+b[None, None, :]*B[:, :, None]
    target = np.einsum('ijk,ja,kb->iab', dense, Q, Q, optimize=True)
    result = tensor_contractions(Q, p, lam, W, star, b, B)
    rel = float(np.linalg.norm(target-result)/np.linalg.norm(target))
    # Orthogonal mode rotation must preserve the tensor Frobenius quantities.
    rotation = np.linalg.qr(rng.normal(size=(k, k)))[0]
    rotated = tensor_contractions(Q@rotation, p, lam, W, star, b, B)
    equiv = np.einsum('pa,ipq,qb->iab', rotation, result, rotation, optimize=True)
    equiv_rel = float(np.linalg.norm(rotated-equiv)/np.linalg.norm(equiv))
    row = dict(n=n, modes=k, relative_error=rel, rotation_equivariance_relative_error=equiv_rel, passed=max(rel, equiv_rel)<1e-12)
    rows.append(row)
    print(json.dumps(row), flush=True)
receipt = dict(rows=rows, passed=all(r['passed'] for r in rows), qualification='Small synthetic contraction certificate only.')
out.write_text(json.dumps(receipt, indent=2)+'\n')
assert receipt['passed'], receipt
