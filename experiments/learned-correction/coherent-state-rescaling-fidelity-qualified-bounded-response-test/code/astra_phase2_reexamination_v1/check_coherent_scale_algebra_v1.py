"""Dense centered-tensor scaling certificate, including stored-float rounding."""
import json
from pathlib import Path
import numpy as np
from coherent_central_scale_v1 import transform

HERE = Path(__file__).resolve().parent
OUT = HERE / 'coherent_central_scale_algebra_v1.json'
assert not OUT.exists()


def tensors(C, p, lam, star, b, alive, radial):
    n = len(b)
    I = np.eye(n)
    metric = C.copy()
    np.fill_diagonal(metric, np.diag(C) * alive)
    T = np.einsum('ir,jr,kr,r->ijk', p, p, p, lam)
    T += (np.einsum('ij,jk->ijk', star, I)
          + np.einsum('ji,ik->ijk', star, I)
          + np.einsum('ki,ij->ijk', star, I))/3
    T += (np.einsum('i,jk->ijk', b, metric)
          + np.einsum('j,ik->ijk', b, metric)
          + np.einsum('k,ij->ijk', b, metric))
    Q = radial*(np.einsum('ij,kl->ijkl', C, C)
                + np.einsum('ik,jl->ijkl', C, C)
                + np.einsum('il,jk->ijkl', C, C))/3
    return T, Q


rows = []
for n in (8, 16):
    rng = np.random.default_rng(965400+n)
    X = rng.normal(size=(n,n))
    C = X@X.T/n + np.eye(n)
    p, lam = rng.normal(size=(n,n+3)), rng.normal(size=n+3)
    star, b = rng.normal(size=(n,n)), rng.normal(size=n)
    alive, radial = rng.uniform(size=n) > .3, .1
    scale = np.exp(rng.normal(size=n)*.05)
    for dtype in (np.float64, np.float32):
        pp, aa = p.astype(dtype), star.astype(dtype)
        T, Q = tensors(C, pp.astype(float), lam, aa.astype(float), b, alive, radial)
        C1, p1, a1, b1 = transform(C, pp, aa, b, scale)
        T1, Q1 = tensors(C1, p1.astype(float), lam, a1.astype(float), b1, alive, radial)
        expectedT = T*scale[:,None,None]*scale[None,:,None]*scale[None,None,:]
        expectedQ = Q*scale[:,None,None,None]*scale[None,:,None,None]*scale[None,None,:,None]*scale[None,None,None,:]
        rel = lambda x,y:float(np.linalg.norm(x-y)/np.linalg.norm(y))
        rt, rq = rel(T1, expectedT), rel(Q1, expectedQ)
        zero = transform(C, pp, aa, b, np.ones(n))
        assert all(np.array_equal(x,y) for x,y in zip(zero,(C,pp,aa,b)))
        passed = rt < (1e-10 if dtype == np.float64 else 1e-6) and rq < 1e-10
        rows.append(dict(n=n, storage=str(np.dtype(dtype)), third_relative=rt,
                         fourth_relative=rq, zero_action_exact=True, passed=passed))
result = dict(rows=rows, passed=all(r['passed'] for r in rows),
              qualification='Synthetic algebra and float32 storage-rounding certificate only; no targets, wide-network fidelity, learned correction or score.')
OUT.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
assert result['passed']
