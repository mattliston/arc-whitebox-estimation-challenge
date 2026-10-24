"""Independent explicit-matrix certificate for the ten fixed features."""
from pathlib import Path
import hashlib
import json
import numpy as np
from feature import twohop

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(93091917)
rows = []
for n, outputs in [(8,7),(19,11)]:
    x = rng.normal(size=(n,n))
    c = x @ x.T + np.eye(n)
    c += 1e-7 * rng.normal(size=c.shape)  # Declared feature symmetrization.
    s = rng.normal(size=(n,n))
    w = rng.normal(size=(n,outputs))
    sigma = rng.uniform(.5,2.,outputs)
    alpha = rng.uniform(-2,2,outputs)
    pdf = np.exp(-alpha**2/2)/np.sqrt(2*np.pi)
    actual = twohop(np,c,s,w,sigma,alpha,pdf)
    scale = np.sqrt(np.maximum(np.diag(c),1e-10))
    r = c / np.outer(scale,scale)
    r = (r+r.T)/2
    np.fill_diagonal(r,0)
    t = s / np.outer(scale**2,scale)
    np.fill_diagonal(t,0)
    # Explicit products in FP64 are independent of the factored contractions.
    operators = [r@t, r@t.T, t.T@t, t@t.T, t@t]
    u = scale[:,None]*w
    q = np.stack([np.diag(u.T @ op @ u) for op in operators],axis=1)/sigma[:,None]**2
    expected = np.concatenate([q*pdf[:,None],q*(alpha*pdf)[:,None]],axis=1)
    relative = np.linalg.norm(actual-expected)/np.linalg.norm(expected)
    assert relative < 2e-6, relative
    zero = twohop(np,c,np.diag(np.diag(s)),w,sigma,alpha,pdf)
    assert np.array_equal(zero,np.zeros((outputs,10),np.float32))
    assert actual.dtype == np.float32 and np.isfinite(actual).all()
    rows.append({'width':n,'outputs':outputs,'relative_error':float(relative),'diagonal_third_zero_exact':True})
result={'passed':True,'cases':rows,'feature_sha256':hashlib.sha256((HERE/'feature.py').read_bytes()).hexdigest()}
with (HERE/'certificate.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
