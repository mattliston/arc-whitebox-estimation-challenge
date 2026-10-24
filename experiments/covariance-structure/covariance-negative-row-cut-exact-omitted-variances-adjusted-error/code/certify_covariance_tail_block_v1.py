"""Small exact-arithmetic-model check for localized covariance approximation."""
from pathlib import Path
import hashlib
import json
import numpy as np
HERE=Path(__file__).resolve().parent
rng=np.random.default_rng(2026092701)
checks=[]
for n,k,m in [(9,5,7),(12,8,10),(5,1,6),(8,8,4)]:
    x=rng.normal(size=(n,n));C=x@x.T
    W=rng.normal(size=(n,m))
    keep=np.sort(rng.permutation(n)[:k]);tail=np.setdiff1d(np.arange(n),keep)
    approx=np.diag(np.diag(C));approx[np.ix_(keep,keep)]=C[np.ix_(keep,keep)]
    wk=W[keep];wd=W[tail];ck=C[np.ix_(keep,keep)];dt=np.diag(C)[tail]
    pushed=wk.T@(ck@wk)+wd.T@(dt[:,None]*wd)
    direct=((wk*(ck@wk)).sum(0)+(wd*wd*dt[:,None]).sum(0))
    ref=W.T@approx@W
    checks.append(dict(n=n,kept=k,outputs=m,push_max_abs=float(np.max(np.abs(pushed-ref))),
                       diagonal_max_abs=float(np.max(np.abs(direct-np.diag(ref))))))
assert max(max(c['push_max_abs'],c['diagonal_max_abs']) for c in checks)<1e-10
protocol=HERE/'covariance_tail_block_protocol_v1.json'
record=dict(passed=True,checks=checks,protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
            scope='Identity for the stated approximate covariance only; no accuracy or native-bill claim.')
with (HERE/'covariance_tail_block_certificate_v1.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
