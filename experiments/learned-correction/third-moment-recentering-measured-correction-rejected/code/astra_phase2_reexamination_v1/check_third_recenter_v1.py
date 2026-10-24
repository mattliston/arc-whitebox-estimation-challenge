"""Tiny dense algebra certificate, not an accuracy or runtime claim."""
import json
from pathlib import Path
import numpy as np
from radial_third_recenter_v1 import mean_cubes, shift_plane

out = Path(__file__).resolve().parent/'third_recenter_certificate_v1.json'
assert not out.exists()
rng = np.random.default_rng(13983)
n = 7
m = rng.normal(size=n)
d = rng.normal(size=n)*1e-4
X = rng.normal(size=(n, n))
C = X@X.T
Cnew = C-np.outer(m, d)-np.outer(d, m)-np.outer(d, d)
def cube(a):
    return np.einsum('i,j,k->ijk', a,a,a)
def sym(a,B):
    return (np.einsum('i,jk->ijk',a,B)+np.einsum('j,ik->ijk',a,B)+np.einsum('k,ij->ijk',a,B))
def cumulant3(mean, R2, R3):
    return R3-sym(mean,R2)+2*cube(mean)
R2 = C+np.outer(m,m)
R3 = rng.normal(size=(n,n,n))
delta = cumulant3(m+d,R2,R3)-cumulant3(m,R2,R3)
analytic = -sym(d,Cnew)+sym(d,np.outer(m,m))-cube(d)
p,lam = mean_cubes(m,d)
cp = np.einsum('ir,jr,kr,r->ijk',p,p,p,lam)
represented = -sym(d,Cnew)+cp
ids = np.arange(n)
plane = represented[ids,ids,:]
w = rng.normal(size=(n,n))
pushed = np.einsum('ijk,ia,jb,kc->abc',represented,w,w,w)
pp = w.T@p
represented_push = -sym(w.T@d,w.T@Cnew@w)+np.einsum('ir,jr,kr,r->ijk',pp,pp,pp,lam)
pf = p.astype(np.float32)
cp32 = np.einsum('ir,jr,kr,r->ijk',pf,pf,pf,lam.astype(np.float32)).astype(np.float64)
result = dict(raw_identity_max_abs=float(np.max(np.abs(delta-analytic))),
    representation_max_abs=float(np.max(np.abs(analytic-represented))),
    plane_max_abs=float(np.max(np.abs(plane-shift_plane(m,Cnew,d)))),
    push_max_abs=float(np.max(np.abs(pushed-represented_push))),
    balanced_cp_float32_relative=float(np.linalg.norm(cp-cp32)/np.linalg.norm(cp)))
assert result['raw_identity_max_abs'] < 1e-12
assert result['representation_max_abs'] < 1e-12
assert result['plane_max_abs'] < 1e-12
assert result['push_max_abs'] < 1e-12
assert result['balanced_cp_float32_relative'] < 1e-5
out.write_text(json.dumps(dict(pass_all=True,**result),indent=2)+'\n')
print(json.dumps(result),flush=True)
