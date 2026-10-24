"""Small one-high/unrestricted/unrestricted tensor-sector certificate."""
import json
from pathlib import Path
import numpy as np
from one_high_third_reference_v1 import compare,matrix_coefficients

HERE=Path(__file__).resolve().parent;OUT=HERE/'one_high_third_small_v1.json';assert not OUT.exists()
rows=[]
for n,q in ((8,2),(16,4)):
    rng=np.random.default_rng(980227+n);p=rng.normal(size=(n,n+5));lam=rng.normal(size=n+5)
    star=rng.normal(size=(n,n));U=np.linalg.qr(rng.normal(size=(n,q)))[0];I=np.eye(n)
    W=rng.normal(size=(n,n))/np.sqrt(n)
    T=np.einsum('ir,jr,kr,r->ijk',p,p,p,lam)
    T+=(np.einsum('ij,jk->ijk',star,I)+np.einsum('ji,ik->ijk',star,I)+np.einsum('ki,ij->ijk',star,I))/3
    Q=I-U@U.T;low=np.einsum('abc,ai,bj,ck->ijk',T,Q,Q,Q)
    coef=matrix_coefficients(T,U)
    high=(np.einsum('ia,ajk->ijk',U,coef)+np.einsum('ja,aik->ijk',U,coef)+np.einsum('ka,aij->ijk',U,coef))/3
    plane=np.einsum('iij->ij',low);S=3*plane.T-2*np.diag(np.diag(plane))
    repaired=T-low+(np.einsum('ij,jk->ijk',S,I)+np.einsum('ji,ik->ijk',S,I)+np.einsum('ki,ij->ijk',S,I))/3
    got=compare(p,lam,star,U,W)
    rel=lambda a,b:float(np.linalg.norm(a-b)/np.linalg.norm(b))
    errors=dict(q_full_matrix_representation=rel(high,T-low),low_plane=rel(got['low_plane'],plane),
                full_read=rel(got['full'],np.einsum('ijk,ia,ja,ka->a',T,W,W,W)),
                restored_read=rel(got['restored'],np.einsum('ijk,ia,ja,ka->a',repaired,W,W,W)),
                repeated_preservation=rel(np.einsum('iij->ij',repaired),np.einsum('iij->ij',T)))
    rows.append(dict(n=n,q=q,errors=errors,passed=max(errors.values())<1e-10))
result=dict(rows=rows,passed=all(r['passed'] for r in rows),qualification='Small exact tensor-sector/coordinate-restoration identities, not recursive fidelity, cost or score.')
OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['passed']
