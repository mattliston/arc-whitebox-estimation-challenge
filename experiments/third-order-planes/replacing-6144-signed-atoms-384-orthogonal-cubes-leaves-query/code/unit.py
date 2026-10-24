import json
from pathlib import Path
import resource
import time
import numpy as np
from algebra import Products, merge, contract, plane, norm_squared

HERE=Path(__file__).resolve().parent
started=time.monotonic()
rng=np.random.Generator(np.random.PCG64(2026092039))
rows=[]
scaled=lambda x,y:float(np.linalg.norm(x-y)/max(1.,np.linalg.norm(y)))


def dense(p,lam):
    return np.einsum('ir,jr,kr,r->ijk',p,p,p,lam)


for n in (3,5,7):
    for kind in ('orthogonal','general','near_cancel','zero'):
        q0,_=np.linalg.qr(rng.normal(size=(n+2,n)))
        change=np.diag(np.linspace(.4,1.7,n))
        q=q0@change
        if kind=='orthogonal':
            v,_=np.linalg.qr(rng.normal(size=(n,n)))
            f=np.linalg.solve(change,v)
            lam=np.linspace(.3,1.3,n)*np.where(np.arange(n)%2,1.,-1.)
        else:
            f=rng.normal(size=(n,2*n))
            lam=rng.normal(size=2*n)
            if kind=='near_cancel':
                f[:,n:]=f[:,:n]+1e-5*rng.normal(size=(n,n))
                lam[n:]=-lam[:n]
            if kind=='zero':
                # Exact duplicated columns with exactly opposite weights.
                q=np.eye(n+2,n)
                f=np.concatenate([np.eye(n),np.eye(n)],axis=1)
                lam=np.r_[np.ones(n),-np.ones(n)]
        direction=rng.normal(size=n+2)
        products=Products()
        basis,c,a,u,eigen,checks=merge(q,f,lam,direction,products)
        p=q@f
        T=dense(p,lam)
        H=dense(basis,c)
        projected=u.T@direction
        projected=projected/np.linalg.norm(projected)
        dense_slice=np.einsum('ijk,k->ij',T,u@projected)
        reconstructed_slice=(basis*eigen[None,:])@basis.T
        direct_c=np.einsum('ijk,ir,jr,kr->r',T,basis,basis,basis)
        energy,absolute=norm_squared(p,lam,products,block=3)
        residual=float(np.sum((T-H)**2))
        identity=energy-float(c@c)
        x=rng.normal(size=(n+2,11));y=rng.normal(size=(n+2,11))
        actual=contract(p,lam,x,y,products,'unit')
        wanted=np.einsum('ijk,ja,ka->ia',T,x,y)
        pp=plane(p,lam,products,'unit_plane')
        expected=np.einsum('iij->ij',T)
        errors=dict(slice=scaled(reconstructed_slice,dense_slice),
                    coefficient=scaled(c,direct_c),query=scaled(actual,wanted),
                    plane=scaled(pp,expected),norm=abs(energy-float(np.sum(T*T)))/max(1.,float(np.sum(T*T))),
                    projection_identity=abs(identity-residual)/max(1.,float(np.sum(T*T))))
        if kind=='orthogonal': errors['orthogonal_recovery']=scaled(H,T)
        if kind=='zero':
            assert np.linalg.norm(H)<=1e-12 and np.linalg.norm(T)<=1e-12
        assert all(value<=1e-11 for value in errors.values()),(n,kind,errors)
        rows.append(dict(n=n,kind=kind,errors=errors,checks=checks))
out=dict(passed=True,rows=rows,elapsed_s=time.monotonic()-started,
         peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (HERE/'unit.json').open('x') as stream:json.dump(out,stream,indent=2)
print(json.dumps(out),flush=True)
