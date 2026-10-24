"""Research-only complete signed CP contractions in NumPy FP64."""
import numpy as np

def contract(p,lam,a,b):
    return p @ (lam[:,None]*(p.T@a)*(p.T@b))

def plane(p,lam):
    return (p*p) @ (p*lam[None,:]).T

def project(p,q):
    return q@(q.T@p)

def basis(p,lam,a,b):
    z=contract(p,lam,a,b)
    q,r=np.linalg.qr(z,mode='reduced')
    if not np.isfinite(q).all() or not np.isfinite(r).all():
        raise ValueError('nonfinite sketch QR')
    return q,r
