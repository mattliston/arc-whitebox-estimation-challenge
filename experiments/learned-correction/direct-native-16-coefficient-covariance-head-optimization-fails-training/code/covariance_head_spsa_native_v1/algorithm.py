import numpy as np
from common import TRAIN

def gains(t):return .1*(4.6/(t+4.6))**.602,(1/16)/(t+1)**.101

def schedule():
    rng=np.random.Generator(np.random.PCG64(2026092066));rows=[]
    for cycle in range(6):
        ranks=rng.permutation(TRAIN)
        for rank in ranks:
            t=len(rows);a,c=gains(t);delta=2*rng.integers(0,2,size=16)-1
            rows.append(dict(t=t,cycle=cycle,rank=int(rank),a=a,c=c,delta=delta.tolist()))
    return rows

def project(x,base):
    d=x-base;n=float(np.linalg.norm(d));return base+d*min(1.,.75/max(n,1e-300))

def estimate(lp,lm,c,delta):return (lp-lm)/(2*c)*np.asarray(delta,dtype=np.float64)

def update(theta,base,lp,lm,item):
    g=estimate(lp,lm,item['c'],item['delta']);norm=float(np.linalg.norm(g))
    clipped=g/max(1.,norm);new=project(theta-item['a']*clipped,base)
    assert np.linalg.norm(new-base)<=.750000000001
    return new,dict(gradient=g.tolist(),gradient_norm=norm,clipped_gradient=clipped.tolist(),center=new.tolist(),distance_from_parent=float(np.linalg.norm(new-base)))
