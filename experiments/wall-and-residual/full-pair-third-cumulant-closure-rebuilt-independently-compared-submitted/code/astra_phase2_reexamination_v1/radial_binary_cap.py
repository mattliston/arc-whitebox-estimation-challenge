"""Cap signed cube PAIRS, preserving their cancellation and repeated planes.

Pair self energy is a^2 (||p||^6+||q||^6-2(p.q)^3), not the sum
of separate cube energies. Only dropped terms are read to update the star,
so the cap does not require another full retained-bank matrix read.
"""
import inspect
import numpy as np
import radial_binary_cubic as binary

def make_predict(cap):
    src=inspect.getsource(binary.predict)
    old='''        p=np.concatenate([g.astype(np.float32)[:,None]*pp,p1.astype(np.float32),p2.astype(np.float32)],axis=1)
        lam=np.concatenate([lam,a.astype(np.float32),-a.astype(np.float32)])'''
    new='''        ng=len(lam)//2
        positive=np.concatenate([g.astype(np.float32)[:,None]*pp[:,:ng],p1.astype(np.float32)],axis=1)
        negative=np.concatenate([g.astype(np.float32)[:,None]*pp[:,ng:],p2.astype(np.float32)],axis=1)
        weights=np.concatenate([lam[:ng],a.astype(np.float32)])
        if len(weights)>pair_cap:
            aa=np.sum(positive.astype(np.float64)**2,axis=0)
            bb=np.sum(negative.astype(np.float64)**2,axis=0)
            ab=np.sum(positive.astype(np.float64)*negative.astype(np.float64),axis=0)
            energy=weights.astype(np.float64)**2*np.maximum(aa**3+bb**3-2*ab**3,0)
            order=np.argsort(-energy,kind="stable");keep=order[:pair_cap];drop=order[pair_cap:]
            dropped=np.concatenate([positive[:,drop],negative[:,drop]],axis=1)
            dw=np.concatenate([weights[drop],-weights[drop]])
            cp_post=cp_post-plane(dropped,dw)
            positive,negative,weights=positive[:,keep],negative[:,keep],weights[keep]
        p=np.concatenate([positive,negative],axis=1)
        lam=np.concatenate([weights,-weights])'''
    assert src.count(old)==1
    env=dict(vars(binary));env["pair_cap"]=cap
    exec(src.replace(old,new),env)
    return env["predict"]
