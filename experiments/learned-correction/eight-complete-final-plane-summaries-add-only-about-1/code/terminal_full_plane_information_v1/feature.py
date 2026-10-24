"""Research FP64 complete formal plane and eight fixed summary columns."""
import numpy as np
NAMES=['rho_mean','rho_rms','rho_abs_mean','third_row_norm','third_column_norm',
       'third_row_sum','third_column_sum','rho_alpha']
def planes(cap):
    P=cap['P'].astype(np.float64);lam=cap['lam'].astype(np.float64)
    V=cap['W'].astype(np.float64).T;U=V@cap['A'].astype(np.float64)
    C=cap['C'].astype(np.float32).astype(np.float64);C=(C+C.T)*.5
    B=(V@C)@V.T
    K=(P*P)@(P*lam[None,:]).T
    K+=(2*((U*V)@V.T)+(V*V)@U.T)/3
    b=cap['b'].astype(np.float64)
    K+=np.diag(B)[:,None]*b[None,:]+2*B*b[:,None]
    return B,K
def summaries(B,K,sigma,alpha):
    s=sigma.astype(np.float64);a=alpha.astype(np.float64);n=len(s)
    assert np.isfinite(s).all() and (s>0).all()
    R=B/(s[:,None]*s[None,:]);T=K/(s[:,None]**2*s[None,:])
    F=np.column_stack([np.mean(R,axis=1),np.sqrt(np.mean(R*R,axis=1)),np.mean(np.abs(R),axis=1),
        np.sqrt(np.sum(T*T,axis=1)),np.sqrt(np.sum(T*T,axis=0)),
        np.sum(T,axis=1)/np.sqrt(n),np.sum(T,axis=0)/np.sqrt(n),R@a/n])
    return F*s[:,None]
def relative(a,b):
    ref=float(np.linalg.norm(b));return float(np.linalg.norm(a-b)/(ref if ref else 1.))
