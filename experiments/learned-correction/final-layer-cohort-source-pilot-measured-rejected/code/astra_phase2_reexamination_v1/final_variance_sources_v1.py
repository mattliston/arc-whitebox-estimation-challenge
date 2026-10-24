"""Exact current pair covariance split, before any predicted-mean feedback."""
import math
import numpy as np
from k3_radial_numpy import gaussian_raw
from radial_mehler import derivatives

NAMES=tuple([f'gaussian_r{r}' for r in range(1,7)]+
            [f'third_r{r}' for r in range(3)]+
            [f'fourth_r{r}' for r in range(3)]+['post_diagonal','mean_feedback'])


def covariance_parts(mu,C,K21,k4,k22,post_covariance):
    v=np.diag(C).copy()
    s,a,pdf,g,mg,*_=gaussian_raw(mu,v)
    rho=C/np.outer(s,s);np.fill_diagonal(rho,0)
    T21=K21/(v[:,None]*s[None,:]);np.fill_diagonal(T21,0)
    T22=k22/np.outer(v,v)
    t3=np.diag(K21)/(s*s*s);t4=k4/(v*v)
    T31=t4[:,None]*rho
    A,_=derivatives(a,pdf,g,mg/s,6)
    scale=np.outer(s,s)
    parts=[]
    for r in range(1,7):
        parts.append(rho**r/math.factorial(r)*np.outer(A[r],A[r])*scale)
    for r in range(3):
        value=(T21*np.outer(A[r+2],A[r+1])+T21.T*np.outer(A[r+1],A[r+2]))/2
        if r:value+=(np.outer(t3*A[r+3],A[r])+np.outer(A[r],t3*A[r+3]))/6
        value*=rho**r/math.factorial(r)*scale
        np.fill_diagonal(value,0);parts.append(value)
    for r in range(3):
        value=T22*np.outer(A[r+2],A[r+2])/4
        value+=(T31*np.outer(A[r+3],A[r+1])+T31.T*np.outer(A[r+1],A[r+3]))/6
        if r:value+=(np.outer(t4*A[r+4],A[r])+np.outer(A[r],t4*A[r+4]))/24
        value*=rho**r/math.factorial(r)*scale
        np.fill_diagonal(value,0);parts.append(value)
    parts.append(np.diag(np.diag(post_covariance)))
    return np.stack(parts)


def masked_with_feedback(parts,host,delta,alive):
    fb=-np.outer(host,delta)-np.outer(delta,host)-np.outer(delta,delta)
    out=np.concatenate([parts,fb[None]],axis=0)
    diag=np.diagonal(out,axis1=1,axis2=2).copy()
    out*=alive[None,:,None]*alive[None,None,:]
    ix=np.arange(len(host));out[:,ix,ix]=diag
    return out


def variance_reads(parts,W):
    w=W.astype(np.float32);wd=w.astype(np.float64)
    return np.stack([np.sum(wd*(p.astype(np.float32)@w).astype(np.float64),axis=0) for p in parts],axis=1)
