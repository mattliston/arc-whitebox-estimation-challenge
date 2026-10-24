"""Scalar final-layer map: no off-diagonal moments are consumed afterward."""
import numpy as np
from k3_radial_numpy import gaussian_raw


def nonlinear_scalar(mu, v, k3, k4):
    s,a,pdf,g,mg,r2g,r3g,r4g = gaussian_raw(mu,v)
    d = pdf/s; e = -a*pdf/v; f = (a*a-1)*pdf/(v*s)
    m = mg+k3*e/6+k4*f/24
    r2 = r2g+k3*d/3+k4*e/12
    r3 = r3g+k3*g+k4*d/4
    r4 = r4g+4*k3*mg+k4*g
    var = r2-m*m
    post_k3 = r3-3*m*r2+2*m**3
    post_k4 = r4-4*m*r3+6*m*m*r2-3*m**4-3*var*var
    aux = dict(sigma=s,alpha=a,pdf=pdf,gate=g,gaussian_mean=mg,pre_v=v,pre_k3=k3,
               pre_k4=k4,post_v=var,post_k3=post_k3,post_k4=post_k4)
    return m,aux


def final_scalar_features(l,m,aux,W):
    n = len(m); s = aux["sigma"]; v = s*s
    w = W.astype(np.float32); w2 = w*w; w3 = w2*w; w4 = w2*w2
    ones = np.ones(n); zero = np.zeros(n)
    vals = (ones*(2*l/15-1),ones*(l/16),ones,aux["alpha"],np.log(s)-np.mean(np.log(s)),
            aux["gate"],aux["pdf"],aux["gaussian_mean"]/s,m/s,(m-aux["gaussian_mean"])/s,
            aux["pre_k3"]/(v*s),aux["post_v"]/v,aux["pre_k4"]/(v*v),zero,zero,zero,
            np.sum(w,axis=0)*0.7071067811865475,np.sum(w2-0.001953125,axis=0)*11.313708498984761,
            np.sum(w3,axis=0)*93.47798314754834,np.sum(w4-1.1444091796875e-5,axis=0)*836.0924988699916,
            aux["post_k3"]/(v*s),aux["post_k4"]/(v*v),zero,zero,zero,zero,zero,zero,zero,
            ones*np.mean(m/s))
    return np.stack(vals,axis=1).astype(np.float32)
