"""Balanced signed-cube representation of the low-mode anisotropic source."""
import numpy as np


def source_factors(Z,eta,V,variance,g,d):
    n,r = Z.shape
    s = V.shape[1]
    columns,weights = [],[]
    def unit(x):
        norm = float(np.linalg.norm(x))
        return x/max(norm,1e-30),norm
    def add(x,weight):
        columns.append(x)
        weights.append(float(weight))
    def uvv_group(u,vs,coefficients):
        un,nu = unit(u)
        pure = 0.
        for v,coefficient in zip(vs.T,coefficients):
            vn,nv = unit(v)
            coefficient = float(coefficient)*nu*nv*nv
            add(vn+un,coefficient/6)
            add(vn-un,-coefficient/6)
            pure -= coefficient/3
        add(un,pure)
    diagonal_A = (Z*Z)@eta
    uvv_group(d*diagonal_A,g[:,None]*V,np.ones(s)/4)
    uvv_group(d*variance,g[:,None]*Z,eta/4)
    for a in range(r):
        for b in range(s):
            un,nu = unit(d*Z[:,a]*V[:,b])
            vn,nv = unit(g*Z[:,a])
            wn,nw = unit(g*V[:,b])
            coefficient = eta[a]*nu*nv*nw/24
            add(un+vn+wn,coefficient)
            add(un-vn-wn,coefficient)
            add(un+vn-wn,-coefficient)
            add(un-vn+wn,-coefficient)
    factors = np.stack(columns,axis=1)
    coefficients = np.asarray(weights)
    assert factors.shape[1] == 4*r*s+2*r+2*s+2
    return factors,coefficients
