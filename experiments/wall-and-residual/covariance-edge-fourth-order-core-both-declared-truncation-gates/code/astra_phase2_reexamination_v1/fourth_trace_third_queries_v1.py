"""Thin queries for distinct-third terms in a fourth partial trace.

Exact only for the supplied covariance-edge low-rank model. Not a complete
fourth-trace evaluator, a truncation certificate or deployable estimator.
"""
import numpy as np


class ThirdContraction:
    def __init__(self, p, lam, star, star_push, b, metric):
        self.p,self.lam,self.star = p,lam,star
        self.w,self.b,self.metric = star_push,b,metric

    def __call__(self, h, X):
        p,lam,A,w,b,B = self.p,self.lam,self.star,self.w,self.b,self.metric
        out = (p*((p.T@h)*lam))@(p.T@X)
        hp,Xp = w@h,w@X
        star = (A@(hp[:,None]*Xp)+hp[:,None]*(A.T@Xp)+(A.T@hp)[:,None]*Xp)/3
        out += w.T@star
        bh = B@h
        out += (b@h)*(B@X)+b[:,None]*(bh@X)[None,:]+bh[:,None]*(b@X)[None,:]
        return out


class DistinctQueries:
    def __init__(self, contraction, repeated, modes):
        self.M,self.K,self.V = contraction,repeated,modes
        self.c0 = (modes*modes).sum(1)

    def F(self, h, X):
        K = self.K
        diag = K@h-2*h*np.diag(K)
        return self.M(h,X)-h[:,None]*(K@X)-K.T@(h[:,None]*X)-diag[:,None]*X

    def edge_hadamard(self, K, X):
        out = -(self.c0*np.diag(K))[:,None]*X
        for v in self.V.T:
            out += v[:,None]*(K@(v[:,None]*X))
        return out

    def E(self, h, X, transpose=False):
        K = self.K
        diag = -(self.c0*h)*np.diag(K)
        for v in self.V.T:
            diag += v*(K@(h*v))
        if not transpose:
            out = -(self.c0*h)[:,None]*(K@X)
            for v in self.V.T:
                out += v[:,None]*self.M(h*v,X)
            out -= self.edge_hadamard(K.T,h[:,None]*X)
        else:
            out = -K.T@((self.c0*h)[:,None]*X)
            for v in self.V.T:
                out += self.M(h*v,v[:,None]*X)
            out -= h[:,None]*self.edge_hadamard(K,X)
        return out-diag[:,None]*X

    def fourth_third_part(self, h1, h2, g, d, X):
        out = g[:,None]*self.F(h1,g[:,None]*X)
        dx = d[:,None]*X
        for v in self.V.T:
            # F's diagonal is identically zero, so the removed diagonal of
            # V V^T contributes zero to this Hadamard product.
            out += (d*v)[:,None]*self.F(h1,v[:,None]*dx)
        out += d[:,None]*self.E(h2,g[:,None]*X)
        out += g[:,None]*self.E(h2,dx,transpose=True)
        return out
