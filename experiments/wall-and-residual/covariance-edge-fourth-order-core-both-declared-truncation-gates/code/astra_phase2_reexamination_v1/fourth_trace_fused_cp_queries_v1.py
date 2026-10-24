"""Reuse thin CP projections across all third-input fourth-trace terms."""
import numpy as np


def cp_query(p,lam,K,modes,h1,h2,g,d,X):
    n,q = X.shape
    s = modes.shape[1]
    edge = modes@modes.T
    np.fill_diagonal(edge,0)
    c0 = (modes*modes).sum(1)
    gx,dx = g[:,None]*X,d[:,None]*X
    queries = np.concatenate([gx]+[v[:,None]*dx for v in modes.T],axis=1)
    projections = p.T@queries
    ph = p.T@h1
    images = (p*(lam*ph))@projections
    out = g[:,None]*images[:,:q]
    for j,v in enumerate(modes.T):
        out += (d*v)[:,None]*images[:,(j+1)*q:(j+2)*q]
    out -= (d*d*c0*(K@h1))[:,None]*X

    # Edge-times-third contraction E = G P^T, before coincidences.
    contracted = p.T@(h2[:,None]*modes)
    G = (modes@contracted.T-(c0*h2)[:,None]*p)*(p*lam)
    out += d[:,None]*(G@projections[:,:q])
    out += g[:,None]*(p@(G.T@dx))

    Frep = h1[:,None]*K+K.T*h1[None,:]
    Frep += np.diag(K@h1-2*h1*np.diag(K))
    Erep = edge*K.T*h2[None,:]
    Erep += np.diag((edge*K)@h2)
    repair = Frep*(np.outer(g,g)+edge*np.outer(d,d))
    repair += Erep*np.outer(d,g)+Erep.T*np.outer(g,d)
    return out-repair@X


class FusedCPQueries:
    def __init__(self,p,lam,repeated,modes,other_queries=None):
        self.p,self.lam,self.K,self.V = p,lam,repeated,modes
        self.other = other_queries

    def fourth_third_part(self,h1,h2,g,d,X):
        out = cp_query(self.p,self.lam,self.K,self.V,h1,h2,g,d,X)
        if self.other is not None:
            out += self.other.fourth_third_part(h1,h2,g,d,X)
        return out
