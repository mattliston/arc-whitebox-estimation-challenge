"""Target-blind all-distinct cubic information along covariance directions.

Existing plane row/column summaries cannot recover T[:,u,u]. These exact
contractions of the carried CP+star tensor cost O(n*bank), not a new bank
push. Directions use the mean and its covariance action, standardized and
orthogonalized in the covariance metric. No truth or applied-current-mean
correction enters either the directions or their tensor contractions.
"""
import numpy as np

DIRECTION_NAMES=("cubic_u1u1","cubic_u1u2","cubic_u2u2","corr_u1","corr_u2",
                 "cubic_111","cubic_112","cubic_122","cubic_222")


def star_bilinear(A,q,r):
    return (A@(q*r)+q*(A.T@r)+r*(A.T@q))/3


def direction_features(mu,C,pp,lam,W,star):
    n=len(mu)
    cm=C@mu
    u1=mu/np.sqrt(np.maximum(mu@cm,1e-20))
    r=C@u1
    cr=C@r
    u2=r-u1*(u1@cr)
    cu2=C@u2
    u2=u2/np.sqrt(np.maximum(u2@cu2,1e-20))
    p64=pp.astype(np.float64)
    directions=np.stack([u1,u2],axis=1)
    projections=p64.T@directions
    weights=np.stack([projections[:,0]**2,projections[:,0]*projections[:,1],projections[:,1]**2],axis=1)*lam.astype(np.float64)[:,None]
    contractions=p64@weights
    w=W.astype(np.float64);A=star.astype(np.float64)
    q=w@directions
    star_inputs=np.stack([star_bilinear(A,q[:,0],q[:,0]),star_bilinear(A,q[:,0],q[:,1]),star_bilinear(A,q[:,1],q[:,1])],axis=1)
    contractions=contractions+w.T@star_inputs
    s=np.sqrt(np.maximum(np.diag(C),1e-10))
    couplings=(C@directions)/s[:,None]
    ones=np.ones(n)
    globals_=np.stack([ones*(u1@contractions[:,0]),ones*(u2@contractions[:,0]),
                       ones*(u1@contractions[:,2]),ones*(u2@contractions[:,2])],axis=1)
    return np.concatenate([contractions/s[:,None],couplings,globals_],axis=1).astype(np.float32)
