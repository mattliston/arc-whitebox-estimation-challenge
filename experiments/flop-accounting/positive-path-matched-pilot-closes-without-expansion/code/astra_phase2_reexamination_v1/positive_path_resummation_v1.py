"""Rank-one positive component of the exact 24-term path quadratic."""
import numpy as np
from linear_transport_residual_source_v1 import coord_slice


def factors(C,g,d,eta,K21,K22,radial):
    B=g[:,None]*C
    D=(.5*g[:,None]*K21.T*eta[None,:]+.5*d[:,None]*K21
       +d[:,None]*eta[None,:]*(.25*K22+.5*C*C))
    np.fill_diagonal(B,0);np.fill_diagonal(D,0)
    E=D/(1+radial/3)
    return B,E


def positive_factor(B,E):
    norm2=np.sum(B*B,axis=0)
    safe=np.maximum(norm2,1e-60)
    parallel=np.sum(B*E,axis=0)/safe
    F=E-B*parallel[None,:]
    perp2=np.sum(F*F,axis=0)/safe
    diagonal=1+2*parallel
    root=np.sqrt(diagonal*diagonal+4*perp2)
    positive=.5*(diagonal+root)
    scale=np.sqrt(np.maximum(positive,0))/np.sqrt(np.maximum(positive*positive+perp2,1e-60))
    Z=(B*positive[None,:]+F)*scale[None,:]
    Z[:,norm2==0]=0
    return Z,dict(positive=positive,negative=.5*(diagonal-root),norm2=norm2,
                  parallel=parallel,perpendicular2=perp2)


def birth_difference_slices(B,Z,t,H):
    K=(Z*Z-B*B)*t[None,:]
    star=3*K.T-2*np.diag(np.diag(K));answer=[]
    for h in H.T:
        X=(Z*(t*h)[None,:])@Z.T-(B*(t*h)[None,:])@B.T
        Y=(t*(Z.T@h))[:,None]*Z.T-(t*(B.T@h))[:,None]*B.T
        answer.append(X+Y+Y.T-coord_slice(star,h))
    return np.stack(answer)


def dense_difference(B,Z,t):
    I=np.eye(len(t))
    T=np.einsum('ia,ja,ka,a->ijk',Z,Z,I,t)-np.einsum('ia,ja,ka,a->ijk',B,B,I,t)
    T=T+T.transpose(0,2,1)+T.transpose(2,1,0)
    ii,jj,kk=np.indices(T.shape)
    return T*((ii!=jj)&(ii!=kk)&(jj!=kk))
