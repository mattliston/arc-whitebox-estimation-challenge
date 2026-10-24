"""Exact next variance query for the CP+coordinate+trace third tensor."""
import numpy as np


def variance_response(p,lam,star,b,B,eta,W):
    pp=W.T@p
    value=np.sum(pp*pp*(lam*(p.T@eta))[None,:],axis=1)
    A=W.T@star
    value+=(2*np.sum(A*W.T*eta[None,:],axis=1)+(W*W).T@(star.T@eta))/3
    BW=B@W;bv=np.sum(W*BW,axis=0)
    value+=(b@eta)*bv+2*(W.T@b)*(W.T@(B@eta))
    return value


def certificate():
    rng=np.random.default_rng(26952015);n=8;r=13
    p=rng.normal(size=(n,r))*.2;lam=rng.normal(size=r);star=rng.normal(size=(n,n))*.01
    b=rng.normal(size=n)*.01;B=rng.normal(size=(n,n));B=B@B.T/n
    eta=rng.normal(size=n)*.01;W=rng.normal(size=(n,n))/np.sqrt(n);I=np.eye(n)
    T=np.einsum('ia,ja,ka,a->ijk',p,p,p,lam)
    S=np.einsum('ij,jk->ijk',star,I);T+=(S+S.transpose(1,0,2)+S.transpose(2,1,0))/3
    T+=np.einsum('i,jk->ijk',b,B)+np.einsum('j,ik->ijk',b,B)+np.einsum('k,ij->ijk',b,B)
    M=np.einsum('ijk,k->ij',T,eta)
    exact=np.diag(W.T@M@W);query_error=float(np.max(np.abs(variance_response(p,lam,star,b,B,eta,W)-exact)))
    X=rng.normal(size=(4096,n));X=np.maximum(X,0);m=X.mean(0);Z=X-m
    C=Z.T@Z/len(Z);v=rng.normal(size=n)*.1
    expected=np.einsum('bi,bj,b->ij',Z,Z,Z@v)/len(Z)
    h=1e-4
    def moments(t):
        a=np.exp(t*(X@v));a/=a.sum();mu=a@X
        Y=X-mu;return mu,(Y*a[:,None]).T@Y
    mp,cp=moments(h);mm,cm=moments(-h)
    mean_error=float(np.max(np.abs((mp-mm)/(2*h)-C@v)))
    covariance_error=float(np.max(np.abs((cp-cm)/(2*h)-expected)))
    assert query_error<1e-13 and max(mean_error,covariance_error)<1e-8
    return dict(passed=True,query_max_error=query_error,tilt_mean_derivative_max_error=mean_error,tilt_covariance_derivative_max_error=covariance_error)
