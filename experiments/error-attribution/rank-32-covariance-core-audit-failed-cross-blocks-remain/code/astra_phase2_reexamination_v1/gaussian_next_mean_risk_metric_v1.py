"""Exact expected squared Gaussian mean response with frozen variance."""
import numpy as np


def geometry(mean,covariance,tau2):
    v0=float(tau2*np.trace(covariance))
    norm2=float(mean@mean)
    assert v0>0 and tau2>0 and norm2>0
    u=mean/np.sqrt(norm2)
    t=2*tau2*norm2/v0
    beta=t/(1+t)
    factor=tau2*tau2/(8*np.pi*v0*np.sqrt(1+t))
    return u,beta,factor,v0


def risk(E,u,beta,factor):
    E=(E+E.T)/2
    Eu=E@u;ueu=float(u@Eu)
    tr=float(np.trace(E))-beta*ueu
    frobenius=float(np.sum(E*E))-2*beta*float(Eu@Eu)+beta*beta*ueu*ueu
    return factor*(2*frobenius+tr*tr)


def variable_response(mean,covariance,W):
    mu=mean@W
    variance=np.sum(W*(covariance@W),axis=0)
    assert variance.min()>0
    return np.exp(-mu*mu/(2*variance))/np.sqrt(2*np.pi)/(2*np.sqrt(variance))
