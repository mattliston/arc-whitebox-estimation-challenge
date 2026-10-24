"""Flopscope-only positive path factor; no estimator or unmetered math."""
import flopscope.numpy as fnp


def positive_path_factor(C,g,d,eta,K21,K22,radial):
    B=g[:,None]*C
    D=(.5*g[:,None]*K21.T*eta[None,:]+.5*d[:,None]*K21
       +d[:,None]*eta[None,:]*(.25*K22+.5*C*C))
    fnp.fill_diagonal(B,0);fnp.fill_diagonal(D,0)
    E=D/(1+radial/3)
    norm2=fnp.sum(B*B,axis=0)
    parallel=fnp.sum(B*E,axis=0)/fnp.maximum(norm2,1e-60)
    F=E-B*parallel[None,:]
    perp2=fnp.sum(F*F,axis=0)/fnp.maximum(norm2,1e-60)
    diagonal=1+2*parallel
    positive=.5*(diagonal+fnp.sqrt(diagonal*diagonal+4*perp2))
    scale=fnp.sqrt(fnp.maximum(positive,0))/fnp.sqrt(fnp.maximum(positive*positive+perp2,1e-60))
    Z=(B*positive[None,:]+F)*scale[None,:]
    return fnp.where(norm2[None,:]>0,Z,0)
