"""Phase2 n_input=1024, exact fixed chi-radius plane algebra only."""
import numpy as np

INPUT_DIM = 1024
C1 = .9997558892136928
D1 = .0004881619825383952
A3 = 1.000732213324253
A4 = 1.001953125
BETA = 2.384766942619451e-7
U4 = .0014652012032244678
Z4 = 2*D1/INPUT_DIM
W4 = 1.7069648435085733e-13


def gaussian_planes(mean, covariance, third21, radial, bank_fourth_trace=None):
    mu = mean/C1
    B = covariance-D1*np.outer(mu, mu)
    v = np.diag(B)
    S = (third21-C1/INPUT_DIM*(2*mu[:, None]*B+v[:, None]*mu[None, :])
         -BETA*mu[:, None]**2*mu[None, :])/A3
    DB = float(np.trace(B)**2+2*np.sum(B*B))
    if not np.isfinite(DB) or DB <= 0:
        raise ValueError('Angular fourth radial denominator is undefined.')
    if bank_fourth_trace is None:
        bank_fourth_trace = float(radial/3*(np.trace(covariance)**2+2*np.sum(covariance*covariance)))
    norm2 = float(mu@mu)
    trace_R = float(4*U4*(mu@S.sum(axis=0))
                    +Z4*(2*norm2*np.trace(B)+4*mu@B@mu)+W4*norm2**2)
    eta = (bank_fourth_trace-trace_R)/DB
    mu2 = mu*mu
    k22 = (eta*(np.outer(v, v)+2*B*B)
           +2*U4*(mu[:, None]*S.T+S*mu[None, :])
           +Z4*(np.outer(mu2, v)+np.outer(v, mu2)+4*np.outer(mu, mu)*B)
           +W4*np.outer(mu2, mu2))
    k31 = (3*eta*v[:, None]*B
           +U4*(3*mu[:, None]*S+np.diag(S)[:, None]*mu[None, :])
           +3*Z4*(mu2[:, None]*B+(mu*v)[:, None]*mu[None, :])
           +W4*mu[:, None]**3*mu[None, :])
    k4 = 3*eta*v*v+4*U4*mu*np.diag(S)+6*Z4*mu2*v+W4*mu2**2
    return dict(mu_A=mu, C_A=B, K21_A=S, k22=k22, k31=k31, k4=k4,
                eta=eta, angular_radial=3*(eta-2/INPUT_DIM)/A4,
                angular_radial_denominator=DB, trace_R=trace_R,
                trace_bank=bank_fourth_trace, trace_new=float(k22.sum()))
