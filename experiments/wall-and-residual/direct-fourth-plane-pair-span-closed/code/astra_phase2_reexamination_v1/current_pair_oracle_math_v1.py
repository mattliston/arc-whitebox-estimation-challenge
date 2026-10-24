"""Exact finite substitutions for the current first-order pair consumer.

Diagnostic only. Off-diagonal pair expressions are linear in cumulants;
the central diagonal has the additional quadratic change of the mean.
"""
import inspect
import textwrap
import numpy as np
from quadratic_skew_covariance_v1 import response
from radial_mehler import make_nonlinear


def mean_change(sigma, alpha, pdf, dskew, dkurt):
    return sigma*pdf*(-alpha*dskew/6+(alpha*alpha-1)*dkurt/24)


def exact_change(common, dS=None, dQ22=None, dQ31=None, dq4=None):
    R,sigma,alpha,pdf,gate,mg,host=common
    skew=np.zeros(len(R)) if dS is None else np.diag(dS)
    kurt=np.zeros(len(R)) if dq4 is None else dq4
    dm=mean_change(sigma,alpha,pdf,skew,kurt)
    dc=response(*common,dS=dS,dQ22=dQ22,dQ31=dQ31,dq4=dq4)
    dc[np.diag_indices(len(R))]-=dm*dm
    return dm,dc


def independent_fourth_map():
    original=make_nonlinear(6,2,True,True)
    source=textwrap.dedent(inspect.getsource(original))
    old='metric="identity"):';assert source.count(old)==1
    source=source.replace(old,'metric="identity",q31_override=None):')
    old='T31=t4[:,None]*rho if k31 else None';assert source.count(old)==1
    source=source.replace(old,old+'\n    if q31_override is not None: T31=q31_override')
    env=dict(original.__globals__,gauss_order=6,edge_order=2,local=True,k31=True)
    exec(source,env)
    return env['nonlinear']
