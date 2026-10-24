"""Diagonal-trace K4 anisotropy, composed with its first-order K3 source.

NumPy research only. The trace projection and O(E^2) birth remainder are
explicit approximations. Neither targets nor trained parameters enter here.
"""
import inspect
import textwrap
import numpy as np
import radial_mehler
import radial_compact_feed_fullpair_v1 as parent
from radial_cold_cp_trace_v1 import trace_fit


def fit_diagonal(C, D, target_trace):
    D = D.copy()
    np.fill_diagonal(D, 0)
    v = np.diag(C); ct = np.trace(C)
    rhs = 6*target_trace-4*np.sum(C*D, axis=1)
    den = ct+4*v
    dt = np.sum(rhs/den)/(1+np.sum(v/den))
    np.fill_diagonal(D, (rhs-v*dt)/den)
    return D


def projected_extra_source(d, Cpre, Epre, g, Cpost, alive):
    gg = np.outer(g*alive, g*alive)
    Cg, Eg = gg*Cpre, gg*Epre
    u, v = d*np.diag(Cpre)*alive, d*np.diag(Epre)*alive
    target = (np.trace(Eg)*u+2*(Eg@u)+np.trace(Cg)*v+2*(Cg@v))/12
    metric = Cpost*alive[:,None]*alive[None,:]
    return trace_fit(target, metric)


nlsource = textwrap.dedent(inspect.getsource(parent.env['nonlinear']))
old = 'metric="identity"):'
assert nlsource.count(old) == 1
nlsource = nlsource.replace(old, 'metric="identity", k31_override=None):')
old = 'T31=t4[:,None]*rho if k31 else None'
assert nlsource.count(old) == 1
nlsource = nlsource.replace(old, 'T31=k31_override/(s[:,None]**3*s[None,:]) if k31_override is not None else (t4[:,None]*rho if k31 else None)')
old = 'return m,cov,post21,radial,aux'
assert nlsource.count(old) == 1
nlsource = nlsource.replace(old, 'aux["post22"]=post22\n    '+old)
nlenv = dict(vars(radial_mehler), gauss_order=6, edge_order=2, local=True, k31=True)
exec(nlsource, nlenv)

source = parent.PREDICT_SOURCE
edits = {
    '    radial = 0.0': '    radial = 0.0\n    E = np.zeros((n,n))',
    '        bpre = w.T@b': '        bpre = w.T@b\n        Epre = (w32.T@E.astype(np.float32)@w32).astype(np.float64)',
    '        k4 = radial*vv*vv\n        k22 = radial*(np.outer(vv, vv)+2*Cpre*Cpre)/3':
        '''        Dpre = radial*Cpre+Epre
        dv = np.diag(Dpre)
        k4 = vv*dv
        k22 = (np.outer(vv,dv)+np.outer(dv,vv)+4*Cpre*Dpre)/6
        k31_plane = (vv[:,None]*Dpre+dv[:,None]*Cpre)/2''',
    'nonlinear(mu, Cpre, K21, k4, k22, metric)': 'nonlinear(mu, Cpre, K21, k4, k22, metric, k31_plane)',
    '        b = (g*bpre+radial*d*vv/6)*alive':
        '        b = (g*bpre+radial*d*vv/6+projected_extra_source(d,Cpre,Epre,g,C,alive))*alive',
    '        bp = g[:, None]*Cpre':
        '''        column_shift = Epre/(6*(1+radial/3))
        column_shift_relative = float(np.linalg.norm(column_shift)/np.linalg.norm(Cpre))
        bp = g[:, None]*(Cpre+column_shift)''',
    'states.append(dict(layer=l, active=int(alive.sum()), births=len(selected), columns=p.shape[1]))':
        'states.append(dict(layer=l, active=int(alive.sum()), births=len(selected), columns=p.shape[1],column_shift_relative=column_shift_relative,radial=float(radial),anisotropy_relative=float(np.linalg.norm(Epre)/np.linalg.norm(Cpre))))',
    '        radial = radial_new':
        '''        Dpost = fit_diagonal(C,np.outer(g,g)*Dpre,np.sum(aux["post22"],axis=1))
        E = (Dpost-radial_new*C)*live
        radial = radial_new''',
}
for old,new in edits.items():
    assert source.count(old) == 1, old
    source = source.replace(old,new)
env = dict(parent.env, nonlinear=nlenv['nonlinear'], fit_diagonal=fit_diagonal,
           projected_extra_source=projected_extra_source)
exec(source,env)
predict = env['predict']
