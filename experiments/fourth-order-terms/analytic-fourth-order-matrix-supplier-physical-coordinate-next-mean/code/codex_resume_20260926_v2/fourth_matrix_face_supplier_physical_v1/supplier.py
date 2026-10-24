"""Passively expose the existing pair map's analytic fourth output faces."""
import inspect
import textwrap
from radial_mehler import make_nonlinear


def make_supplier():
    original=make_nonlinear(6,2,True,True)
    source=textwrap.dedent(inspect.getsource(original))
    edits={
        'metric="identity"):':'metric="identity",q31_override=None):',
        'T31=t4[:,None]*rho if k31 else None':
            'T31=t4[:,None]*rho if k31 else None\n    if q31_override is not None: T31=q31_override',
        '    return m,cov,post21,radial,aux':
            '''    H=[r3g/s**3]+[3*B[r-1] for r in range(1,maxorder+1)]
    pk31=pair(H,A)*(s[:,None]**3*s[None,:])
    # pair(A,A) before scalar diagonal restoration is needed only off diagonal.
    post31=pk31-3*m[:,None]*pk21+(3*m*m-3*var)[:,None]*cov
    np.fill_diagonal(post31,post_k4)
    aux["post31"]=post31
    aux["post22"]=post22
    return m,cov,post21,radial,aux'''
    }
    for old,new in edits.items():
        assert source.count(old)==1,old
        source=source.replace(old,new)
    env=dict(original.__globals__,gauss_order=6,edge_order=2,local=True,k31=True)
    exec(source,env)
    return env['nonlinear']
