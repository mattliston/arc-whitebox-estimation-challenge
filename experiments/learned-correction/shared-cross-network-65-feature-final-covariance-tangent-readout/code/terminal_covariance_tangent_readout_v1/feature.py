"""Fixed formal tangent features; no label, fit, or finite covariance write."""
import numpy as np
from tangent import covariance_jacobian,terminal

def features(a):
    assert a['prediction'].dtype==np.float32 and a['prediction'].shape==(16,1024)
    assert np.isfinite(a['prediction']).all() and (a['prediction']>=0).all()
    assert a['C'].shape==a['W'].shape==(1024,1024)
    assert a['H14'].shape==a['H15'].shape==(1024,64)
    assert (a['v']>=1e-10).all()
    X=np.column_stack([a['H14'].astype(np.float64),np.ones(1024)])
    dv,da=covariance_jacobian(a['C'],a['W'],a['indices'],a['alive'],X)
    J=terminal(a,dv,da,native_anchors=True).d
    B=a['sigma'].astype(np.float64)[:,None]*np.column_stack([np.ones(1024),a['H15'].astype(np.float64)])
    assert J.shape==B.shape==(1024,65) and np.isfinite(J).all() and np.isfinite(B).all()
    return J,B
