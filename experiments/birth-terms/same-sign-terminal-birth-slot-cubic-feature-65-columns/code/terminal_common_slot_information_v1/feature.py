"""Fixed same-sign de-repeated source and complete formal terminal readout."""
import numpy as np
from tangent import terminal

def kernel(w,b,t,s,indices,alive):
    w=np.asarray(w,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    indices=np.asarray(indices);alive=np.asarray(alive,dtype=bool)
    assert w.ndim==2 and b.shape==(w.shape[0],len(indices))
    assert np.all(b[indices,np.arange(len(indices))]==0)
    b=b*alive[:,None];t=np.asarray(t,dtype=np.float64);s=np.asarray(s,dtype=np.float64)
    z1=w.T@b;z2=(w*w).T@(b*b);z3=(w*w*w).T@(b*b*b)
    e=w[indices].T*alive[indices][None,:]
    return 2*(z1*z1*z1-3*z2*z1+2*z3)+3*e*(t+s)[None,:]*(z1*z1-z2)

def features(a):
    assert a['prediction'].shape==(16,1024) and a['prediction'].dtype==np.float32
    assert a['H14'].shape==a['H15'].shape==(1024,64)
    assert (a['v']>=1e-10).all() and np.isfinite(a['prediction']).all()
    assert np.array_equal(a['birth_minus'],-a['birth_a'])
    ix=a['birth_indices'];D=kernel(a['W'],a['birth_base'],a['birth_t'],a['birth_s'],ix,a['alive'])
    X=np.column_stack([a['H14'].astype(np.float64),np.ones(1024)])
    K=D@(a['birth_a'].astype(np.float64)[:,None]*X[ix])
    J=terminal(a,K,native_anchors=True).d
    B=a['sigma'].astype(np.float64)[:,None]*np.column_stack([np.ones(1024),a['H15'].astype(np.float64)])
    assert J.shape==B.shape==(1024,65) and np.isfinite(J).all() and np.isfinite(B).all()
    theta=np.linspace(-.01,.01,65)
    collapsed=terminal(a,(K@theta)[:,None],native_anchors=True).d[:,0]
    error=float(np.max(np.abs(collapsed-J@theta))/max(1.,float(np.max(np.abs(collapsed)))))
    assert error<=2e-12,error
    return J,B,dict(collapsed_direction_scaled_error=error,k3_columns_rms=float(np.sqrt(np.mean(K*K))),
                    kernel_shape=list(D.shape),source_has_zero_current_repeated_entries=True)

