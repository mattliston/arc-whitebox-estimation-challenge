"""Conditional signed-birth reweighting with incremental plane restoration.

NumPy research reference. Zero coefficients leave the parent unchanged.
The increment uses STORED columns/coefficients, including clipped diagonals.
"""
import numpy as np
import radial_compact_feed_fullpair_v1 as parent


def birth_scale(layer,state,birth,births,theta):
    if layer == 0:
        return np.ones(len(birth),np.float32)
    h = np.asarray(state,np.float32)
    weight = np.asarray(birth*births,np.float32)
    weight = weight/np.maximum(weight.sum(),np.float32(1e-20))
    centered = h-(weight[:,None]*h).sum(0)
    rms = np.sqrt((weight[:,None]*centered*centered).sum(0)+np.float32(1e-6))
    standardized = np.clip(centered/rms,-8,8)
    z = np.float32(2*layer/15-1)
    basis = np.asarray([1,z,(3*z*z-1)/2],np.float32)
    coefficient = basis@np.asarray(theta,np.float32)
    response = np.tanh(standardized@coefficient)
    return np.float32(1)+np.float32(.2)*(response-(weight*response).sum())


def coefficient_plane_shift(p1,p2,a0,a1,births):
    first = p1.astype(np.float32).astype(np.float64)
    t = np.diag(first).copy()
    s = np.diag(p2.astype(np.float32)).astype(np.float64)
    np.fill_diagonal(first,0)
    da = (a1.astype(np.float64)-a0.astype(np.float64))*births
    scale = da*(t-s)
    shift = first*first*scale[None,:]+(scale*(t+s))[:,None]*first.T
    np.fill_diagonal(shift,scale*(t*t+t*s+s*s))
    return shift


PREDICT_SOURCE = parent.PREDICT_SOURCE
edits = {
    'gate_floor=.001):':'gate_floor=.001, birth_hook=None):',
    '        cp_post = (inherited+born)*live':
        '''        a0 = a.astype(np.float32)
        eta = np.ones(n,np.float32) if birth_hook is None else np.asarray(birth_hook(l,birth,births),np.float32)
        a1 = (a0*eta).astype(np.float32)
        shift = coefficient_plane_shift(p1,p2,a0,a1,births)
        cp_post = (inherited+born+shift)*live''',
    'lam = np.concatenate([lam, a[selected].astype(np.float32), -a[selected].astype(np.float32)])':
        'lam = np.concatenate([lam, a1[selected], -a1[selected]])',
}
for old,new in edits.items():
    assert PREDICT_SOURCE.count(old) == 1,old
    PREDICT_SOURCE = PREDICT_SOURCE.replace(old,new)
env = dict(parent.env,coefficient_plane_shift=coefficient_plane_shift)
exec(PREDICT_SOURCE,env)
predict = env['predict']
