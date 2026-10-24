"""Full exact rank-two two-edge source, with unchanged CP birth diagonal."""
import numpy as np
import radial_compact_feed_fullpair_v1 as parent
from positive_path_resummation_v1 import factors, positive_factor


def signed_factors(C, K21, K22, aux, radial, enabled):
    B, E = factors(C, aux['gate'], aux['pdf']/aux['sigma'],
                   -aux['alpha']/aux['sigma'], K21, K22, radial)
    if not enabled:
        return B, np.zeros_like(B)
    positive, stats = positive_factor(B, E)
    F = E - B*stats['parallel'][None, :]
    negative = stats['negative']
    scale = np.sqrt(np.maximum(-negative, 0)) / np.sqrt(np.maximum(
        negative*negative + stats['perpendicular2'], 1e-60))
    Zminus = (B*negative[None, :] + F)*scale[None, :]
    Zminus[:, stats['norm2'] == 0] = 0
    return positive, Zminus


def compose(source):
    changes = (
        ('gate_floor=.001):', 'gate_floor=.001, signed_path=True):'),
        ('        bp = g[:, None]*Cpre',
         '        bp,negative_bp = signed_factors(Cpre,K21,k22,aux,radial,signed_path)'),
        ('        t = (u+delta)/2',
         '        birth_delta = delta\n        if signed_path:\n            delta = delta*np.sqrt(2.)\n        t = (u+delta)/2'),
        ('        np.fill_diagonal(born, birth*births*(3*u*u+delta*delta)/4)',
         '''        if signed_path:
            born -= negative_bp*negative_bp*(birth*births)[None,:]
        np.fill_diagonal(born, birth*births*(3*u*u+birth_delta*birth_delta)/4)'''),
        ('        p = alive.astype(np.float32)[:, None]*p',
         '''        if signed_path:
            neg1,neg2 = negative_bp.copy(),negative_bp.copy()
            np.fill_diagonal(neg1,birth_delta/2)
            np.fill_diagonal(neg2,-birth_delta/2)
            p = np.concatenate([p,neg1[:,selected].astype(np.float32),
                                neg2[:,selected].astype(np.float32)],axis=1)
        p = alive.astype(np.float32)[:, None]*p'''),
        ('        rest = target-cp_post-trace_post',
         '''        if signed_path:
            negative_a = birth/birth_delta
            lam = np.concatenate([lam,-negative_a[selected].astype(np.float32),
                                   negative_a[selected].astype(np.float32)])
        rest = target-cp_post-trace_post'''),
    )
    for old, new in changes:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    return source


source = compose(parent.PREDICT_SOURCE)
env = dict(parent.env, signed_factors=signed_factors)
exec(compile(source, __file__, 'exec'), env)
predict = env['predict']
