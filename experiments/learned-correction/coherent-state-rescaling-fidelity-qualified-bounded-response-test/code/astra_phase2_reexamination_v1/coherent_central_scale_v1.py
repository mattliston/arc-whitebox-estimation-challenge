"""Independent NumPy coherent rescaling of every represented central moment."""
import numpy as np
import compact_fullpair_final_reference_v1 as parent


def scaling(layer, aux, theta):
    n = len(aux['sigma'])
    if layer == 0:
        return np.ones(n, np.float64)
    s = aux['sigma']
    raw = np.stack((aux['pre_k3'] / s**3, aux['gate'], np.log(s)), 1)
    centered = raw - raw.mean(0)
    phi = np.concatenate((np.ones((n, 1)),
                          np.clip(centered / np.sqrt((centered**2).mean(0) + 1e-12), -4, 4)), 1)
    z = 2 * layer / 15 - 1
    basis = np.asarray((1, z, (3*z*z-1)/2))
    return np.exp(.002 * np.tanh(phi @ (basis @ theta)))


def transform(C, p, star, b, scale):
    return (scale[:, None] * C * scale[None, :],
            (scale[:, None] * p).astype(p.dtype),
            (scale[:, None] * star * scale[None, :]**2).astype(star.dtype),
            scale * b)


source = parent.source
old = 'gate_floor=.01, final_hook=None):'
assert source.count(old) == 1
source = source.replace(old, 'gate_floor=.01, final_hook=None, scale_hook=None):')
old = '        if collect_states:\n'
assert source.count(old) == 1
source = source.replace(old, '''        if scale_hook is not None:
            scale = scale_hook(l,aux)
            C,p,star,b = transform(C,p,star,b,scale)
''' + old)
env = dict(parent.env, transform=transform)
exec(source, env)
predict = env['predict']
