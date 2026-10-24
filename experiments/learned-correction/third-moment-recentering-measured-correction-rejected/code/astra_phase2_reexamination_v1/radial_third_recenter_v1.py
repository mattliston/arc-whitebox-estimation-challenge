"""Change third-cumulant centering after the existing mean feedback.

Offline NumPy reference. Full raw-moment algebra is certified separately;
active/cold truncation follows the existing compact host, not a new mask.
"""
import numpy as np
import radial_compact_feed_fullpair_v1 as base


def mean_cubes(m, delta):
    mn = max(float(np.linalg.norm(m)), 1e-30)
    dn = max(float(np.linalg.norm(delta)), 1e-30)
    scale = (dn/mn)**(1/3)
    a, b = m*scale, delta/(scale*scale)
    return np.stack([a+b, a-b, b, delta], axis=1), np.array([.5, -.5, -1., -1.])


def shift_plane(m, Cnew, delta):
    return ((m*m-np.diag(Cnew))[:, None]*delta[None, :]
            + 2*delta[:, None]*(np.outer(m, m)-Cnew)
            - (delta*delta)[:, None]*delta[None, :])


source = base.PREDICT_SOURCE
edits = {
    '        host = m.copy()': '        host = m.copy()\n        dm = np.zeros(n)',
    '        means.append(m.copy())': '''        if hook is not None:
            target = target+shift_plane(host,C,dm)
            centering_p,centering_lam = mean_cubes(host,dm)
            centering_plane = (centering_p**2*centering_lam)@centering_p.T
        else:
            centering_p = np.zeros((n,0))
            centering_lam = np.zeros(0)
            centering_plane = np.zeros((n,n))
        means.append(m.copy())''',
    '        b = (g*bpre+radial*d*vv/6)*alive': '        b = (g*bpre+radial*d*vv/6-dm)*alive',
    '        residual = target-inherited-trace_post': '        residual = target-inherited-trace_post-centering_plane',
    '        cp_post = (inherited+born)*live': '        cp_post = (inherited+born+centering_plane)*live',
    'p2[:, selected].astype(np.float32)], axis=1)': 'p2[:, selected].astype(np.float32),centering_p.astype(np.float32)], axis=1)',
    '-a[selected].astype(np.float32)])': '-a[selected].astype(np.float32),centering_lam.astype(np.float32)])',
}
for old, new in edits.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
env = dict(base.env, mean_cubes=mean_cubes, shift_plane=shift_plane)
exec(source, env)
predict = env['predict']
