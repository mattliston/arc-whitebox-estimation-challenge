"""Byte-exact replay of BOTH sides of the saved precision/support boundary."""
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
import radial_compact_feed_fullpair_v1 as parent

HERE = Path(__file__).resolve().parent


def qualify(current, frozen):
    W = np.load(HERE / 'rows_current_transport_shadow_v2/row_0035/weights.npy')
    t = np.load(HERE.parent / 'public_g19_attribution_v1/mps_cov2_rank0035.npz')
    mt = t['mc_post'].astype(np.float64)
    ct = t['mc_post_cov'].astype(np.float64)
    checked = {}

    class Done(Exception):
        pass

    def hook(l, m, C, K21, aux, w):
        if l == 13:
            for name, v in (('mu', aux['alpha']*aux['sigma']), ('Cpre', aux['pre_cov']),
                            ('K21', K21), ('host', m)):
                assert np.array_equal(v, current[name][5]), name
            checked['consumer_state_byte_exact'] = True
            raise Done
        return mt[l]-m if l >= 1 else np.zeros_like(m)

    def cov_hook(l, C):
        return ct[l].copy()

    def post_probe(l, v):
        if l != 12:
            return
        p = v['p'].astype(np.float64)
        lam = v['lam'].astype(np.float64)
        star = v['star'].astype(np.float64)
        b, B = v['b'], v['Bpost']
        for a, u in enumerate(frozen['U'].T):
            M = (p*((p.T@u)*lam))@p.T
            M += (star*u[None, :]+u[:, None]*star.T+np.diag(star.T@u))/3
            bu = B@u
            M += (b@u)*B+np.outer(b, bu)+np.outer(bu, b)
            assert np.array_equal(M, frozen['model_slices'][a]), ('slice', a)
        wn = W[13].astype(np.float64)
        indices = frozen['indices']
        pp = wn.T@p
        K = (pp[indices]**2*lam)@pp.T
        S = wn.T@star
        V = wn.T
        K += ((2*S[indices]*V[indices])@V.T+(V[indices]**2)@S.T)/3
        Bn = wn.T@B@wn
        bn = wn.T@b
        K += np.diag(Bn)[indices, None]*bn[None, :]+2*Bn[indices]*bn[indices, None]
        assert np.array_equal(K, frozen['model_next_K21'])
        checked['physical_slices_and_plane_byte_exact'] = True

    source = parent.PREDICT_SOURCE
    for a, b in (
        ('gate_floor=.001):', 'gate_floor=.001, cov_hook=None):'),
        ('            m = m+dm', '            m = m+dm\n        if cov_hook is not None:\n            C = cov_hook(l,C)'),
        ('        radial = radial_new', '        radial = radial_new\n        if cov_hook is not None:\n            C = cov_hook(l,C)'),
        ('        previous_alive = alive', '        post_probe(l,locals())\n        previous_alive = alive')):
        assert source.count(a) == 1
        source = source.replace(a, b)
    env = dict(parent.env, post_probe=post_probe)
    exec(source, env)
    with threadpool_limits(limits=4):
        try:
            env['predict'](W, hook=hook, cov_hook=cov_hook, gate_floor=.01,
                           cov32=True, metric='covariance')
        except Done:
            pass
    assert all(checked.values()) and len(checked) == 2
    return checked
