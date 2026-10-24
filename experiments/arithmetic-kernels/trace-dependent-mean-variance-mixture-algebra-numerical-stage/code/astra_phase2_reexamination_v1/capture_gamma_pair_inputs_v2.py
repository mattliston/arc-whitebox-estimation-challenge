"""Capture actual call arguments and certify the archived rounded boundary."""
from pathlib import Path
import numpy as np
import radial_compact_feed_fullpair_v1 as parent

HERE = Path(__file__).resolve().parent


def capture():
    current = np.load(HERE/'current_pair_oracle_attribution_v2/inputs_rank_0035.npz')
    W = np.load(HERE/'rows_current_transport_shadow_v2/row_0035/weights.npy')
    # Fixed HISTORY inputs of the established coherent diagnostic. Not
    # current-output targets or fitted sources; no score computed here.
    history = np.load(HERE.parent/'public_g19_attribution_v1/mps_cov2_rank0035.npz')
    mt = history['mc_post'].astype(np.float64)
    ct = history['mc_post_cov'].astype(np.float64)
    result = {k: [] for k in ('mu', 'Cpre', 'K21', 'k4', 'k22', 'host')}
    last = None
    original = parent.env['nonlinear']

    class Done(Exception):
        pass

    def nonlinear(*args):
        nonlocal last
        last = args
        return original(*args)

    def hook(l, m, C, K, aux, w):
        if 8 <= l <= 14:
            i = l-8
            rounded = dict(mu=aux['alpha']*aux['sigma'], Cpre=aux['pre_cov'],
                           K21=K, k4=aux['pre_k4'], host=m)
            for key, value in rounded.items():
                assert np.array_equal(value, current[key][i]), (l, key)
            for key, value in zip(('mu', 'Cpre', 'K21', 'k4', 'k22'), last[:5]):
                result[key].append(value.copy())
            result['host'].append(m.copy())
        if l == 14:
            raise Done
        return mt[l]-m if l >= 1 else np.zeros_like(m)

    def cov_hook(l, C):
        return ct[l].copy()

    source = parent.PREDICT_SOURCE
    for before, after in (
        ('gate_floor=.001):', 'gate_floor=.001, cov_hook=None):'),
        ('            m = m+dm', '            m = m+dm\n        if cov_hook is not None:\n            C = cov_hook(l,C)'),
        ('        radial = radial_new', '        radial = radial_new\n        if cov_hook is not None:\n            C = cov_hook(l,C)')):
        assert source.count(before) == 1
        source = source.replace(before, after)
    env = dict(parent.env, nonlinear=nonlinear)
    exec(source, env)
    try:
        env['predict'](W, hook=hook, cov_hook=cov_hook, gate_floor=.01,
                       cov32=True, metric='covariance')
    except Done:
        pass
    assert len(result['host']) == 7
    return {key: np.stack(values) for key, values in result.items()}
