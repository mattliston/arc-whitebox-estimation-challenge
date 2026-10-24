# [opus-kinkdecomp] variants of the dump-mode oracle (cand_s2tr_kinkoracle_v2.py = S2tr + L14 dump) that switch ONE source of the L13
# post-covariance C13 off (or exact), so each variant's L14 kink-unit variance change D_s can be regressed against the true error.
import hashlib
O = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1'
K = O + '/lead/runs/opus_kinkoracle_v1'
base = open(K + '/cand_s2tr_kinkoracle_v2.py').read()
RERECORD = "_ASTRA_DENSE_CONTEXT = tuple((_vs_n, globals()[_vs_n]) for _vs_n, _vs_f in _ASTRA_DENSE_CONTEXT)\n"
NL = '''import sys as _vs_sys
_VS_NL_PARENT = nonlinear


def nonlinear(mu, C, K21, k4, k22_override=None, metric='identity', extra31=None):
    if _vs_sys._getframe(1).f_locals.get('l') == 13:
        if %s:
            d = fnp.diag(K21).copy()
            K21 = fnp.zeros_like(K21)
            fnp.fill_diagonal(K21, d)
        if %s and k22_override is not None:
            d = fnp.diag(k22_override).copy()
            k22_override = fnp.zeros_like(k22_override)
            fnp.fill_diagonal(k22_override, d)
    if extra31 is None:
        return _VS_NL_PARENT(mu, C, K21, k4, k22_override, metric)
    return _VS_NL_PARENT(mu, C, K21, k4, k22_override, metric, extra31=extra31)


''' + RERECORD
EDGE = '''_VS_EDGE_PARENT = _message_edge_source


def _message_edge_source(layer, m, K21, aux, w, state, P, context):
    out = _VS_EDGE_PARENT(layer, m, K21, aux, w, state, P, context)
    if layer == 13:
        return out * 0.0
    return out
'''
STARX = '''

def _ss_factors(wt, wt2, A32, hold):
    return wt, hold(wt2 @ A32.T)
'''
arms = {
    'k21off13': NL % ('True', 'False'),
    'k22off13': NL % ('False', 'True'),
    'gnoff13': "_GN_COV_LAYERS = tuple(_l for _l in _GN_COV_LAYERS if _l != 13)\n",
    'edgeoff13': EDGE,
    'nojoin': "_JOIN_SCHEDULE = {}\n",
    'starx': STARX,
    'ottall': "_OTT_Z = 1e9\n_OTT_NEG_Z = 1e9\n",
}
for tag, frag in arms.items():
    out = base + '\n# BEGIN [opus-kinkdecomp] VARIANT %s\n' % tag + frag + '# END [opus-kinkdecomp] VARIANT %s\n' % tag
    p = K + '/cand_decomp_%s_v1.py' % tag
    open(p, 'w').write(out)
    print(tag, hashlib.sha256(out.encode()).hexdigest()[:8])
