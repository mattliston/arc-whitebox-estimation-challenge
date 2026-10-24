# [opus-ladder2] hunter share: build value/approximation-off arms on S2tr 46e9315d, each = S2tr + one fragment appended LAST.
# ecutx: covariance edge learned part EXACT (every row x every column, exact tanh); ecutt: exact tanh on the kept block only;
# j3r: the birth update's gate-jet leg j = 3 restored (four-direction parent's form, c[3] + S19 raw X3); readx: the bank read's
# plane on ALL live columns, exact (RC z2 prefix off AND RTT2 rank-2 tail off); starx: the star transport exact, s2 = (W o W)' A' W.
import hashlib, re, sys
O = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1'
H = O + '/lead/runs/opus_ladder2h_v1'
BASE = O + '/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py'
src = open(BASE).read()
assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
lines = src.split('\n')

STATS = "_L2H_STATS = dict(ecut=0, j3=0, reads=0, star=0)\n"

ECUTX = STATS + '''_ECUT_STANDIN = False
_L2H_ECUT_PARENT = _ecut_raw


def _ecut_raw(left, right, t, k, pf, gf, hold):
    rosters = (_E2_LAYER['ecut_rows'], _E2_LAYER['ecut_cols'])
    every = fnp.arange(int(pf.shape[0]))
    _E2_LAYER.update(ecut_rows=every, ecut_cols=every)
    try:
        raw = _L2H_ECUT_PARENT(left, right, t, k, pf, gf, hold)
    finally:
        _E2_LAYER.update(ecut_rows=rosters[0], ecut_cols=rosters[1])
    _L2H_STATS['ecut'] += 1
    return raw
'''

ECUTT = STATS + "_ECUT_STANDIN = False\n"

# j3r: verbatim copy of the live float32 _bd_update (the last definition) with the j = 3 leg added before the diagonal fill
starts = [i for i, l in enumerate(lines) if l.startswith('def _bd_update(')]
i0 = starts[-1]
i1 = next(i for i in range(i0 + 1, len(lines)) if lines[i].startswith('def '))
body = '\n'.join(lines[i0:i1]).rstrip() + '\n'
old_fallback = "        return _BD32_PARENT(bp, Cpre, K21, aux, d, alive, births, a, t, s, coefficients)\n"
assert body.count(old_fallback) == 1
body = body.replace(old_fallback, "        raise RuntimeError('[opus-ladder2] J3R: off the live float32 path')\n")
anchor = "    _e2_hold.append(fnp.subtract(change, _e2_H(bp * _bl_projection), out=change))\n    fnp.fill_diagonal(change, 0.0)\n"
assert body.count(anchor) == 1
leg = '''    _e2_hold.append(fnp.subtract(change, _e2_H(bp * _bl_projection), out=change))
    # [opus-ladder2] J3R: leg j = 3 as the four-direction parent formed it: D3 = Cpre gate_jet_i, gate_jet = skew (alpha^2 - 1) pdf / 6,
    # orthogonalized against bp with coefficient c[3], plus the S19 raw leg X3 D3 (not orthogonalized)
    _l2h_c4 = coefficients.astype(fnp.float64)
    assert int(_l2h_c4.shape[0]) == 4
    _l2h_s = aux['sigma']
    _l2h_jet = _e2_H(_e2_H(_e2_H(aux['pre_k3'] / _e2_H(_e2_H(_l2h_s * _l2h_s) * _l2h_s)) * _e2_H(_e2_H(aux['alpha'] * aux['alpha']) - 1)) * aux['pdf']) / 6
    _l2h_D3 = _e2_H(C32 * _e2_H(_e2_H(_l2h_jet.astype(f32))[:, None]))
    fnp.fill_diagonal(_l2h_D3, 0.0)
    _l2h_p3 = _e2_H(_e2_H(fnp.sum(_e2_H(bp * _l2h_D3), axis=0)) / denominator)
    _l2h_c3 = fnp.asarray(_l2h_c4[3], dtype=fnp.float64)
    _l2h_x3 = float(_S19_BD_ADD[3]) if _S19_BD_ADD is not None and len(_S19_BD_ADD) > 3 else 0.0
    _l2h_e3 = _e2_H(_e2_H(_l2h_c3 + _l2h_x3).astype(f32))
    _l2h_q3 = _e2_H(_e2_H(_l2h_c3.astype(f32)) * _l2h_p3)
    _e2_hold.append(fnp.add(change, _e2_H(_e2_H(_l2h_D3 * _l2h_e3) - _e2_H(bp * _l2h_q3)), out=change))
    _L2H_STATS['j3'] += 1
    fnp.fill_diagonal(change, 0.0)
'''
body = body.replace(anchor, leg)
J3R = STATS + "_L2H_BD_PARENT = _bd_update\n\n\n" + body + '''

# the dense core checks its bookkeeping functions by identity: re-record them with this _bd_update
_ASTRA_DENSE_CONTEXT = tuple((_l2h_n, globals()[_l2h_n]) for _l2h_n, _l2h_f in _ASTRA_DENSE_CONTEXT)
'''

READX = STATS + '''_L2H_READ_EXACT = _RTT_PARENT_READ


def _l2h_read_all(left, young_right, lam, live_indices, state):
    _RC_STATE[0] = None
    _L2H_STATS['reads'] += 1
    return _L2H_READ_EXACT(left, young_right, lam, live_indices, state)


_RS_PARENT_READ = _l2h_read_all
'''

STARX = STATS + '''

def _ss_factors(wt, wt2, A32, hold):
    _L2H_STATS['star'] += 1
    return wt, hold(wt2 @ A32.T)
'''

arms = dict(ecutx=ECUTX, ecutt=ECUTT, j3r=J3R, readx=READX, starx=STARX)
for tag in (sys.argv[1:] or arms):
    frag = arms[tag]
    out = src + '\n\n# BEGIN [opus-ladder2] ARM %s (append LAST to S2tr 46e9315d; approximation switched to exact)\n' % tag + frag + '# END [opus-ladder2] ARM %s\n' % tag
    p = H + '/cand_s2tr_l2h_%s_v1.py' % tag
    open(p, 'w').write(out)
    print(tag, hashlib.sha256(out.encode()).hexdigest()[:8], p)
