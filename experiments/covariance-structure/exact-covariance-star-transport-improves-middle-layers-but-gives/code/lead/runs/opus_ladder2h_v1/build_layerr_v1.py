# [opus-layerr] lead addendum 15:1xZ: LAYER-RESTRICTED versions of the accuracy directions (value-only; the constant becomes per-layer)
# on S2tr 46e9315d. OTT (pos, neg) per layer via a wrapper on the live _shared_old384_push (the OTT recon reads _OTT_Z/_OTT_NEG_Z inside
# the push); _DEADCUT_Z per layer via a wrapper on the live _deadcut_push; _LCR_RU per layer via the UBC parent of _uvp_state.
import hashlib, sys
O = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1'
H = O + '/lead/runs/opus_ladder2h_v1'
src = open(O + '/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py').read()
assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
OTT = '''_L2H_OTT_TABLE = %s
_L2H_OTTW_PARENT = _shared_old384_push


def _shared_old384_push(layer, w32, p, previous, newborn, lam, state, rows=None):
    global _OTT_Z, _OTT_NEG_Z
    saved = (_OTT_Z, _OTT_NEG_Z)
    z = _L2H_OTT_TABLE.get(layer)
    if z is not None:
        _OTT_Z, _OTT_NEG_Z = z
    try:
        return _L2H_OTTW_PARENT(layer, w32, p, previous, newborn, lam, state, rows=rows)
    finally:
        _OTT_Z, _OTT_NEG_Z = saved
'''
DC = '''_L2H_DEADCUT_TABLE = %s
_L2H_DEADCUT_BASE = 3.5
_L2H_DCP_PARENT = _deadcut_push


def _deadcut_push(layer, w32, p, previous, newborn, lam, state, a):
    global _DEADCUT_Z
    saved = _DEADCUT_Z
    _DEADCUT_Z = _L2H_DEADCUT_TABLE.get(layer, _L2H_DEADCUT_BASE)
    try:
        return _L2H_DCP_PARENT(layer, w32, p, previous, newborn, lam, state, a)
    finally:
        _DEADCUT_Z = saved
'''
LCR = '''_L2H_LCR_TABLE = %s
_L2H_LCR_INNER = _UBC_PARENT_STATE


def _l2h_lcr_state(state):
    global _LCR_RU
    saved = _LCR_RU
    _LCR_RU = _L2H_LCR_TABLE.get(_UVP_CUR[0], saved)
    try:
        return _L2H_LCR_INNER(state)
    finally:
        _LCR_RU = saved


_UBC_PARENT_STATE = _l2h_lcr_state
'''
arms = dict(
    ott30_1314=OTT % "{13: (3.0, 3.0), 14: (3.0, 3.0)}",
    ott30_1214=OTT % "{12: (3.0, 3.0), 13: (3.0, 3.0), 14: (3.0, 3.0)}",
    ottp30_1314=OTT % "{13: (3.0, 2.5), 14: (3.0, 2.5)}",
    dc40_1214=DC % "{12: 4.0, 13: 4.0, 14: 4.0}",
    lcr14x=LCR % "{14: 384}",
)
for tag in (sys.argv[1:] or arms):
    out = src + '\n\n# BEGIN [opus-layerr] ARM %s (append LAST to S2tr 46e9315d; per-layer constant, value-only)\n' % tag + arms[tag] + '# END [opus-layerr] ARM %s\n' % tag
    p = H + '/cand_s2tr_l2h_%s_v1.py' % tag
    open(p, 'w').write(out)
    print(tag, hashlib.sha256(out.encode()).hexdigest()[:8])
