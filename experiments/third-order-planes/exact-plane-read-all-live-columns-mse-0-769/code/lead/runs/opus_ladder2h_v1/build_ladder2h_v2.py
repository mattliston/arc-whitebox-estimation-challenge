# [opus-ladder2] hunter share, READX partials (READX = read on all columns exact: 8-seed xMSE -0.77% se .85, bill +3.24%):
# readr = RTT2 rank-2 tail OFF only (the RC z2 prefix kept: columns up to z2 exact, zero beyond);
# readc = RC z2 prefix OFF only (RTT2 kept: exact to z1.5, its rank-2 vc/mu tail now spans ALL live columns instead of z1.5..z2 + zeros).
import hashlib, sys
O = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1'
H = O + '/lead/runs/opus_ladder2h_v1'
src = open(O + '/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py').read()
assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
STATS = "_L2H_STATS = dict(ecut=0, j3=0, reads=0, star=0)\n"
READR = STATS + '''_L2H_READ_EXACT = _RTT_PARENT_READ
_RC_PARENT_READ = _L2H_READ_EXACT
'''
READC = STATS + '''_L2H_READ_RTT = _RC_PARENT_READ


def _l2h_read_rtt_all(left, young_right, lam, live_indices, state):
    _RC_STATE[0] = None
    _L2H_STATS['reads'] += 1
    return _L2H_READ_RTT(left, young_right, lam, live_indices, state)


_RS_PARENT_READ = _l2h_read_rtt_all
'''
arms = dict(readr=READR, readc=READC)
for tag in (sys.argv[1:] or arms):
    out = src + '\n\n# BEGIN [opus-ladder2] ARM %s (append LAST to S2tr 46e9315d; approximation switched to exact)\n' % tag + arms[tag] + '# END [opus-ladder2] ARM %s\n' % tag
    p = H + '/cand_s2tr_l2h_%s_v1.py' % tag
    open(p, 'w').write(out)
    print(tag, hashlib.sha256(out.encode()).hexdigest()[:8], p)
