# [opus-ladder2] DEEPER / cheaper direction in the hunter's modules (lead 14:1xZ), value-only arms on S2tr 46e9315d:
# edge0  = the learned covariance edge (L8-14) OFF: _EDGE_ENABLED = False (ECUTX showed the edge's outer block carries nothing);
# rt125l = RTT2 exact young-read prefix cutoff 1.5 -> 1.25 sigma at L7-14 (rank-2 tail takes the band; READR: exact z1.5..2 = nothing);
#          text edit of the LIVE (last) _rtt_read floor line, table _L2H_RTTF, as opus-early's RT10 did at L1-6;
# hz7g0  = _HZ7_GAIN 0.03 -> 0.0 (seven-level strip routing whenever the fitted bill model says cheaper; rounding-level).
import hashlib, sys
O = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1'
H = O + '/lead/runs/opus_ladder2h_v1'
src = open(O + '/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py').read()
assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')

def append(tag, frag, base=src):
    return base + '\n\n# BEGIN [opus-ladder2] ARM %s (append LAST to S2tr 46e9315d; deeper / cheaper, value-only)\n' % tag + frag + '# END [opus-ladder2] ARM %s\n' % tag

def rtt(table):
    old = "    floor=.5*_rc_math.erfc(1.5/_rc_math.sqrt(2.0))\n"
    assert src.count(old) == 3
    i = src.rfind(old)
    assert src.rfind('\ndef _rtt_read(') < i and src.find('\ndef _rtt_read(', i) == -1   # inside the LAST _rtt_read
    new = "    floor=_L2H_RTTF.get(layer,.5*_rc_math.erfc(1.5/_rc_math.sqrt(2.0)))   # [opus-ladder2] per-layer exact-prefix cutoff\n"
    edited = src[:i] + new + src[i + len(old):]
    return edited, "_L2H_RTTF = %s\n" % table

arms = {}
arms['edge0'] = append('edge0', "_EDGE_ENABLED = False\n")
e, t = rtt("{l: .5*_rc_math.erfc(1.25/_rc_math.sqrt(2.0)) for l in range(7, 15)}")
arms['rt125l'] = append('rt125l', t, e)
arms['hz7g0'] = append('hz7g0', "_HZ7_GAIN = 0.0\n")
for tag in (sys.argv[1:] or arms):
    p = H + '/cand_s2tr_l2h_%s_v1.py' % tag
    open(p, 'w').write(arms[tag])
    print(tag, hashlib.sha256(arms[tag].encode()).hexdigest()[:8], p)
