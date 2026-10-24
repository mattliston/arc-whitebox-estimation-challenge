# [opus-joinf64] the live join's (16246 _astra_bcc_push_impl) Nystrom core solve / QR in float64 instead of float32 (S2tr 46e9315d).
# js64 = core = solve(Qn[rows], R') in float64 (cast in, cast core back to float32); jqs64 = also QR of Y in float64 (Qn back to float32).
import hashlib, sys
O = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1'
H = O + '/lead/runs/opus_ladder2h_v1'
src = open(O + '/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py').read()
assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
i0 = src.rfind('\ndef _astra_bcc_push_impl(')
SOLVE = "            core = fnp.linalg.solve(Qn[_jn_rows, :].astype(fnp.float32), R64.T).astype(fnp.float32)\n"
SOLVE64 = "            core = fnp.linalg.solve(Qn[_jn_rows, :].astype(fnp.float64), unused_R.T.astype(fnp.float64)).astype(fnp.float32)   # [opus-joinf64] f64 solve\n"
QR = "        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')\n"
QR64 = "        Qn, unused_R = fnp.linalg.qr(Y.astype(fnp.float64), mode='reduced'); Qn = Qn.astype(fnp.float32)   # [opus-joinf64] f64 QR, f64 R kept for the solve\n"
def edit(s, old, new):
    j = s.find(old, i0)
    assert j > i0 and s.find(old, j + 1) == -1
    return s[:j] + new + s[j + len(old):]
arms = dict(js64=edit(src, SOLVE, SOLVE64), jqs64=edit(edit(src, SOLVE, SOLVE64), QR, QR64))
for tag in (sys.argv[1:] or arms):
    out = arms[tag] + '\n\n# [opus-joinf64] ARM %s: text edit(s) of the live join in _astra_bcc_push_impl (S2tr 46e9315d)\n' % tag
    p = H + '/cand_s2tr_l2h_%s_v1.py' % tag
    open(p, 'w').write(out)
    print(tag, hashlib.sha256(out.encode()).hexdigest()[:8], p)
