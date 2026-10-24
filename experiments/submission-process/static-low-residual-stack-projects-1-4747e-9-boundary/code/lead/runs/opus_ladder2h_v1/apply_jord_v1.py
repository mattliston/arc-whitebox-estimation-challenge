"""[opus-layerr] JORD fragment applier (value-only text edit, zero bill): energy-ordered Nystrom join rows in the LIVE join.
In the LAST `def _astra_bcc_push_impl(` of the input file, replace the one line
    _jn_rows = fnp.sort(_e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank])
by
    _jn_rows = _e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank]   # JORD
(same row SET; only the order changes, so the QR frame's leading coordinates carry the most energy, as the UVP header intends;
the u-leg keeps pair means in the leading r_u coordinates). 24 seeds on S2tr: xnet -0.236% (SE .19), aggregate -0.233%, bill 0.000.
usage: apply_jord_v1.py <in.py> <out.py>   (asserts exactly one occurrence inside the last _astra_bcc_push_impl; prints shas)"""
import hashlib, sys
src = open(sys.argv[1]).read()
OLD = "            _jn_rows = fnp.sort(_e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank])\n"
NEW = "            _jn_rows = _e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank]   # JORD: energy order (UVP leading r_u coordinates)\n"
i0 = src.rfind('\ndef _astra_bcc_push_impl(')
assert i0 > 0, 'no _astra_bcc_push_impl'
i1 = src.find('\ndef ', i0 + 1)
body = src[i0:i1 if i1 > 0 else len(src)]
assert body.count(OLD) == 1, 'expected exactly one sorted _jn_rows line in the live join, found %d' % body.count(OLD)
j = i0 + body.index(OLD)
out = src[:j] + NEW + src[j + len(OLD):]
open(sys.argv[2], 'x').write(out)
print('in', hashlib.sha256(src.encode()).hexdigest()[:8], 'out', hashlib.sha256(out.encode()).hexdigest()[:8])
