#!/usr/bin/env python3
"""TAG fable-porter (PT-42, lead GO 22:0x "build C natively on S21"): Nystrom old-tier join. In the LIVE (last-defined) _shared_old384_push only,
the join's test matrix Omega = W[live, :384] is replaced by a row SELECTION E_rows, rows = the 384 live rows with the largest Gram diagonal
d_i = sum_c w_c J_ic^2 (+ (Qp core Qp^T)_ii when an old tier exists), so J^T Omega becomes a gather J[rows].T and Qp^T Omega a gather
Qp[rows].T; QR, Fj, the weighted core and the old-tier update are unchanged. Mirrors join_sketch_test.py arm C (PT-40: VALID16 raw +0.49%
[-0.21, +1.20] vs the current join, torch). Switch _JOIN_NYSTROM (appended last): False (--null) = the S21 op sequence exactly.
Torch mirror for refits: fable_analyst_v1/s2card.py make_push (the join block inside push) must take the same Omega = E_rows.
usage: build_join_nystrom.py <in.py> <out.py> [--null]"""
import sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2]); null = '--null' in sys.argv[3:]
s = src.read_text()
i = s.rfind('\ndef _shared_old384_push(')
assert i >= 0
j = i + 1
while True:   # end of the function: the next line that starts at column 0 and is not blank or a comment continuation inside it
    k = s.find('\n', j) + 1
    if k == 0 or (k < len(s) and s[k] not in ' \n\t' and not s.startswith('def _shared_old384_push(', k)):
        break
    j = k
end = k if k else len(s)
body = s[i:end]
old = ("        Y = _shared_old384_mm(J, _e2_H(_e2_H(weight[:, None]) * _e2_H(_join_rows_mm(_e2_H(J.T), Omega))))\n"
       "        if old_columns:\n"
       "            Y = Y + _e2_H(_shared_old384_mm(Qp, _e2_H(_shared_old384_mm(state['core'], _e2_H(_shared_old384_mm(_e2_H(Qp.T), Omega))))))\n")
assert body.count(old) == 1, body.count(old)
new = ("        if _JOIN_NYSTROM:   # PT-42 Nystrom join: Omega = E_rows (the 384 live rows of largest Gram diagonal); gathers replace J^T Omega, Qp^T Omega\n"
       "            _jn_d = fnp.matmul(_e2_H(J * J), weight)\n"
       "            if old_columns:\n"
       "                _jn_d = _jn_d + fnp.sum(_e2_H(_e2_H(_shared_old384_mm(Qp, state['core'])) * Qp), axis=1)\n"
       "            _jn_rows = fnp.sort(_e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank])\n"
       "            Y = _shared_old384_mm(J, _e2_H(_e2_H(weight[:, None]) * _e2_H(_e2_H(J[_jn_rows, :]).T)))\n"
       "            if old_columns:\n"
       "                Y = Y + _e2_H(_shared_old384_mm(Qp, _e2_H(_shared_old384_mm(state['core'], _e2_H(_e2_H(Qp[_jn_rows, :]).T)))))\n"
       "            del _jn_d, _jn_rows\n"
       "        else:\n"
       + ''.join('    ' + l + '\n' for l in old.rstrip('\n').split('\n')))
body2 = body.replace(old, new)
out = s[:i] + body2 + s[end:]
out = out.rstrip('\n') + ("\n\n\n# PT-42 Nystrom old-tier join switch (False = the S21 join exactly).\n_JOIN_NYSTROM = " + ("False" if null else "True") + "\n")
dst.write_text(out); print(f"wrote {dst} ({'NULL' if null else 'NYSTROM'}); live _shared_old384_push at char {i}, edited once")
