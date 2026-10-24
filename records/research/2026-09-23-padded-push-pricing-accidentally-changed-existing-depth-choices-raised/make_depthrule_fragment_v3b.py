#!/usr/bin/env python3
"""TAG fable-engineer (E-30b, 2026-09-23): S7-lite depth rule = the S6b push5 rule UNCHANGED + padded six-level pushes priced at a
stricter exchange rate. In the v3 fragment one MU prices every push decision, so raising MU to cut the padded moves also demoted
S6b's aligned depth-6 pushes to depth 5 (+0.57% bill per net). Here every push keeps the push5 rule's choice (options 4/5/6 at
_DR_MU = 0.4 Gflop of bill per compute-second), and a 1024-row push whose k or n is not a multiple of 64 switches to the zero-padded
six-level product only if that beats the push5 choice at _DR_MU_PAD.
usage: make_depthrule_fragment_v3b.py <v3 fragment at MU 0.4, e.g. depthrule_fragment_push5p6.py> <mu_pad Gflop/s, e.g. 0.7> <out.py>"""
import sys
from pathlib import Path
src, mu_pad, out = Path(sys.argv[1]).read_text(), float(sys.argv[2]), Path(sys.argv[3])
assert not out.exists(), f"refusing to overwrite {out}"
edits = (
    ("_DR_MU = 400000000.0\n",
     "_DR_MU = 400000000.0\n_DR_MU_PAD = %.1f   # v3b: exchange rate for the zero-padded six-level option only\n" % (mu_pad * 1e9)),
    ("""        if m == 1024 and m % 64 == k % 64 == n % 64 == 0:
            options = (4, 5, 6)
        elif m == 1024:
            options = (4, 5, 7)   # 7 = six levels on the zero-padded product
        else:
            options = (4, 5)
        def value(L):
            if L != 7:
                return _dr_value('push', m, k, n, L, _DR_MU)
            kp, npad = (k + 63) // 64 * 64, (n + 63) // 64 * 64
            return _dr_value('push', m, kp, npad, 6, _DR_MU) + m * k + k * n + _DR_MU * (m * kp + kp * npad) / 2e9
        choice = min(options, key=value)
        choice = 6 if choice == 7 else choice
""",
     """        aligned = m == 1024 and m % 64 == k % 64 == n % 64 == 0
        def value(L, mu):
            if L != 7:   # 7 = six levels on the zero-padded product
                return _dr_value('push', m, k, n, L, mu)
            kp, npad = (k + 63) // 64 * 64, (n + 63) // 64 * 64
            return _dr_value('push', m, kp, npad, 6, mu) + m * k + k * n + mu * (m * kp + kp * npad) / 2e9
        choice = min((4, 5, 6) if aligned else (4, 5), key=lambda L: value(L, _DR_MU))   # the push5 rule's choice (S6b)
        if m == 1024 and not aligned and value(7, _DR_MU_PAD) < value(choice, _DR_MU_PAD):
            choice = 6
"""),
)
for old, new in edits:
    assert src.count(old) == 1, old[:60]
    src = src.replace(old, new)
src = src.replace("# BEGIN FABLE-ENGINEER DEPTH RULE v3 (", "# BEGIN FABLE-ENGINEER DEPTH RULE v3b (push5 decisions + padded D6 at _DR_MU_PAD = %.1f; " % mu_pad, 1)
out.write_text(src)
print(out)
