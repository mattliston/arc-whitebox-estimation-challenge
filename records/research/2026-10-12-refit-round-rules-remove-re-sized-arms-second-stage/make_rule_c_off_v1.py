"""TAG opus-trainer RAD (10-12 16:3xZ; lead 16:4xZ / 16:5xZ: radlab's round design from S52 on: rule (c) OFF, escal OUT, phase 2 off): writes the
next version of a round's make file (make_s53_v3.py, make_s54.py, make_s<N>.py from gen_round) with (1) 'escal' added to the dropped gradient
arms and (2) rule (c) switched off (no re-sized strength arms: strength = spec = [] before the fixed-arm build). Rules (a) and (b) unchanged.
usage (from O/trainer): python -B cert/RAD/make_rule_c_off_v1.py <make in> <make out (new file)>"""
import os
import re
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
assert not dst.exists(), dst
s = src.read_text()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:90])
    s = s.replace(old, new)


rep("    drop = ['erest', 'h20', 'h6'] + sorted(a for a, v in last.items() if v >= 0 and a not in ('erest', 'h20', 'h6'))",
    "    drop = ['erest', 'h20', 'h6', 'escal'] + sorted(a for a, v in last.items() if v >= 0 and a not in ('erest', 'h20', 'h6', 'escal'))      # escal OUT (lead 16:5xZ)")
rep("    DF = HOME / 'arms_fixed_v1'\n", "    strength, spec = [], []      # rule (c) OFF (lead 16:5xZ, radlab: the previous round's strength arms are spent)\n    DF = HOME / 'arms_fixed_v1'\n")
k = len(re.findall(r"strength_rule=\"%s's forward arms with single-arm LOO < 0 on %s's sizing networks\" % \(src, src\)", s))
assert k == 1, k
s = re.sub(r"strength_rule=\"%s's forward arms with single-arm LOO < 0 on %s's sizing networks\" % \(src, src\)",
           "strength_rule='OFF (lead 16:5xZ, radlab: re-sized strength arms are spent)'", s)
stamp = os.popen('date -u +%FT%TZ').read().strip()
assert s.startswith('"""')
s = ('"""%s (written %s, lead 16:4xZ / 16:5xZ, radlab, before any of the round\'s data): = %s with rule (c) OFF (no re-sized strength\n'
     'arms) and escal OUT; rules (a) and (b) unchanged; the chain runs phase 1 only unless its PHASE2_ON file exists.\n' % (dst.name, stamp, src.name)) + s[3:]
dst.write_text(s)
print('%s -> %s: rule (c) off, escal out' % (src, dst))
