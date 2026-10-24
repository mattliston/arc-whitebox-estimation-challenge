"""v2 (18:3xZ): v1 with the print line fixed and the compile check before the write.
TAG opus-trainer RAD (written 2026-10-12T18:3xZ; lead 18:1xZ): writes a new make-file version = the old one + the ZERO-ARM GUARD
(cert/RAD/zero_arms_v1.py) in the arm stage: every arm whose parameter file equals the start point bit for bit is removed from its arms dir's
arms.json 'live' list before keep_v1.json is written and before any arm is queued; RULES_v1.json records zero_arms_dropped. Nothing else changes.
usage (from O/trainer): python -B cert/RAD/fix_make_zero_v1.py <round, e.g. S53> <old make> <new make>"""
import os
import sys
from pathlib import Path

name, old, new = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
src = old.read_text()
stamp = os.popen('date -u +%FT%TZ').read().strip()
K52 = "    keep = read(DG / 'arms.json')['live']\n"
K53 = "    keep = read(DG / 'arms.json')['live'] + (read(DF / 'arms.json')['live'] if (DF / 'arms.json').exists() else [])\n"
GUARD = ("    sys.path.insert(0, 'cert/RAD')\n"
         "    from zero_arms_v1 import drop_zero_arms      # zero-arm guard (lead 18:1xZ): out of arms.json 'live' before any arm is queued\n"
         "    zero = drop_zero_arms(%s, cfg, '%s')\n")
if src.count(K53) == 1:
    src = src.replace(K53, GUARD % ("[d for d in (DG, DF) if (d / 'arms.json').exists()]", name) + K53)
else:
    assert src.count(K52) == 1, 'no keep line'
    src = src.replace(K52, GUARD % ('[DG]', name) + K52)
R = 'written_before_any_arm_result=True,'
assert src.count(R) == 1, 'no RULES anchor'
src = src.replace(R, R + ' zero_arms_dropped=zero,')
P = "    print('%s arms" % name
assert src.count(P) == 1, 'no print anchor'
src = src.replace(P, "    print('zero arms dropped: %s' % (' '.join(zero) or 'none'))\n" + P)
assert src.startswith('"""')
src = ('"""%s (written %s, lead 18:1xZ): = %s with the ZERO-ARM GUARD (cert/RAD/zero_arms_v1.py, cert/RAD/fix_make_zero_v1.py): an arm '
       'whose parameter file equals the start point bit for bit leaves arms.json \'live\' before keep_v1.json and before any arm is queued; one '
       'TEAM_LOG line when any is dropped. Nothing else changed.\n' % (new.name, stamp, old.name)) + src[3:]
compile(src, str(new), 'exec')      # v2: compile BEFORE writing (v1 wrote a file with a doubled-percent print)
new.open('x').write(src)
print(old, '->', new)
