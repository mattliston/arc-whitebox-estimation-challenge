"""TAG opus-trainer RAD (10-12 16:4xZ; lead 16:5xZ, radlab's acceptance proposal; switched on only after the physicist's stage 2 judges against the
last fired point): writes the next version of a round's make file whose start point = the latest round that CONTINUES: a round continues when a
verdict file says 'accepted' (the -SE rule, as before) or, ONLY when cert/RAD/CONTINUE_RULE reads 'plus1se', when its freeze is not NULL and its
pooled check (the RESERVE / SECOND verdict's pooled numbers if present, else CHECK's frozen-vs-start) is NOT worse than +1 SE. With no
CONTINUE_RULE file, or 'se', the start point is chosen exactly as before. Nothing else changes.
usage (from O/trainer): python -B cert/RAD/make_continue_rule_v1.py <make in> <make out (new file)>"""
import os
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
assert not dst.exists(), dst
s = src.read_text()
old = """def accepted(rnd):
    files = [Path('cert/RAD/%s/%s' % (rnd, f)) for f in ('CHECK_VERDICT_v1.json', 'RESERVE_VERDICT_v1.json', 'SECOND_VERDICT_v1.json')]
    return any(p.exists() and read(p).get('accepted') is True for p in files)
"""
new = """CRULE = (Path('cert/RAD/CONTINUE_RULE').read_text().split() or ['se'])[0] if Path('cert/RAD/CONTINUE_RULE').exists() else 'se'


def accepted(rnd):      # the round CONTINUES: its frozen point may start the next round (make_continue_rule_v1.py)
    files = [Path('cert/RAD/%s/%s' % (rnd, f)) for f in ('CHECK_VERDICT_v1.json', 'RESERVE_VERDICT_v1.json', 'SECOND_VERDICT_v1.json')]
    if any(p.exists() and read(p).get('accepted') is True for p in files):
        return True
    if CRULE != 'plus1se':
        return False
    fp = Path('cert/RAD/%s/FROZEN_POINT_v1.json' % rnd)
    if not fp.exists() or read(fp).get('null'):
        return False
    for name in ('RESERVE_VERDICT_v1.json', 'SECOND_VERDICT_v1.json'):
        p = Path('cert/RAD/%s/%s' % (rnd, name))
        if p.exists():
            q = read(p)['pooled']
            return q['pooled_pct'] <= q['se_pct']
    p = Path('cert/RAD/%s/CHECK_VERDICT_v1.json' % rnd)
    if not p.exists():
        return False
    c = read(p)['pt_vs_ab']
    return c['pooled_pct'] <= c['se_pct']
"""
assert s.count(old) == 1, s.count(old)
s = s.replace(old, new)
stamp = os.popen('date -u +%FT%TZ').read().strip()
assert s.startswith('\"\"\"')
s = ('\"\"\"%s (written %s, lead 16:5xZ / radlab): = %s with the start point = the latest round that CONTINUES (cert/RAD/CONTINUE_RULE:\n'
     'absent or se = accepted by the -SE rule as before; plus1se = also a non-NULL freeze whose pooled check is not worse than +1 SE).\n' % (dst.name, stamp, src.name)) + s[3:]
dst.write_text(s)
print('%s -> %s: start point by CONTINUE_RULE' % (src, dst))
