"""TAG opus-trainer RAD (10-12 16:5xZ; lead / radlab 16:5xZ: k4 at L15 has a certified native route): writes the next version of a round's make
file whose rule (b) reads its routes from the data file cert/RAD/ROUTES.json at the arm stage (per_layer / all_only / pending as before, plus
'bands': a family's own band list, e.g. k4 1-7 / 8-11 / 12-14 / 15-15 / all); without the file the built-in tables apply exactly as before.
The strict rule (t^2 > 6, both halves of the same sign at |t| > 1.5) is unchanged. Later route additions are data, not new make files.
usage (from O/trainer): python -B cert/RAD/make_routes_table_v1.py <make in> <make out (new file)>"""
import os
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
assert not dst.exists(), dst
s = src.read_text()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:90])
    s = s.replace(old, new)


rep("ENG = [l for l in Path('../TEAM_LOG.md').read_text().splitlines() if l.startswith('[engineer] ')]\n",
    "ENG = [l for l in Path('../TEAM_LOG.md').read_text().splitlines() if l.startswith('[engineer] ')]\n"
    "ROUTES = read('cert/RAD/ROUTES.json') if Path('cert/RAD/ROUTES.json').exists() else None      # route table (data, make_routes_table_v1.py)\n"
    "if ROUTES:\n"
    "    PER_LAYER, ALL_ONLY, PENDING = tuple(ROUTES['per_layer']), tuple(ROUTES['all_only']), dict(ROUTES['pending'])\n"
    "BANDS = {k: [tuple(b) for b in v] for k, v in (ROUTES or {}).get('bands', {}).items()}\n"
    "DEFAULT_BANDS = [(1, 6), (7, 10), (11, 14), (0, 15)]\n")
rep("def route(fam):\n    if fam in PER_LAYER:", "def route(fam):\n    if fam in BANDS:\n        return 'bands'\n    if fam in PER_LAYER:")
rep("""        for band, (lo, hi) in (('1-6', (1, 6)), ('7-10', (7, 10)), ('11-14', (11, 14)), ('all', (0, 15))):
            if routes[fam] == 'all' and band != 'all':
                continue
""", """        for lo, hi in (BANDS[fam] if routes[fam] == 'bands' else DEFAULT_BANDS if routes[fam] == 'layer' else [(0, 15)]):
""")
rep("                dials.append(fam if band == 'all' else '%s:%d:%d' % (fam, lo, hi))", "                dials.append(fam if (lo, hi) == (0, 15) else '%s:%d:%d' % (fam, lo, hi))")
rep("routes={f: r for f, r in sorted(routes.items()) if r}, dials=dials,", "routes={f: r for f, r in sorted(routes.items()) if r}, route_table=ROUTES, dials=dials,")
stamp = os.popen('date -u +%FT%TZ').read().strip()
assert s.startswith('"""')
s = ('"""%s (written %s, lead / radlab 16:5xZ): = %s with rule (b)\'s routes read from cert/RAD/ROUTES.json at the arm stage (adds\n'
     'per-family band lists: k4 1-7 / 8-11 / 12-14 / 15-15 / all); the strict rule unchanged.\n' % (dst.name, stamp, src.name)) + s[3:]
dst.write_text(s)
print('%s -> %s: routes from ROUTES.json' % (src, dst))
