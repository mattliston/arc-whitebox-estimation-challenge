"""TAG opus_engineer a62 (file written 2026-10-12 16:46Z; lead 16:4xZ: certify the 10-10 DLY routes for per-layer bscale and X1): applies the builder's own
per-layer edits to a built file, the exact strings of stack/build.py (DLY@ with 'bscale' / 'X1' in the values file, lines 929-932):
  bscale: '                b = b * _S19_BSCALE' -> '... b * _pld('bscale', l, _S19_BSCALE)' (1 site: the loop, after the b-fix; = torch b_post)
  X1:     '_BD_J1_PHYS * (1.0 - _j1m)' -> '_pld('X1', _PHYS_LAYER[0], _BD_J1_PHYS) * (1.0 - _j1m)' (4 occurrences, 2 definitions of the birth update)
so the result is what a port of a point with per-layer bscale / X1 builds; then appends a values block setting _PLD['bscale'] / _PLD['X1']
(absolute values by layer; {} = the parent bit for bit). Needs the per-layer dials fragment (fragment_pld_v3) already in the parent.
usage: make_pld_bx_v1.py <parent.py> <out name> [bscale=<layer>:<value>,...] [X1=<layer>:<value>,...]"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
parent, name = Path(sys.argv[1]).resolve(), sys.argv[2]
sets = dict(a.split('=', 1) for a in sys.argv[3:])
out = HERE/'s49'/name
out.mkdir(parents=True)
sha = lambda b: hashlib.sha256(b).hexdigest()
text = parent.read_text()
assert 'def _pld(' in text, 'the parent has no per-layer dials fragment'
EDITS = [("                b = b * _S19_BSCALE\n", "                b = b * _pld('bscale', l, _S19_BSCALE)   # opus-engineer pld: per-layer bscale\n", 1),
         ("_BD_J1_PHYS * (1.0 - _j1m)", "_pld('X1', _PHYS_LAYER[0], _BD_J1_PHYS) * (1.0 - _j1m)", 4)]
for old, new, count in EDITS:
    assert text.count(old) == count, (old, text.count(old), count)
    text = text.replace(old, new)
    print('edited %d x %r' % (count, old.strip()[:40]))
vals = {}
for fam in ('bscale', 'X1'):
    if fam in sets:
        vals[fam] = {int(k): float(v) for k, v in (p.split(':') for p in sets[fam].split(','))}
block = '\n# BEGIN OPUS-ENGINEER a62 PLD BSCALE / X1 VALUES (make_pld_bx_v1.py)\n'
for fam, d in vals.items():
    block += '_PLD[%r] = %r\n' % (fam, d)
block += '_PLD_KNOBS.clear()\n# END OPUS-ENGINEER a62 PLD BSCALE / X1 VALUES\n'
text = text.rstrip('\n') + '\n' + block
(out/'candidate.py').write_text(text)
(out/'build.json').write_text(json.dumps(dict(parent=str(parent), parent_sha256=sha(parent.read_bytes()), values=vals,
                                               candidate=str(out/'candidate.py'), candidate_sha256=sha(text.encode())), indent=1) + '\n')
print(name, sha(text.encode())[:8], vals)
