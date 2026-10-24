"""TAG opus_engineer a52 (2026-10-10 22:0xZ): the per-layer values of a trainer CHECKPOINT (.pt, read without torch by a54_jmiv/read_pt.py)
-> a52_pld/values/<name>.json for the lever DLY@<name>, for the families the trainer's exporter cannot carry per layer (it asserts one
shift for all layers). Value at layer l = today's base + the checkpoint's shift pl[family][l], written for EVERY layer 0-15 once the shifts
vary over layers 1-14 (right whatever value the exported literal holds):
  iso     base -0.094 (otr_pl / export_dials BASE)                      -> _GN_KNOBS['iso'] at layer l
  bscale  base = the checkpoint's dial 'bscale'                        -> _S19_BSCALE at layer l
  X1      base = the checkpoint's dial 'X1'                            -> _BD_J1_PHYS at layer l
k4la1 (the age-1 K4L strength by layer) is taken from the hand-over's physics_dials.json when given (as dly_from_trainer.py does).
usage: dly_from_pt.py <params.pt> <name (new)> [<hand-over physics_dials.json>]"""
import json
import sys
from pathlib import Path
here = Path(__file__).resolve().parent
src = (here.parent/'a54_jmiv/read_pt.py').read_text().split('\na = load(sys.argv[1])')[0]
ns = {}
exec(compile(src, 'read_pt', 'exec'), ns)
pt, name = sys.argv[1], sys.argv[2]
a = ns['load'](pt)
pl = {k: [float(x) for x in v] for k, v in a.get('pl', {}).items()}
dials = {k: float(v) for k, v in a['dials'].items()}
base = dict(iso=-0.094, bscale=dials['bscale'], X1=dials['X1'])
out = {}
for fam in ('iso', 'bscale', 'X1'):
    v = pl.get(fam)
    if not v:
        continue
    if any(x != v[1] for x in v[1:15]):
        out[fam] = {l: base[fam] + v[l] for l in range(16)}   # every layer explicit: right whatever the exported literal holds
if len(sys.argv) > 3:
    k = json.load(open(sys.argv[3])).get('builder_levers', {}).get('K4L_per_age')
    if k and isinstance(k[0], list):
        b = float(k[0][1])
        diff = {l: float(k[0][l]) for l in range(1, 16) if float(k[0][l]) != b}
        if diff:
            out['k4la1'] = diff
others = sorted(f for f, v in pl.items() if f not in ('iso', 'bscale', 'X1', 'kc', 'e2', 'k4la1', 'k4la2', 'nfs', 'nfd', 'n6b', 'n6r', 'n6d', 'nkl1', 'nkl2', 'nkl3')
                and any(x != v[1] for x in v[1:15]))
if others:
    print('WARNING: per-layer shifts this lever does not carry:', others)
path = here/'values'/(name + '.json')
assert not path.exists(), path
json.dump(dict({f: {str(l): x for l, x in v.items()} for f, v in out.items()}, source='%s (dly_from_pt.py)' % pt), open(path, 'w'))
print(' '.join('%s:%s' % (f, ','.join('%d=%r' % (l, x) for l, x in sorted(v.items()))) for f, v in out.items()) or 'none')
