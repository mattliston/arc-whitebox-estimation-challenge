"""TAG opus_engineer a52: the per-layer values of the trainer's global dials in a hand-over's physics_dials.json -> a52_pld/values/<name>.json
for the lever DLY@<name>: bscale / X1 / iso from dials[name] when it holds a per-layer list ('value' or 'values' as a list of 16, index =
layer) or 'value_10_14'; k4la1 from builder_levers['K4L_per_age'][0] when that is a list of 16 (the age-1 strength by the layer the term is
applied). Only layers whose value differs from the global one (layer 1's) are written. Prints the layers per dial, or 'none'.
usage: dly_from_trainer.py <physics_dials.json> <name>"""
import json
import sys
from pathlib import Path
d = json.load(open(sys.argv[1]))
out = {}
for name in ('bscale', 'X1', 'iso'):
    e = d.get('dials', {}).get(name)
    if not isinstance(e, dict):
        continue
    values = e.get('values', e.get('value'))
    if isinstance(values, list):
        assert len(values) == 16, (name, len(values))
        base = float(values[1])
        diff = {l: float(values[l]) for l in range(1, 15) if float(values[l]) != base}
        if diff:
            out[name] = diff
    elif 'value_10_14' in e:
        out[name] = {l: float(e['value_10_14']) for l in range(10, 15)}
k = d.get('builder_levers', {}).get('K4L_per_age')
if k and isinstance(k[0], list):
    assert len(k[0]) == 16
    base = float(k[0][1])
    diff = {l: float(k[0][l]) for l in range(1, 16) if float(k[0][l]) != base}
    if diff:
        out['k4la1'] = diff
if not out:
    print('none')
    sys.exit(0)
path = Path(__file__).resolve().parent/'values'/(sys.argv[2] + '.json')
json.dump(dict({n: {str(l): v for l, v in vals.items()} for n, vals in out.items()}, source=sys.argv[1]), open(path, 'w'))
print(' '.join('%s:%s' % (n, ','.join(str(l) for l in sorted(v))) for n, v in out.items()))
