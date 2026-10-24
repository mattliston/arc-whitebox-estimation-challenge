"""TAG opus_engineer: the lever string for stack/build.py from a run M parameter file (families of O/trainer/otr_m.py).
usage: python -B levers_from_params.py <params.pt> [--standalone]
Prints  F4v4t<theta age 1>,<theta age 2>+K4Lv3c<c age 1>,<c age 2>[+K22s<scale>]  = otr_m.py's base values (F4_THETA, K4L_C, read
from its source) plus the shifts stored under 'pl' in the parameter file (f4a1, f4a2, k4la1, k4la2, k22).
Each of these families must hold ONE value on all 16 layers (master.py --extra trains global dials and stores them 16 times);
a per-layer table is refused: the native fragments take one strength per age.
Every other family the trainer's export already writes INTO the exported candidate (O/trainer/announce.py + export.py: rc and
edge as tables, bsrc as _S19_BSRC_DIV, iso / k21n / k31 as _GN_KNOBS values, ds folded into the head literals), so no lever is
printed for them here: a lever on top would apply the dial twice. A family outside both lists is refused.
--standalone (for a parent that was NOT exported with those families, e.g. RRPI itself): also print DS<1 + ds> and
GN<knob><value> levers for ds, iso, k21n, k31 (one value on the layers where each acts).
The candidate-3 card (otr_m.py with TRAINER_F4C=1): when any of the exact strengths x1a1, x2a1, x1a2, x2a2 is not 0 the first two
levers become  F4Ct<theta age 1>,<theta age 2>c<x1a1>,<x2a1>,<x1a2>,<x2a2>  (the physicist's mixture in place of F4; its rule
strengths are the same f4a1 / f4a2 leaves) and  K4Lv4c...  (v3's arithmetic, operations independent of the values). A nonzero e2
(the second-order skew term of the mean, one leaf per layer) adds  E2s<value>  or  E2s<15 values, layer 1 first>."""
import re
import sys
from pathlib import Path

import torch

O = Path(__file__).resolve().parents[2]
match = re.search(r'^F4_THETA, F4_AGE, K4L_C, K4L_AGE = ([0-9.]+), 2, ([0-9.]+), 2$', (O/'trainer/otr_m.py').read_text(), re.M)
theta0, c0 = float(match.group(1)), float(match.group(2))
pl = {family: [float(x) for x in values] for family, values in torch.load(sys.argv[1], map_location='cpu', weights_only=True).get('pl', {}).items()}
standalone = '--standalone' in sys.argv[2:]
mine = ('f4a1', 'f4a2', 'k4la1', 'k4la2', 'k22', 'x1a1', 'x2a1', 'x1a2', 'x2a2')
GN = dict(iso=-0.094, k21n=-0.028, k31=-0.212)   # the native _GN_KNOBS literal = the card's V3 (fable_covhunter_v1/gn3_knobs.py)
exported = ('rc', 'edge', 'bsrc', 'ds', 'X1', 'bscale') + tuple(GN)   # X1, bscale: shifts of two of the 12 dials, exported as the dial literals _BD_J1_PHYS, _S19_BSCALE
other = sorted(family for family, values in pl.items() if any(values) and family not in mine + exported + ('e2',))
assert not other, 'no native form for: %s' % other
assert not (standalone and (any(pl.get('X1', [0.0])) or any(pl.get('bscale', [0.0])))), 'X1 / bscale shifts have no lever: export the dial literals'


def num(x):
    text = repr(float(x))
    assert re.fullmatch(r'-?[0-9]+\.[0-9]+', text), 'not a plain decimal: %s' % text   # the builder's lever syntax has no exponent form
    return text


shift = {}
for family in mine:
    values = pl.get(family, [0.0] * 16)
    assert len(values) == 16 and all(x == values[0] for x in values), '%s is not one value for all layers: %r' % (family, values)
    shift[family] = values[0]
exact = [shift[family] for family in ('x1a1', 'x2a1', 'x1a2', 'x2a2')]   # the exact S1, S2 strengths of the candidate-3 card (TRAINER_F4C=1), base 0
if any(exact):   # the physicist's mixture in place of F4 (same rule strengths), with the K4L form whose operations do not depend on the values
    levers = ['F4Ct%s,%sc%s' % (num(theta0 + shift['f4a1']), num(theta0 + shift['f4a2']), ','.join(num(x) for x in exact)),
              'K4Lv4c%s,%s' % (num(c0 + shift['k4la1']), num(c0 + shift['k4la2']))]
else:
    levers = ['F4v4t%s,%s' % (num(theta0 + shift['f4a1']), num(theta0 + shift['f4a2'])), 'K4Lv3c%s,%s' % (num(c0 + shift['k4la1']), num(c0 + shift['k4la2']))]
if shift['k22']:
    levers.append('K22s%s' % num(1.0 + shift['k22']))
e2 = pl.get('e2', [0.0] * 16)   # the second-order skew term of the mean, one strength per layer 1-15 (index 0 is never used)
if any(e2[1:]):
    levers.append('E2s' + (num(e2[1]) if all(x == e2[1] for x in e2[1:]) else ','.join(num(x) for x in e2[1:16])))
if standalone:
    ds = pl.get('ds', [0.0] * 16)
    assert all(x == ds[1] for x in ds[1:]), 'ds is not one value on layers 1-15: %r' % ds
    if ds[1]:
        levers.append('DS%r' % (1.0 + ds[1]))
    for knob, base in GN.items():
        values = pl.get(knob, [0.0] * 16)
        assert len(values) == 16 and all(x == values[2] for x in values[2:15]), '%s is not one value on layers 2-14: %r' % (knob, values)
        if values[2]:
            levers.append('GN%s%r' % (knob, base + values[2]))
else:
    left = sorted(family for family in exported if any(pl.get(family, [0.0])))
    if left:
        print('in the exported candidate already (no lever): %s' % ', '.join(left), file=sys.stderr)
assert all(re.fullmatch(r'[A-Za-z0-9.,-]+', lever) and 'e-' not in lever and 'e+' not in lever for lever in levers), levers   # plain decimals only (no exponent)
print('+'.join(levers))
