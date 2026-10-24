"""TAG opus-trainer (2026-10-02): export a parameter file whose physics shifts ('pl': one shift per family, the same on every layer)
may include families that are really one of the 12 dials or the scale of the learned mean correction:
  ds                -> multiplied into the head / head_b literals of layers 1-15 (the correction is linear in them)
  bscale, s_carry, s_new, c0, X0, X1, X2 -> added to the dial of that name (a shift on all layers is the dial itself)
  rc, edge, bsrc, iso, k21n, k31 -> the exporter's native forms (literals; bsrc = the one changed line)
  f4a1, f4a2, k4la1, k4la2 -> NOT in the file: levers of the engineer's builder; written to physics_dials.json with their values
  x1a1, x1a2, x2a1, x2a2 (exact S1 / S2 strengths of the physicist's exact-term module, per pair age) -> NOT in the file: the
                    engineer's F4c lever; e2 (second-order skew term of the mean formula, per layer) -> NOT in the file: needs
                    its native port; both written to physics_dials.json. k4la1 / k4la2 and e2 may differ between layers
                    (the json then lists all 16 values).
  nfs, nfd, n6b, n6r, n6d (the pair-carried law's dials: fold strength, factor on the removed multiple of C, births / rows / diag
                    source strengths) and nkl1, nkl2, nkl3 (k31l's strengths, one leaf per instance with age limit 1 / 2 / 3)
                    -> NOT in the file: literals of the engineer's pair-law lever; written to physics_dials.json per layer,
                    with the strength per cohort AGE of k31l worked out (age 1 = nkl1 + nkl2 + nkl3, age 2 = nkl2 + nkl3, age 3 = nkl3,
                    each instance's own base added: 0 / 1 / 0 as the dial solves run them).
Writes <out dir>/candidate.py (the RRPI source with those literals) and <out dir>/physics_dials.json.
usage: python -B export_dials.py <params.pt> <new out dir> [note text]"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from export import NATIVE, export
from otr_common import load_params, load_pl, sha256

BASE = dict(f4a1=0.6, f4a2=0.6, k4la1=1.0, k4la2=1.0, bsrc=1.0, rc=-0.08, k31=-0.212, iso=-0.094, k21n=-0.028, edge=0.164, ds=1.0)
AS_DIAL = ('bscale', 's_carry', 's_new', 'c0', 'X0', 'X1', 'X2')
src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
arrays, dials = load_params(torch, src)
pl = {family: values for family, values in load_pl(torch, src).items() if any(values)}
arrays = {key: value.numpy().copy() for key, value in arrays.items()}
LAW = ('nfs', 'nfd', 'n6b', 'n6r', 'n6d', 'nkl1', 'nkl2', 'nkl3')
PER_LAYER_OK = ('e2', 'k4la1', 'k4la2', 'kc') + LAW      # kc: exported per layer as kappa_C_per_layer
for family, values in pl.items():
    assert family in PER_LAYER_OK or all(x == values[1] for x in values[1:15]), '%s: one shift for all layers expected' % family
note = dict(parameters=str(src), candidate=str(out / 'candidate.py'), dials={}, note=sys.argv[3] if len(sys.argv) > 3 else '')
if 'ds' in pl:
    scale = 1.0 + np.asarray(pl['ds'], dtype=np.float64)
    scale[0] = 1.0
    arrays['p.head'] = (arrays['p.head'] * scale[:, None]).astype(np.float32)
    arrays['p.head_b'] = (arrays['p.head_b'] * scale).astype(np.float32)
    note['dials']['ds'] = dict(base=1.0, value=1.0 + pl['ds'][1], where='in the file: head and head_b of layers 1-15 multiplied')
for family in AS_DIAL:
    if family in pl:
        old = dials[family]
        dials[family] = old + pl[family][1]
        note['dials'][family] = dict(base=old, value=dials[family], where='in the file: the dial literal')
native = {family: values for family, values in pl.items() if family in NATIVE}
for family in native:
    note['dials'][family] = dict(base=BASE[family], value=BASE[family] + pl[family][1], where='in the file')
levers = {family: BASE[family] + pl.get(family, [0, 0])[1] for family in ('f4a1', 'f4a2')}
flat = lambda family, base: (base + pl[family][1] if all(x == pl[family][1] for x in pl[family][1:16]) else [base + x for x in pl[family]]) if family in pl else base
levers.update(k4la1=flat('k4la1', 1.0), k4la2=flat('k4la2', 1.0))
exact = {family: pl.get(family, [0, 0])[1] for family in ('x1a1', 'x1a2', 'x2a1', 'x2a2')}
note['builder_levers'] = dict(rule_strength_per_age=[levers['f4a1'], levers['f4a2']], K4L_per_age=[levers['k4la1'], levers['k4la2']],
                              exact_S1_per_age=[exact['x1a1'], exact['x1a2']], exact_S2_per_age=[exact['x2a1'], exact['x2a2']],
                              where='NOT in the file: levers of O/engineer/stack/build.py (age 1 first). With all exact strengths 0 it is '
                                    'F4v4t<rule a1>,<rule a2> + K4Lv3c<a1>,<a2>; otherwise the F4c lever (rule strength and exact S1, S2 per age).')
if 'e2' in pl:
    note['second_order_skew_term_e2_per_layer'] = dict(values=pl['e2'], where='NOT in the file: needs its native port (lead, arm e2 of e16b: '
                                                       'e2[l] x k3^2 (alpha^4 - 6 alpha^2 + 3) pdf / (72 sigma^5) added to the mean through the correction slot, layers 1-15)')
if any(family in pl for family in LAW):
    get = lambda family: [float(x) for x in pl.get(family, [0.0] * 16)]
    kl = {family: get(family) for family in ('nkl1', 'nkl2', 'nkl3')}
    note['pair_law_dials'] = dict(
        fold_strength_per_layer=[1.0 + x for x in get('nfs')], factor_on_removed_multiple_of_C_per_layer=[1.0 + x for x in get('nfd')],
        births_source_strength=1.0 + get('n6b')[1], rows_source_strength=1.0 + get('n6r')[1], diag_source_strength=1.0 + get('n6d')[1],
        k31l_strength_per_cohort_age={'1': kl['nkl1'][1] + 1.0 + kl['nkl2'][1] + kl['nkl3'][1], '2': 1.0 + kl['nkl2'][1] + kl['nkl3'][1], '3': kl['nkl3'][1]},
        where='NOT in the file: literals of the pair-law lever (index = layer 0..15; the fold acts at layers 1-14; source strengths act at the layer of '
              'birth; the factor multiplies the coefficient of tr C x C that the diagonal regression gives, constant regressor form, or the table RHO1). '
              'k31l: fold strength of the fourth-chaos plane for cohorts of that age (age 3 = 0 means ages 1-2 only, as S41 was specified).')
    for family in ('n6b', 'n6r', 'n6d', 'nkl1', 'nkl2', 'nkl3', 'nfd'):
        assert all(x == get(family)[1] for x in get(family)[1:15]), '%s: one value for all layers expected' % family
if 'kc' in pl:      # kappa_C (otr_m comment kc; physicist T74)
    note['kappa_C_per_layer'] = dict(values=[1.0 - float(x) for x in pl['kc']], where='NOT in the file: the share of the learned mean correction d '
                                     'that the step folds into the post covariance handed to the next layer, layers 1-14: C_out = C - kappa (host d\' + d host\' + d d\') '
                                     '(the card as fired: kappa 1); index = layer 0..15')
unknown = set(pl) - set(AS_DIAL) - set(NATIVE) - {'ds', 'f4a1', 'f4a2', 'k4la1', 'k4la2', 'x1a1', 'x1a2', 'x2a1', 'x2a2', 'e2', 'kc'} - set(LAW)
assert not unknown, 'no export form for %s' % sorted(unknown)
out.mkdir(parents=True, exist_ok=False)
changed = export(arrays, dials, out / 'candidate.py', native)
note['changed_statements'] = changed
note['sha256'] = sha256(out / 'candidate.py')
(out / 'physics_dials.json').write_text(json.dumps(note, indent=1))
print(json.dumps(note, indent=1))
