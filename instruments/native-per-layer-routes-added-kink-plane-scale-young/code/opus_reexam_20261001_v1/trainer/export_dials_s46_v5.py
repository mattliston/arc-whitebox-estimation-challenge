"""v5 (trainer, 10-12 07:5xZ; engineer 07:4xZ: pld v4 'k4la2' = the same per-layer mechanisms as k4la1): a K4L age-2 strength that VARIES
over the layers (S52's arms fk4la2_1_7 / fk4la2_8_15) goes to the per-layer sidecar as 'k4la2' (absolute 1 + shift, all 16 layers), as k4la1;
a flat one is exported exactly as v4 did (builder lever only, no sidecar key), so v4's files are unchanged byte for byte.
v4 (trainer, 10-12 07:15Z): = v3 with k21n (per-layer) and rsy also left to the port (read_from_pt).
v3 (trainer, 10-12 04:30Z, engineer / lead for the S50 / S51 ports): = v2, except (1) the families k4, ssb (read by the engineer's port straight from
the .pt: k4_from_pt_v1, star_beta_from_pt_v1) and kappa (dropped from the port: its native route failed parity) are left out of the file and out of the
unknown-family check, and listed in physics_dials.json under read_from_pt / dropped; (2) the k31l strengths nkl1 / nkl2 / nkl3 may vary over the
layers (the port reads them per read layer from the .pt); the json keeps layer 1's per-age values as before.
TAG opus-trainer (2026-10-02): export a parameter file whose physics shifts ('pl': one shift per family, the same on every layer)
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
Astra v2: validates all 16 finite shifts before zero filtering, rejects unsupported
per-layer variation across the entire vector, and preserves layer-0 provenance.
usage: python -B export_dials_s46_v5.py <params.pt> <new out dir> [note text]"""
import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from export import NATIVE, export
from otr_common import load_params, load_pl, sha256, read_source, source_nodes

BASE = dict(f4a1=0.6, f4a2=0.6, k4la1=1.0, k4la2=1.0, bsrc=1.0, rc=-0.08, k31=-0.212, iso=-0.094, k21n=-0.028, edge=0.164, ds=1.0)
AS_DIAL = ('bscale', 's_carry', 's_new', 'c0', 'X0', 'X1', 'X2')
src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
arrays, dials = load_params(torch, src)
raw_pl = load_pl(torch, src)
for family, values in raw_pl.items():
    assert isinstance(values, (list, tuple)) and len(values) == 16, '%s: exactly16 shifts required' % family
    assert all(isinstance(value, (int, float)) and math.isfinite(value) for value in values), '%s: finite numeric shifts required' % family
pl = {family: [float(value) for value in values] for family, values in raw_pl.items() if any(values)}
FROM_PT, DROPPED = ('k4', 'ssb', 'k21n', 'rsy'), ('kappa',)      # v3; v4 (engineer 07:2xZ): + k21n (per-layer: base V3 -0.028 + shift) and rsy
aside = {family: pl.pop(family) for family in FROM_PT + DROPPED if family in pl}
arrays = {key: value.numpy().copy() for key, value in arrays.items()}
LAW = ('nfs', 'nfd', 'n6b', 'n6r', 'n6d', 'nkl1', 'nkl2', 'nkl3')
PER_LAYER_NATIVE = ('iso', 'bscale', 'X1')
PER_LAYER_OK = ('ds', 'e2', 'k4la1', 'k4la2', 'kc') + LAW + PER_LAYER_NATIVE      # kc: exported per layer as kappa_C_per_layer
for family, values in pl.items():
    assert family in PER_LAYER_OK or all(x == values[1] for x in values), '%s: unsupported per-layer variation; all16 shifts must match' % family
# The base literal uses layer1; the complete sidecar carries every layer's
# absolute value. Never silently flatten a supported per-layer parameter.
full_per_layer = {'iso': [BASE['iso'] + x for x in pl.get('iso', [0.0] * 16)],
                  'bscale': [dials['bscale'] + x for x in pl.get('bscale', [0.0] * 16)],
                  'X1': [dials['X1'] + x for x in pl.get('X1', [0.0] * 16)],
                  'k4la1': [1.0 + x for x in pl.get('k4la1', [0.0] * 16)]}
assert all(len(v) == 16 for v in full_per_layer.values())
if 'k4la2' in pl and not all(x == pl['k4la2'][1] for x in pl['k4la2']):      # v5: per-layer K4L age 2 -> the sidecar (engineer's pld v4)
    full_per_layer['k4la2'] = [1.0 + x for x in pl['k4la2']]
note = dict(parameters=str(src), candidate=str(out / 'candidate.py'), dials={}, note=sys.argv[3] if len(sys.argv) > 3 else '')
note['read_from_pt'] = {f: aside[f] for f in FROM_PT if f in aside}      # v3: per-layer shifts the engineer's port reads from the .pt
note['dropped'] = {f: aside[f] for f in DROPPED if f in aside}            # v3: not ported (kappa: native route failed parity)
note['k31l_per_layer_shifts'] = {f: pl.get(f, [0.0] * 16) for f in ('nkl1', 'nkl2', 'nkl3')}      # v3: the port takes these per read layer
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
native = {family: ([values[1]] * 16 if family == 'iso' else values)
          for family, values in pl.items() if family in NATIVE}
for family in native:
    note['dials'][family] = dict(base=BASE[family], value=BASE[family] + pl[family][1], where='in the file')
for family in PER_LAYER_NATIVE:
    if family in pl:
        note['dials'][family]['values'] = full_per_layer[family]
        note['dials'][family]['where'] = 'base literal = layer1; all16 absolute values in per_layer_dials.json and this values field; native DLY port required'
levers = {family: BASE[family] + pl.get(family, [0, 0])[1] for family in ('f4a1', 'f4a2')}
flat = lambda family, base: (base + pl[family][1] if all(x == pl[family][1] for x in pl[family]) else [base + x for x in pl[family]]) if family in pl else base
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
    for family in ('n6b', 'n6r', 'n6d', 'nfd'):      # v3: nkl1 / nkl2 / nkl3 may vary per layer (the port reads them from the .pt)
        assert all(x == get(family)[1] for x in get(family)), '%s: unsupported per-layer variation; all16 shifts must match' % family
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
# Verify the actual serialized global knobs and every per-layer reconstruction.
written = source_nodes((out / 'candidate.py').read_text())
knobs = ast.literal_eval(written['_GN_KNOBS'][-1].value)
bases = dict(iso=float(knobs['iso']),
             bscale=float(ast.literal_eval(written['_S19_BSCALE'][-1].value)),
             X1=float(ast.literal_eval(written['_BD_J1_PHYS'][-1].value)),
             k4la1=float(full_per_layer['k4la1'][1]))
if 'k4la2' in full_per_layer:      # v5
    bases['k4la2'] = float(full_per_layer['k4la2'][1])
assert all(bases[name] == values[1] for name, values in full_per_layer.items())
complete = {name: {str(layer): float(value) for layer, value in enumerate(values)}
            for name, values in full_per_layer.items()}
reconstructed = {name: [complete[name].get(str(layer), bases[name]) for layer in range(16)] for name in bases}
assert reconstructed == full_per_layer
back_arrays, back_dials = read_source(np, out / 'candidate.py')
assert all(back_arrays[name].tobytes() == value.tobytes() for name, value in arrays.items())
assert back_dials == dials
sidecar = dict(complete, schema='trainer_pld_s46_v1', source=str(src), bases=bases)
with (out / 'per_layer_dials.json').open('x') as stream:
    json.dump(sidecar, stream, indent=1)
note['per_layer_sidecar'] = dict(path=str(out / 'per_layer_dials.json'),
                                sha256=sha256(out / 'per_layer_dials.json'),
                                schema=sidecar['schema'], canonical_global_bases=bases,
                                reconstruction_all16_exact=True, arrays_roundtrip_exact=True)
with (out / 'physics_dials.json').open('x') as stream:
    json.dump(note, stream, indent=1)
print(json.dumps(note, indent=1))
