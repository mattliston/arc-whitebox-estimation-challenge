"""TAG opus_engineer a53 (2026-10-10, lead 20:25Z approximation audit): one native approximation of a parent file switched to exact.
Copies the parent, applies the text edits (each must match exactly the stated number of times), appends literal overrides LAST
(the same as the builder's OVR@), writes a53_audit/files/<parent sha8>_<name>.py and prints its sha.
usage: make_variant.py <parent.py> <name>   (the variant's edits are the entry <name> of VARIANTS below)"""
import hashlib
import sys
from pathlib import Path
L15K_OLD1 = "    rows, unused_mask = select(_L15_ROWCUT_Z - z, 0.0)\n"
L15K_NEW1 = ("    rows35, unused_mask = select(_L15_ROWCUT_Z - z, 0.0)   # a53 l15k: k3 on every row, K4L and the head columns on the cut's rows\n"
             "    rows, unused_mask = select(99.0 - z, 0.0)\n"
             "    m35 = int(rows35.shape[0])\n"
             "    if not bool(fnp.all(rows[:m35] == rows35)):\n"
             "        raise RuntimeError('a53 l15k: row orders differ')\n")
L15K_OLD2 = "    _K4L_FINAL[0] = (rows, _k4l_local(layer, pp_rows, None, None, state))"
L15K_NEW2 = "    _K4L_FINAL[0] = (rows35, _k4l_local(layer, pp_rows[:m35], None, None, state))"
ECR = ("            _E2_LAYER.update(ecut_rows=response_indices, ecut_cols=live_indices)",
       "            _E2_LAYER.update(ecut_rows=select(response_score, _ECUT_RF)[0] if _ECUT_RF > gate_floor else response_indices, ecut_cols=select(response_gate, _ECUT_CF)[0] if _ECUT_CF > gate_floor else live_indices)   # a53: stricter edge rosters", 1)
VARIANTS = {
    'l15': dict(literals={'_L15_ROWCUT_Z': 99.0}),
    'l15k': dict(replace=[(L15K_OLD1, L15K_NEW1, 1), (L15K_OLD2, L15K_NEW2, 1)]),
    'dead': dict(literals={'_DEADCUT_Z': 99.0}),
    'ott': dict(literals={'_OTT_Z': 99.0, '_OTT_NEG_Z': 99.0}),
    'ynt': dict(replace=[("    negative = (ctx['a'] < -2.5) & (~ctx['live'])\n", "    negative = (ctx['a'] < -99.0) & (~ctx['live'])\n", 1),
                         ("    score = fnp.where(negative, ctx['a'] + 2.5, 0.0)\n", "    score = fnp.where(negative, ctx['a'] + 99.0, 0.0)\n", 1)]),
    'std': dict(literals={'_ECUT_STANDIN': False}),
    'uvp': dict(literals={'_LCR_RU': 100000, '_UVP_RU': 100000},
                replace=[("    ru = _LCR_RU if _UVP_CUR[0] in _LCR_LAYERS else 256\n", "    ru = _LCR_RU if _UVP_CUR[0] in _LCR_LAYERS else _LCR_RU\n", 1)]),
    'dcv': dict(literals={'_DCV_FLOOR': 0.0}),
    'oyu': dict(literals={'_OTT_Z': 99.0, '_OTT_NEG_Z': 99.0, '_LCR_RU': 100000, '_UVP_RU': 100000},
                replace=[("    negative = (ctx['a'] < -2.5) & (~ctx['live'])\n", "    negative = (ctx['a'] < -99.0) & (~ctx['live'])\n", 1),
                         ("    score = fnp.where(negative, ctx['a'] + 2.5, 0.0)\n", "    score = fnp.where(negative, ctx['a'] + 99.0, 0.0)\n", 1),
                         ("    ru = _LCR_RU if _UVP_CUR[0] in _LCR_LAYERS else 256\n", "    ru = _LCR_RU if _UVP_CUR[0] in _LCR_LAYERS else _LCR_RU\n", 1)]),
    'allx': dict(literals={'_OTT_Z': 99.0, '_OTT_NEG_Z': 99.0, '_LCR_RU': 100000, '_UVP_RU': 100000, '_DEADCUT_Z': 99.0, '_DCV_FLOOR': 0.0},
                 replace=[("    negative = (ctx['a'] < -2.5) & (~ctx['live'])\n", "    negative = (ctx['a'] < -99.0) & (~ctx['live'])\n", 1),
                          ("    score = fnp.where(negative, ctx['a'] + 2.5, 0.0)\n", "    score = fnp.where(negative, ctx['a'] + 99.0, 0.0)\n", 1),
                          ("    ru = _LCR_RU if _UVP_CUR[0] in _LCR_LAYERS else 256\n", "    ru = _LCR_RU if _UVP_CUR[0] in _LCR_LAYERS else _LCR_RU\n", 1)]),
    'ec05': dict(new=('_ECUT_RF', '_ECUT_CF'), literals={'_ECUT_RF': 0.05, '_ECUT_CF': 0.05}, replace=[ECR]),
    'ec10': dict(new=('_ECUT_RF', '_ECUT_CF'), literals={'_ECUT_RF': 0.10, '_ECUT_CF': 0.10}, replace=[ECR]),
    'ecr10': dict(new=('_ECUT_RF', '_ECUT_CF'), literals={'_ECUT_RF': 0.10, '_ECUT_CF': 0.0}, replace=[ECR]),
    'dz35': dict(literals={'_DEADCUT_Z': 3.5}),   # S46 without its push row cut at gate 1e-3 (OVR@dz309 set 3.09)
    'ott234': dict(literals={'_OTT_Z': 2.34, '_OTT_NEG_Z': 2.34}),
    'ottn234': dict(literals={'_OTT_NEG_Z': 2.34}),
    'ottp234': dict(literals={'_OTT_Z': 2.34}),
    'ottp': dict(literals={'_OTT_Z': 99.0}),
    'ottn': dict(literals={'_OTT_NEG_Z': 99.0}),
    'ottp20': dict(literals={'_OTT_Z': 2.0}),
    'ott20': dict(literals={'_OTT_Z': 2.0, '_OTT_NEG_Z': 2.0}),
    'uvp15': dict(replace=[("    ru = _LCR_RU if _UVP_CUR[0] in _LCR_LAYERS else 256\n", "    ru = _LCR_RU if _UVP_CUR[0] in _LCR_LAYERS else 100000\n", 1)]),
    'uvp1014': dict(literals={'_LCR_RU': 100000}),
    'uvp320': dict(literals={'_LCR_RU': 320, '_UVP_RU': 320},
                   replace=[("    ru = _LCR_RU if _UVP_CUR[0] in _LCR_LAYERS else 256\n", "    ru = _LCR_RU if _UVP_CUR[0] in _LCR_LAYERS else _LCR_RU\n", 1)]),
}
parent, name = Path(sys.argv[1]), sys.argv[2]
v = VARIANTS[name]
text = parent.read_text()
for old, new, count in v.get('replace', []):
    assert text.count(old) == count, (name, old, text.count(old))
    text = text.replace(old, new)
lit = v.get('literals', {})
for k in lit:
    assert ('\n%s = ' % k) in text or k in v.get('new', ()), k
if lit:
    text += '\n\n# a53 audit variant %r: literal overrides (exact values)\n' % name + ''.join('%s = %r\n' % kv for kv in lit.items())
psha = hashlib.sha256(parent.read_bytes()).hexdigest()[:8]
out = Path(__file__).resolve().parent / 'files' / f'{psha}_{name}.py'
assert not out.exists(), out
out.write_text(text)
print(hashlib.sha256(out.read_bytes()).hexdigest()[:8], out)
