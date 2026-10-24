"""TAG opus_engineer a54 (2026-10-11, lead 06:07Z): a hand-over's candidate.py against the REAL checkpoint, without torch. Decodes the
file's 28 arrays and 12 dials with the trainer's own reader (otr_common.read_source, read-only) and its _GN_KNOBS literal, reads the
checkpoint with a54_jmiv/read_pt.py, and compares what the exporter must have written: arrays (p.head / p.head_b times 1 + ds per
layer, layer 0 untouched), dials + their uniform shifts, _GN_KNOBS = base + the layer-1 shift. Lists every per-layer family whose
layers 1-14 differ (those must come through DLY@ / KC@ / the K4L lists), with the values a DLY sidecar must carry.
usage: verify_handover.py <candidate.py> <params.pt>"""
import ast
import sys
from pathlib import Path
import numpy as np
E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(E.parent/'trainer'))
from otr_common import DIAL_KEYS, read_source, source_nodes   # torch-free
src = (E/'a54_jmiv/read_pt.py').read_text().split('\na = load(sys.argv[1])')[0]
ns = {}
exec(compile(src, 'read_pt', 'exec'), ns)
cand, pt = Path(sys.argv[1]), sys.argv[2]
arrays_c, dials_c = read_source(np, cand)
P = ns['load'](pt)
arrays = {k: np.asarray(v, dtype=np.float32) for k, v in P['arrays'].items()}
dials = {k: float(v) for k, v in P['dials'].items()}
pl = {k: np.asarray(v, dtype=np.float64) for k, v in P.get('pl', {}).items()}
if 'ds' in pl:
    scale = 1.0 + pl['ds']; scale[0] = 1.0
    arrays['p.head'] = (arrays['p.head'] * scale[:, None]).astype(np.float32)
    arrays['p.head_b'] = (arrays['p.head_b'] * scale).astype(np.float32)
bad = 0
for k in sorted(set(arrays) | set(arrays_c)):
    a, b = arrays.get(k), arrays_c.get(k)
    if a is None or b is None or a.shape != b.shape:
        print('ARRAY MISSING / SHAPE', k); bad += 1; continue
    d = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
    if d:
        print(f'ARRAY {k}: max |file - checkpoint| {d:.3g}'); bad += 1
print(f'arrays: {len(arrays_c)} compared, {bad} differ' + (' (ds folded into p.head / p.head_b per layer as the exporter does)' if 'ds' in pl else ''))
uniform = lambda fam: (fam in pl and np.all(pl[fam][1:15] == pl[fam][1]))
for k in DIAL_KEYS:
    want = dials[k] + (float(pl[k][1]) if k in pl else 0.0)
    got = float(dials_c[k]) if isinstance(dials_c, dict) else float(dials_c[DIAL_KEYS.index(k)])
    flag = '' if abs(got - want) <= 1e-12 * max(1.0, abs(want)) else '   <-- DIFFERS'
    print(f'dial {k:8s} file {got:.12g}  checkpoint dial + shift {want:.12g}{flag}')
nodes = source_nodes(cand.read_text())
knobs = ast.literal_eval(nodes['_GN_KNOBS'][-1].value)
for name, fam, base, layers in (('_GN_EDGE', 'edge', 0.164, range(8, 15)), ('_GN_RC', 'rc', -0.08, range(1, 15))):   # per-layer tables
    table = ast.literal_eval(nodes[name][-1].value) if name in nodes else {}
    worst = max((abs(table.get(l, base) - (base + (float(pl[fam][l]) if fam in pl else 0.0))) for l in layers), default=0.0)
    print(f'{name}: max |file - (base + shift)| over layers {layers.start}-{layers.stop - 1} = {worst:.3g}' + ('' if worst <= 1e-12 else '   <-- DIFFERS'))
BASE = dict(iso=-0.094, k21n=-0.028, k31=-0.212)
for k, base in BASE.items():
    want = base + (float(pl[k][1]) if k in pl else 0.0)
    flag = '' if abs(knobs[k] - want) <= 1e-12 else '   <-- DIFFERS'
    print(f'_GN_KNOBS[{k!r}] file {knobs[k]!r}  base + layer-1 shift {want!r}{flag}')
print('per-layer families (layers 1-14 not all equal): what DLY@ / KC@ / the K4L lists must carry, all 16 layers:')
for fam, v in sorted(pl.items()):
    if not np.all(v[1:15] == v[1]) or (fam == 'ds' and v[15] != v[1]):
        print(f'   {fam}: ' + ' '.join(f'{l}:{x:.9g}' for l, x in enumerate(v)))
