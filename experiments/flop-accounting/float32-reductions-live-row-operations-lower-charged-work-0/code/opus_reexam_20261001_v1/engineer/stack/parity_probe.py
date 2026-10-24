"""TAG opus_engineer stack parity probe: how far the float32 levers T, B, F are from the parent's float64 computations, in situ.
usage: parity_probe.py <rank> <out.json>
Runs RRPI's own prediction (the parent's values are the ones used downstream, so the prediction is RRPI's) and, at every call of
the parent's tail traces, birth update and feature function, evaluates the stack's float32 version on the same inputs and
records the relative difference of the outputs:
  T  _ott_trace / _ynt_trace : repair vector for the rows beyond a row cut
  B  _bd_update              : new base plane (float32 in both) and plane delta
  F  features                : the 30-column per-unit feature matrix of the mean-correction net (per column, worst column)
The extra work is billed inside the prediction's budget; the bill printed is not RRPI's."""
from pathlib import Path
import json
import sys
import types
import numpy as np
import flopscope as flops
fnp = flops.numpy
S = Path(__file__).resolve().parent
E = S.parent
P2 = E.parents[1]
sys.path.insert(0, str(E))
from buildlib import RRPI
import edits_lib
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
rank = int(sys.argv[1]); out = Path(sys.argv[2]); assert not out.exists()
text = RRPI.read_text()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_parity'); mod.__file__ = str(RRPI); sys.modules[mod.__name__] = mod
exec(compile(text, str(RRPI), 'exec', dont_inherit=True), mod.__dict__)
g = mod.__dict__
# the stack's float32 versions, under other names, in the parent's namespace
exec(compile(edits_lib.OTT_NEW.replace('def _ott_trace(', 'def _ott_trace32(').replace("_OTT_DATA.setdefault('blocks', []).append((Ut, F, norm2, tail))", 'pass'), 'ott32', 'exec'), g)
i = text.index('def _ynt_trace(w, p, previous, newborn, coefficient, tail):\n')
exec(compile('def _ynt_trace32(w, p, previous, newborn, coefficient, tail):\n' + edits_lib.YNT_NEW, 'ynt32', 'exec'), g)
exec(compile((E/'a10_bd32/fragment_bd32.py').read_text().replace('def _bd_update(', 'def _bd_update32(').replace('_BD32_PARENT = _bd_update', '_BD32_PARENT = None'), 'bd32', 'exec'), g)
j = text.index('def features(l, m, C, K21, aux, W):\n'); k = text.index('    return fnp.stack(vals, axis=1, dtype=fnp.float32)\n', j)
ftext = text[j:k + len('    return fnp.stack(vals, axis=1, dtype=fnp.float32)\n')]
old, new, _n = edits_lib.features_f32_edits(text)[0]
assert ftext.count(old) == 1
exec(compile(ftext.replace(old, new).replace('def features(', 'def _features32('), 'features32', 'exec'), g)
rows = []


def rel(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    return float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-300))


def wrap(name, probe):
    original = g[name]

    def wrapped(*args):
        result = original(*args)
        probe(result, *args)
        return result
    g[name] = wrapped


def ott(result, U, F, lam, tail):
    rows.append(dict(lever='T', site='_ott_trace', layer=int(mod._UVP_CUR[0]), tail_rows=int(tail.shape[0]), rel=rel(g['_ott_trace32'](U, F, lam, tail), result),
                     size=float(np.linalg.norm(np.asarray(result)))))


def ynt(result, w, p, previous, newborn, coefficient, tail):
    rows.append(dict(lever='T', site='_ynt_trace', layer=int(mod._UVP_CUR[0]), tail_rows=int(tail.shape[0]), rel=rel(g['_ynt_trace32'](w, p, previous, newborn, coefficient, tail), result),
                     size=float(np.linalg.norm(np.asarray(result)))))


def bd(result, bp, *rest):
    new64, delta64 = result
    new32, delta32 = g['_bd_update32'](mod._ps32(bp), *rest)
    rows.append(dict(lever='B', site='_bd_update', layer=len([r for r in rows if r['lever'] == 'B']), rel_base=rel(new32, new64), rel_delta=rel(delta32, delta64),
                     dtype=[str(new32.dtype), str(delta32.dtype)]))


def feat(result, l, m, C, K21, aux, W):
    a = np.asarray(result, dtype=np.float64); b = np.asarray(g['_features32'](l, m, C, K21, aux, W), dtype=np.float64)
    per = np.linalg.norm(a - b, axis=0) / np.maximum(np.linalg.norm(a, axis=0), 1e-300)
    rows.append(dict(lever='F', site='features', layer=int(l), rel=rel(b, a), worst_column=int(np.argmax(per)), worst_column_rel=float(per.max())))


for name, probe in (('_ott_trace', ott), ('_ynt_trace', ynt), ('_bd_update', bd), ('features', feat)):
    wrap(name, probe)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
manifest = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
weights = np.load(STORE/Path(manifest[rank]['path']).parent/'weights.npy', allow_pickle=False)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[weights[l] for l in range(16)], seed=0, name=f'phase2-{rank}')
with flops.BudgetContext(flop_budget=2**41, quiet=True):
    y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
summary = {}
for lever, key in (('T', 'rel'), ('B', 'rel_base'), ('B', 'rel_delta'), ('F', 'rel'), ('F', 'worst_column_rel')):
    v = [r[key] for r in rows if r['lever'] == lever]
    summary[f'{lever} {key}'] = dict(calls=len(v), max=max(v) if v else None, median=float(np.median(v)) if v else None)
out.write_text(json.dumps(dict(rank=rank, summary=summary, rows=rows)))
for k, v in summary.items():
    print(k, v, flush=True)
