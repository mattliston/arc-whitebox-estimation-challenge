"""TRAIN-only, unchanged-estimator matrix capture for offline spectrum diagnosis.

Instrumentation uses local NumPy copies and disk writes. Neither metered cost nor
CPU time from this instrumented run is candidate qualification evidence. Predictions
must match the frozen S30 control. No ground truth enters the instrumented estimator.
"""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
source = (HERE / 'native_eval.py').read_text()
marker = 'class MLP:\n'
assert source.count(marker) == 1
instrument = '''
_nr_layers = {3, 7, 11, 14}
_nr_context = {}
_nr_records = []
_nr_original_cov = mod._flms_cov_star
_nr_original_push = mod._push_bank_shared_newborn
_nr_original_shared = mod._shared_old384_push
def _nr_cov(w, C, A):
    _nr_context.clear()
    # Outside the candidate: read-only snapshots for a separately labelled probe.
    _nr_context['C'] = np.asarray(C, dtype=np.float32).copy()
    _nr_context['w'] = np.asarray(w, dtype=np.float32).copy()
    return _nr_original_cov(w, C, A)
def _nr_shared(layer, w32, p, previous, newborn, lam, state, rows=None):
    _nr_context['layer'] = layer
    try:
        return _nr_original_shared(layer, w32, p, previous, newborn, lam, state, rows=rows)
    finally:
        _nr_context.pop('layer', None)
def _nr_push(w, p, previous, newborn):
    layer = _nr_context.get('layer')
    if layer in _nr_layers and newborn is not None:
        old_columns, selected, base, t, s = newborn
        path = out / f'newborn_rank{rank:04d}_layer{layer:02d}.npz'
        assert not path.exists()
        np.savez(path, C=_nr_context['C'], w=_nr_context['w'],
                 base=np.asarray(base, dtype=np.float32),
                 selected=np.asarray(selected, dtype=np.int64),
                 previous=np.asarray(previous, dtype=np.int64),
                 t=np.asarray(t, dtype=np.float32), s=np.asarray(s, dtype=np.float32))
        _nr_records.append(dict(rank=rank, layer=layer, path=path.name, bytes=path.stat().st_size))
    return _nr_original_push(w, p, previous, newborn)
mod._flms_cov_star = _nr_cov
mod._shared_old384_push = _nr_shared
mod._push_bank_shared_newborn = _nr_push
'''
source = source.replace(marker, instrument + '\n' + marker)
source += '''
with (out / 'capture_manifest.json').open('x') as f:
    json.dump(dict(purpose='TRAIN-only unchanged-estimator offline matrix spectrum probe',
                   cpu_and_metering_are_not_qualification=True, records=_nr_records), f, indent=2)
print(json.dumps(dict(captures=len(_nr_records), bytes=sum(x['bytes'] for x in _nr_records))), flush=True)
'''
exec(compile(source, str(HERE / 'native_eval.py'), 'exec'), globals())
