"""Passive current-parent captures; diagnostic copies are not candidate work."""
from pathlib import Path
import hashlib
import json
import sys

STUDY = Path(__file__).resolve().parent
HERE = STUDY.parent
source = (HERE / 'native_eval.py').read_text()
source = source.replace('HERE=Path(__file__).resolve().parent', 'HERE=STUDY.parent')
marker = 'class MLP:\n'
assert source.count(marker) == 1
instrument = '''
_lc_context = {}
_lc_records = []
_lc_original_push = mod._shared_old384_push
_lc_original_recon = mod._slr_reconstruct
def _lc_push(layer, w32, p, previous, newborn, lam, state, rows=None):
    _lc_context.clear()
    if layer in (7, 10, 14):
        _lc_context.update(layer=layer, lam=lam, rows=rows)
    try:
        return _lc_original_push(layer, w32, p, previous, newborn, lam, state, rows=rows)
    finally:
        _lc_context.clear()
def _lc_recon(U, state):
    answer = _lc_original_recon(U, state)
    if _lc_context:
        layer = _lc_context['layer']
        cols = int(state['columns'])
        group = state['lazy']
        payload = dict(U=np.asarray(U), lam=np.asarray(_lc_context['lam'])[:cols],
            cohorts=np.asarray(state['cohorts'], dtype=np.int64),
            local_rows=(np.arange(U.shape[0]) if _lc_context['rows'] is None else np.asarray(_lc_context['rows'])),
            alpha=np.asarray(mod._E2_LAYER['a']), sigma=np.asarray(mod._E2_LAYER['s']),
            gate=np.asarray(mod._E2_LAYER['g']), alive=np.asarray(mod._E2_LAYER['ott_live_mask']),
            tail=np.asarray(mod._OTT_DATA.get('full_tail', []), dtype=np.int64))
        if group is None:
            payload.update(F=np.asarray(state['F']), split=np.asarray(-1))
        else:
            payload.update(base=np.asarray(group['base']), join=np.asarray(group['join']),
                mapping=np.asarray(group['map']), split=np.asarray(int(group['prefix'])))
        path = out / f'cell_{rank:04d}_{layer:02d}.npz'
        assert not path.exists()
        np.savez(path, **payload)
        _lc_records.append(dict(rank=rank, layer=layer, path=str(path), bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        assert sum(x['bytes'] for x in _lc_records) < 500_000_000
        print(json.dumps(dict(event='captured', **_lc_records[-1])), flush=True)
    return answer
mod._shared_old384_push = _lc_push
mod._slr_reconstruct = _lc_recon
'''
source = source.replace(marker, instrument + '\n' + marker)
source += '''
control = HERE / 'both_tail_train2_v6'
base_rows = [json.loads(line) for line in (control/'rows.jsonl').read_text().splitlines()]
new_rows = [json.loads(line) for line in (out/'rows.jsonl').read_text().splitlines()]
assert [x['rank'] for x in base_rows] == [35,100] == [x['rank'] for x in new_rows]
for b,c in zip(base_rows,new_rows):
    assert b['flops']==c['flops'], (b['flops'],c['flops'])
    assert np.array_equal(np.load(control/f"pred_{b['rank']}.npy"), np.load(out/f"pred_{b['rank']}.npy"))
record=dict(records=_lc_records, outputs_and_bills_equal=True, bytes=sum(x['bytes'] for x in _lc_records))
assert len(_lc_records)==6
with (out/'capture_manifest.json').open('x') as f: json.dump(record,f,indent=2)
print(json.dumps(dict(event='capture_qualified', **record)),flush=True)
'''
exec(compile(source, str(HERE / 'native_eval.py'), 'exec'), globals())
