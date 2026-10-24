"""Unchanged S31 TRAIN2; metadata-only census for exact read reassociation."""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
source = (HERE/'native_eval.py').read_text()
marker = 'class MLP:\n'
assert source.count(marker) == 1
instrument = '''
_yr_context = [None]
_yr_last = {}
_yr_records = []
_yr_push0 = mod._shared_old384_push
_yr_young0 = mod._push_bank_shared_newborn
_yr_read0 = mod._cp_read_young_sq
def _yr_push(layer, w32, p, previous, newborn, lam, state, rows=None):
    _yr_context[0] = dict(layer=layer, compact=rows is not None)
    _yr_last.clear()
    try:
        return _yr_push0(layer,w32,p,previous,newborn,lam,state,rows=rows)
    finally:
        _yr_context[0] = None
def _yr_young(w,p,previous,newborn):
    result = _yr_young0(w,p,previous,newborn)
    if _yr_context[0] is not None and newborn is not None:
        _yr_last.update(_yr_context[0], previous_live=int(previous.shape[0]),
                        inherited=int(newborn[0]), young_columns=int(result.shape[1]),
                        source_shape=list(p.shape), pushed_rows=int(result.shape[0]))
    return result
def _yr_read(left,young_right,lam,live_indices,state):
    if _yr_last:
        _yr_records.append(dict(rank=rank, **_yr_last, response_rows=int(left.shape[0]),
                                current_live=int(young_right.shape[0]),
                                actual_young_columns=int(young_right.shape[1])))
    return _yr_read0(left,young_right,lam,live_indices,state)
mod._shared_old384_push = _yr_push
mod._push_bank_shared_newborn = _yr_young
mod._cp_read_young_sq = _yr_read
'''
source = source.replace(marker,instrument+'\n'+marker)
source += '''
with (out/'shape_census.json').open('x') as f:
    json.dump(dict(scope='Training metadata only; arithmetic and charges unchanged',records=_yr_records),f,indent=2)
print(json.dumps(dict(shape_records=len(_yr_records))),flush=True)
'''
exec(compile(source,str(HERE/'native_eval.py'),'exec'),globals())
