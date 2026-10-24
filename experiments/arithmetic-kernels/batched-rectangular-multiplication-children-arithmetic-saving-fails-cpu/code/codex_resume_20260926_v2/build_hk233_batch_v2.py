"""Repartition the existing arena; preserve the admitted equations and gates."""
from pathlib import Path
import ast
import hashlib
import json
HERE=Path(__file__).resolve().parent
source=HERE/'candidate_old_both_tail_trace_bounded_v6.py'
text=source.read_text();tree=ast.parse(text)
def latest(name):
    return [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
side=ast.get_source_segment(text,latest('_ipt_side_p'))
side=side.replace('def _ipt_side_p(', 'def _hkb_side_p(')
side=side.replace('(_IPT_BA, _IPT_BB, _IPT_BC)[kind]', '_HKB_CACHES[kind]')
side=side.replace('key = (L, P, rows, cols)', 'key = (L, P, rows, cols, _HKB_REGION)')
side=side.replace('region = _IPT_REGION[L]', 'region = _HKB_REGION')
batch=ast.get_source_segment(text,latest('_bb_product'))
batch=batch.replace('def _bb_product(', 'def _hkb_bb_product(')
batch=batch.replace('region = _IPT_REGION.get(L)', 'region = _HKB_REGION')
batch=batch.replace('_ipt_side_p(', '_hkb_side_p(')
batch=batch.replace('return _AB_OLD_BB(As, Bs, outs, L)', "raise RuntimeError('Rectangular batch did not fit the planned arena')")
old=(HERE/'hk233_batch_fragment_v1.py').read_text()
old=old[:old.index('def _hkb6_many')]
head='''
# Dedicated view caches include the complete arena layout in each key.
# No extra arithmetic arena is allocated; old parent views are left intact.
_HKB_CACHES = (_CapLRU(), _CapLRU(), _CapLRU())
for _hkb_cache in _HKB_CACHES:
    _CAP_BY_ID[id(_hkb_cache)] = 8
_HKB_REGION = None
'''
driver='''
def _hkb6_many(pairs):
    global _HKB_REGION
    m, k, n = int(pairs[0][0].shape[0]), int(pairs[0][0].shape[1]), int(pairs[0][1].shape[1])
    assert m % 64 == k % 64 == n % 64 == 0
    assert all(tuple(a.shape) == (m, k) and tuple(b.shape) == (k, n)
               and str(a.dtype) == str(b.dtype) == 'float32' for a, b in pairs)
    counts = (7**6 * (m//64) * (k//64),
              7**6 * (k//64) * (n//64),
              7**6 * (m//64) * (n//64))
    space = min(sum(_IPT_REGION[6]), int(_SCRATCH[0].shape[0]))
    cap = min(15, space // sum(counts))
    assert cap >= 1 and _AB_ON and _SCRATCH is not None
    _HKB_REGION = tuple(cap*q for q in counts)
    assert sum(_HKB_REGION) <= space
    outs = [fnp.empty((m, n), dtype=fnp.float32) for _ in pairs]
    _TR_LAST[0] = None
    for start in range(0, len(pairs), cap):
        block = pairs[start:start + cap]
        _hkb_bb_product([a for a, b in block], [b for a, b in block], outs[start:start+cap], 6)
    _TR_LAST[0] = None
    return outs
'''
result=old+head+'\n'+side+'\n\n'+batch+'\n\n'+driver
dest=HERE/'hk233_batch_fragment_v2.py'
compile(result,str(dest),'exec')
with dest.open('x') as f:f.write(result)
# The tiny numpy algebra test requires only the coefficient function, before caches.
bench=(HERE/'bench_hk233_batch_v1.py').read_text()
bench=bench.replace('hk233_batch_bench_v1.json','hk233_batch_bench_v2.json').replace('hk233_batch_fragment_v1.py','hk233_batch_fragment_v2.py')
bench=bench.replace('exec(helper.read_text(), testenv)', "exec(helper.read_text().split('# Dedicated view caches')[0], testenv)")
with (HERE/'bench_hk233_batch_v2.py').open('x') as f:f.write(bench)
record=dict(parent_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    helper_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
    status='Arena-repartition research implementation, primitive gates unchanged',
    allocation='No added arithmetic arena; total region capacity <= original depth6 region sum',
    cache='Three dedicated bounded8 LRU caches; full region layout included in key')
with (HERE/'hk233_batch_manifest_v2.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
