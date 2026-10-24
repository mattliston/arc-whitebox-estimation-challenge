"""Strengthen before/after input checks and exercise real block boundaries."""
import hashlib,resource
import numpy as np
from common import *
from identity import fingerprint,BLOCK_BYTES
seal_check();done('analyze')
# The first unit's pair copies were made after its first fingerprint pass.
# Take references BEFORE every call here, including multiple streamed blocks.
rng=np.random.Generator(np.random.PCG64DXSM(2026092185))
base=rng.normal(size=(131,8192)).astype(np.float32)
fixtures=[base,np.asfortranarray(base),base.T,base[::-1,::2],base[1::2,::-1],base[:0],base[:,:0],np.array([[0.,-0.,1.,-1.]],np.float32)]
rows=[]
for i,x in enumerate(fixtures):
    before=x.copy(order='K');encoded=x.tobytes(order='C');expected=hashlib.sha256(encoded).hexdigest()
    was_writeable=x.flags.writeable;x.flags.writeable=False
    try:a=fingerprint(x);b=fingerprint(x.T)
    finally:x.flags.writeable=was_writeable
    assert a['sha256']==expected and a['bytes']==len(encoded)
    assert b['sha256']==hashlib.sha256(before.T.tobytes(order='C')).hexdigest()
    assert x.tobytes(order='C')==before.tobytes(order='C')==encoded
    assert max(a['max_block_bytes'],b['max_block_bytes'])<=BLOCK_BYTES
    rows.append(dict(case=i,shape=list(x.shape),bytes=x.nbytes,read_only_input_pass=True,before_after_bytes_exact=True,
        independent_one_shot_and_transpose_exact=True,multiple_blocks=a['bytes']>BLOCK_BYTES,max_payload_block=max(a['max_block_bytes'],b['max_block_bytes'])))
assert sum(x['multiple_blocks'] for x in rows)>=5
# Check the retained, all-call structure independently of the producer loop.
trace_checks=[]
for rank in (35,100):
    r=read(HERE/f'rank{rank:04d}/trace.json');old=read(OLD/f'rank{rank:04d}/attribution.json')
    assert len(r['calls'])==len(old['outermost_mm_calls'])==295
    for call,ref in zip(r['calls'],old['outermost_mm_calls']):
        assert {k:call[k] for k in ref}==ref
        for key in ('a','b','c'):
            v=call[key];t=call[key+'t'];assert v['shape']==t['shape'][::-1] and v['bytes']==t['bytes'] and v['dtype']==t['dtype']
            assert v['all_zero']==t['all_zero'] and v['finite'] and t['finite']
    assert sum(x['flops'] for x in r['calls'])==old['outermost_mm_sum']==r['outer_mm_flops']
    trace_checks.append(dict(rank=rank,all295_records_exact=True,all885_shape_transpose_metadata_pairs_pass=True))
write(HERE/'fingerprint_audit.json',dict(passed=True,reason='Before-call snapshots and read-only multi-block fixtures strengthen the original preservation check; implementation is unchanged',
    fixtures=rows,trace_checks=trace_checks,fingerprint_source=pin(HERE/'identity.py'),
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,fixtures=len(rows),multi_block_fixtures=sum(x['multiple_blocks'] for x in rows)),flush=True)
