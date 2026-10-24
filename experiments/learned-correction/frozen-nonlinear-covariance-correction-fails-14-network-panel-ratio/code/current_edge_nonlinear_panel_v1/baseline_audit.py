"""Independent scalar provenance audit; never decode arrays or targets."""
from common import *
seal_check();meta=read(HERE/'inputs.json');rows=[]
for row in meta['panel']+meta['confirmation']:
    p=row['parent'];verify(p['receipt']);verify(p['source']);receipt=read(p['receipt']['path'])
    assert receipt['candidate_sha256']==p['source']['sha256'] and not receipt['truth_accessed']
    assert len(receipt['records'])==1
    c=receipt['records'][0]
    assert c['rank']==row['rank'] and c['index']==0 and c['includes_initialization'] and c['input_flops_separate']==0
    assert c['output']==p['prediction']['path'] and c['output_sha256']==p['prediction']['sha256']
    assert c['weights_sha256']==p['weights_sha256']==row['weights']['sha256']
    assert c['flops']==p['flops'] and c['multiplier']==p['multiplier']==max(.1,c['flops']/2**41)
    assert c['weight5_cache']['completed'] and c['weight5_cache']['numeric_cache_retired']
    assert p['packing_equivalence_used']==(p['source']['sha256']!=meta['bank']['sha256'])
    rows.append(dict(rank=row['rank'],receipt=p['receipt'],source=p['source'],cold_bill=c['flops'],
        source_weight_output_and_bill_ownership=True,packing_equivalence_used=p['packing_equivalence_used']))
verify(meta['packing_qualification']);packing=read(meta['packing_qualification']['path'])
assert packing['passed'] and packing['cold_flop_delta']==packing['warm_flop_delta']==0
assert packing['candidate']==meta['bank']
assert all(r['source']==(packing['parent'] if r['packing_equivalence_used'] else packing['candidate']) for r in rows)
assert sum(not r['packing_equivalence_used'] for r in rows)==4
write(HERE/'baseline_audit.json',dict(passed=True,rows=rows,inputs=pin(HERE/'inputs.json'),source=pin(Path(__file__)),
    arrays_decoded=False,targets_decoded=False,candidate_or_gates_changed=False))
print(dict(passed=True,network_receipts=len(rows),direct_packed=4,qualified_lossless_equivalent=18),flush=True)
