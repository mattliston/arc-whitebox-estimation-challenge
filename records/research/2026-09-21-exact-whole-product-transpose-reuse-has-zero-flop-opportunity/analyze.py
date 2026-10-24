import resource
from common import *
from identity import group_records,signature
seal_check();done('census35');done('census100');inputs=read(HERE/'traces_sealed.json')
for p in inputs['pins']:verify(p)
results=[]
for rank in (35,100):
    trace=read(HERE/f'rank{rank:04d}/trace.json');rows=trace['calls'];exact=group_records(rows);extended=group_records(rows,True)
    # Independent exhaustive pairwise tuples; no group_records/key calls.
    independently_removed=set();independent_duplicates=[]
    for i,row in enumerate(rows):
        if row['a']['all_zero'] or row['b']['all_zero']:independently_removed.add(i)
        A=(tuple(row['a']['shape']),row['a']['dtype'],row['a']['sha256'])
        B=(tuple(row['b']['shape']),row['b']['dtype'],row['b']['sha256'])
        for j,old in enumerate(rows[:i]):
            same=A==(tuple(old['a']['shape']),old['a']['dtype'],old['a']['sha256']) and B==(tuple(old['b']['shape']),old['b']['dtype'],old['b']['sha256'])
            transpose=A==(tuple(old['bt']['shape']),old['bt']['dtype'],old['bt']['sha256']) and B==(tuple(old['at']['shape']),old['at']['dtype'],old['at']['sha256'])
            if same or transpose:independently_removed.add(i);independent_duplicates.append(i);break
    assert sorted(independently_removed)==extended['removed_indices'] and independent_duplicates==extended['duplicate_indices']
    assert sum(rows[i]['flops'] for i in sorted(independently_removed))==extended['optimistic_union_removed_flops']
    assert set(exact['removed_indices'])<=set(extended['removed_indices'])
    assert trace['full_flops']==read(HERE/f'rank{rank:04d}/result.json')['full_flops']
    zero_output=[r['index'] for r in rows if r['a']['all_zero'] or r['b']['all_zero']]
    assert all(rows[i]['c']['all_zero'] for i in zero_output)
    gram=[r['index'] for r in rows if signature(r['a'])==signature(r['bt'])]
    fraction=extended['optimistic_union_removed_flops']/trace['full_flops']
    results.append(dict(rank=rank,full_flops=trace['full_flops'],outer_mm_flops=trace['outer_mm_flops'],outer_calls=len(rows),
        exact=exact,transpose_and_zero_union=extended,optimistic_complete_saving_fraction=fraction,
        passes_5_percent=fraction>=.05,exact_unweighted_Gram_indices=gram,
        exhaustive_pairwise_audit_pass=True,all_zero_products_verified=True,trace=pin(HERE/f'rank{rank:04d}/trace.json'),
        hashed_payload_bytes=trace['hashed_payload_bytes']))
passed=all(x['passes_5_percent'] for x in results)
write(HERE/'result.json',dict(completed=True,implementation_gate_passed=passed,rows=results,
    minimum_complete_saving_fraction=.05,scope='exact and transposed whole-product equivalence plus complete zero operands on the two current traces',
    optimistic_free_cache_ceiling=True,not_a_measured_candidate_bill=True,targets_read=False,candidate_created=False,new_fit_steps=0,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(completed=True,implementation_gate_passed=passed,rows=[dict(rank=r['rank'],fraction=r['optimistic_complete_saving_fraction'],duplicates=len(r['transpose_and_zero_union']['duplicates']),zero_products=len(r['transpose_and_zero_union']['zero_indices'])) for r in results]),flush=True)
