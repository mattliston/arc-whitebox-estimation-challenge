from fractions import Fraction
import collections,resource
from common import *
seal_check();rows=[]
threshold=1-Fraction('3e-9')/Fraction('3.5844144212297072e-9')
previous=read(read(HERE/'inputs.json')['previous_bound']['path'])
for rank in (35,100):
    done(f'capture{rank}');record=read(HERE/f'rank{rank:04d}/result.json')
    assert record['completed'] and record['all16_bytes_and_complete_bill_exact'];verify(record['trace'])
    t=read(record['trace']['path']);old=next(r for r in previous['rows'] if r['rank']==rank)
    assert len(t['outer_calls'])==295 and sum(r['flops'] for r in t['outer_calls'])==old['inside']
    leaves=t['leaves'];dense=sum(x['actual_flops'] for x in leaves);lower=sum(x['retained_lower_flops'] for x in leaves)
    assert dense==old['retained'] and 0<=lower<=dense
    remaining=old['outside']+lower;removed=old['complete_cold_flops']-remaining
    savings=Fraction(removed,old['complete_cold_flops'])
    sites=[]
    for key in sorted(SITES):
        rr=[x for x in leaves if (x['function'],x['line'])==key]
        sites.append(dict(function=key[0],line=key[1],calls=len(rr),batches=sum(x['batches'] for x in rr),
            original_flops=sum(x['actual_flops'] for x in rr),retained_lower_flops=sum(x['retained_lower_flops'] for x in rr),
            nonzero_terms=sum(x['nonzero_terms'] for x in rr),dense_terms=sum(x['dense_terms'] for x in rr),
            zero_operand_batches=sum(x['zero_operand_batches'] for x in rr)))
    rows.append(dict(rank=rank,complete_flops=old['complete_cold_flops'],outside_flops=old['outside'],
        original_leaf_flops=dense,optimistic_retained_leaf_flops=lower,extra_zero_term_allowance=dense-lower,
        original_free_nonleaf_allowance=old['removed'],optimistic_remaining_complete_flops=remaining,
        optimistic_saving_fraction=float(savings),saving_fraction_exact=[savings.numerator,savings.denominator],
        passes_uniform_3e9_planning_gate=savings>=threshold,leaf_calls=len(leaves),site_counts=sites))
write(HERE/'result.json',dict(completed=True,rows=rows,required_saving_fraction=float(threshold),
    required_fraction_exact=[threshold.numerator,threshold.denominator],both_pass_planning_gate=all(r['passes_uniform_3e9_planning_gate'] for r in rows),
    optimistic_not_executable=True,new_native_calls=2,targets_read=False,candidate_created=False,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(completed=True,rows=rows),flush=True)
