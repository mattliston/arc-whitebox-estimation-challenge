import math,resource
from common import *
seal_check();done('unit_v2');assert read(HERE/'unit_v2.json')['passed'];rows=[]
for rank in (35,100):
    done(f'rank{rank:04d}');record=read(HERE/f'rank{rank:04d}/result.json')
    assert record['completed'] and record['all16_bytes_and_complete_bill_exact'] and record['observers_restored'] and record['numeric_cache_retired']
    for key in ('source','prediction','trace'):verify(record[key])
    trace=read(record['trace']['path']);calls=trace['calls'];assert [r['layer'] for r in calls]==list(range(1,15))
    assert all(r['reference_relative_disagreement']<=1e-8 for r in calls)
    e2=math.fsum(r['effective_error_norm']**2 for r in calls)
    k2=math.fsum(r['full_K21_offdiag_norm']**2 for r in calls)
    c2=math.fsum(r['local_covariance_error_norm']**2 for r in calls)
    base2=math.fsum(r['original_covariance_offdiag_norm']**2 for r in calls)
    assert k2>0 and base2>0
    rows.append(dict(rank=rank,read_calls=14,flagged_layers=[r['layer'] for r in calls if r['exceeds_numerical_tolerance']],
        any_layer_flagged=any(r['exceeds_numerical_tolerance'] for r in calls),
        maximum_effective_error_over_full_K21=max(r['effective_error_over_full_K21'] for r in calls),
        pooled_effective_error_over_full_K21=math.sqrt(e2/k2),
        maximum_effective_read_relative=max(r['effective_read_relative_error'] for r in calls),
        pooled_local_covariance_relative=math.sqrt(c2/base2),maximum_local_covariance_relative=max(r['local_covariance_relative'] for r in calls),
        max_reference_relative_disagreement=max(r['reference_relative_disagreement'] for r in calls),
        max_reference_error_to_measured_error=max(r['reference_error_to_measured_error'] for r in calls),
        modes=sorted({r['mode'] for r in calls}),max_snapshot_bytes=max(r['snapshot_bytes'] for r in calls),
        max_reference_workspace_upper_bytes=max(r['workspace_upper_bytes'] for r in calls),
        full_flops=record['full_flops'],prediction=record['prediction'],trace=record['trace']))
result=dict(completed=True,diagnostic_qualified=True,rows=rows,
    numerical_tolerance=3e-5,both_networks_flagged=all(r['any_layer_flagged'] for r in rows),
    total_full_native_calls=2,full_native_flops=sum(r['full_flops'] for r in rows),
    targets_read=False,candidate_created=False,local_affine_response_is_not_autonomous_score=True,
    parent_outputs_and_bills_unchanged=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
write(HERE/'result.json',result);print(result,flush=True)
