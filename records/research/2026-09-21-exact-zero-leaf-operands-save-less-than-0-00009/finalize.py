"""Preserve the measured fixed-schedule zero-term ceiling and its limits."""
from datetime import datetime,timezone
import math,os,shutil,subprocess
from common import *
seal_check()
for p in read(HERE/'finalizer_sources_sealed.json')['pins']:verify(p)
result=read(HERE/'result.json');audit=read(HERE/'audit_result.json');completed=read(HERE/'completed.json')
assert result['completed'] and audit['passed'] and completed['completed'] and not result['both_pass_planning_gate']
assert not result['targets_read'] and not result['candidate_created']
guards=[done(n) for n in completed['guards']];assert len(guards)==5
records=[read(HERE/f'rank{rank:04d}/result.json') for rank in (35,100)]
replay=[]
for record in records:
    assert record['all16_bytes_and_complete_bill_exact'] and record['observer_restored'] and record['numeric_cache_retired']
    verify(record['prediction']);verify(record['trace'])
    trace=read(record['trace']['path']);total=0
    for x in trace['leaves']:
        a=x['left_shape'];b=x['right_shape'];batch=math.prod(a[:-2]);m,k,n=a[-2],a[-1],b[-1]
        assert x['actual_flops']==batch*m*n*(2*k-1) and x['zero_operand_batches']==0
        # Independent aggregate relaxation; it happens to equal every recorded
        # per-batch lower count on these traces, so no per-batch histogram is needed.
        T=x['nonzero_terms'];qmax=min(T,batch*m*n);lower=2*T-qmax
        assert lower==x['retained_lower_flops'];total+=lower
    row=next(r for r in result['rows'] if r['rank']==record['rank'])
    assert total==row['optimistic_retained_leaf_flops']
    replay.append(dict(rank=record['rank'],leaf_calls=len(trace['leaves']),
        batches=sum(x['batches'] for x in trace['leaves']),dense_terms=sum(x['dense_terms'] for x in trace['leaves']),
        nonzero_terms=sum(x['nonzero_terms'] for x in trace['leaves']),zero_operand_batches=0,
        aggregate_lower_exactly_matches_all_recorded_perbatch_totals=True,retained_lower=total))
write(HERE/'aggregate_replay.json',dict(passed=True,rows=replay,after_negative_result=True,
    no_new_arrays_or_targets=True,gate_unchanged=True))
groups=sorted(g['pid'] for g in guards);active=[]
for line in subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,rss=,command='],text=True).splitlines():
    t=line.strip().split(None,3)
    if len(t)==4 and (int(t[0]) in groups or int(t[1]) in groups or
        (int(t[0])!=os.getpid() and any(s in t[3] for s in (str(RUNTIME),str(D/'run_guarded.py'),str(HERE/'run.py'))))):active.append(line)
assert not active,active
prior=read(D/'continuation_checkpoint_v87.json')
for k in ('study','pending_release','pending_candidate','pending_archive','rejection','official_bank'):verify(prior[k])
pending=D/'frozen_covariance_readout_composition_v1'
for name in ('submission_intent.json','submission.json','submission_attempt.json'):assert not (pending/name).exists()
resources=dict(guarded_groups=5,groups=groups,all_groups_retired=True,controller_terminal_session=11458,
    full_native_calls=2,full_native_flops=sum(r['full_flops'] for r in records),
    leaf_calls=sum(r['leaf_calls'] for r in records),leaf_batches=sum(r['batches'] for r in replay),
    peak_sampled_group_rss_bytes=max(g['max_rss_bytes'] for g in guards),
    peak_self_rss_bytes=max(x['peak_rss_bytes'] for x in records+[read(HERE/'unit.json'),result,audit]),
    max_swap_growth_bytes=max(g['max_swap_growth_bytes'] for g in guards),
    total_guarded_wall_s=math.fsum(g['elapsed_s'] for g in guards),
    targets_read=False,new_fit_steps=0,new_samples=0,new_downloads=0,GPU_calls=0,
    artifact_bytes_before_closeout=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),
    disk_free_bytes=shutil.disk_usage(HERE).free,
    swap=subprocess.check_output(['/usr/sbin/sysctl','-n','vm.swapusage'],text=True).strip())
assert resources['artifact_bytes_before_closeout']<30_000_000 and resources['max_swap_growth_bytes']==0
write(HERE/'resources.json',resources)
report='''# Exact leaf zeros add negligible savings to the fixed-schedule bound

The two unchanged native runs contain **no all-zero operand batches** among
17,475,569 leaf batches. Removing every scalar product with an exact-zero
operand would save at most **350,566 / 357,346 FLOPs** on networks 35 / 100
within this fixed leaf schedule. Each is less than 0.00009% of the complete
bank bill. A substantial sparsity opportunity was not hidden in these leaves.

Even after also granting all non-leaf work inside the current outer matrix
products for free, the optimistic total savings remain **14.759988667% /
13.690531080%**, below the fixed **16.304320666%** uniform planning cut for
3e-9. Both cases fail. No sparse kernel, estimator, package, or upload follows.

| Network | Complete bank FLOPs | Leaf scalar-product terms | Terms with an exact-zero operand | Optimistic complete remainder |
| --- | ---: | ---: | ---: | ---: |
'''
for row,rr in zip(result['rows'],replay):
    report+=f"| {row['rank']} | {row['complete_flops']} | {rr['dense_terms']} | {rr['dense_terms']-rr['nonzero_terms']} | {row['optimistic_remaining_complete_flops']} |\n"
report+='''
For each classical leaf batch, T is the exact number of nonzero scalar
products, obtained as the sum over contracted coordinates of the two
operand support counts. If Q outputs have any such term, ordinary sparse
accumulation costs T multiplies plus T-Q additions. Granting the maximum
Q=min(T,m*n) gives the lower count T+max(0,T-m*n). This is an optimistic
bound for zero-term deletion in the existing term circuit. It does not
assert that sparse dispatch, indexing, copying or complete inference could
achieve that count. All outside-matrix-product work remains paid.

All 795 charged leaf calls match the installed dense FP32 tariff and all
four original source-site totals exactly. Every one of the 590 outer call
records matches the previous shape/dtype/caller/bill census. Both predictions
match every one of the 16 original output rows and their complete cold bills.
Both callbacks were restored and native numeric caches retired.

Streaming count fixtures include signed zeros, sparse/disjoint supports,
batching, strided/transposed inputs, permutation, a buffer larger than 1MiB,
exact repeats, input-byte preservation, and malformed/nonfinite rejection.
The real FNP fixture preserves its output and bill. Independent exhaustive
scalar enumeration covers 4,420 binary matrix pairs and verifies the term
factorization and lower-bound inequality. The metadata audit reproduces
the complete integer accounting and gates. A separate post-result aggregate
replay derives 2T-min(T,batch*m*n) and exactly matches the recorded sum of
per-batch lower counts for every one of the 795 calls; no additional arrays,
targets, parameter choices or changed gates were involved.

This result excludes neither different leaf algorithms/schedules, reuse of
nonzero products, special-constant operations, cancellation of nonzero terms,
other structured representations, changed physical states, nor smaller
useful cost/accuracy compositions. These are two reused development traces,
not a distribution-wide bound or an official-score forecast. The earlier
finite-catalog and free-overhead allowances overlap and must not be added.
The source review also explains why arbitrary child schedule choice cannot
improve an unrestricted additive-cost optimum when every child has the same
shape; structural zeros and shared products are outside that observation.
'''
report+=f'''
Both full native calls used {resources['full_native_flops']:,} measured
FLOPs. All five guarded groups retired; controller 11458 is terminal. Peak
sampled group RSS was {resources['peak_sampled_group_rss_bytes']:,} bytes,
peak process RSS {resources['peak_self_rss_bytes']:,} bytes, with zero added
swap. About {resources['artifact_bytes_before_closeout']/1e6:.3f} MB of new
artifacts was preserved before closeout. No truth labels, fitting, new
sampling, GPU work, downloads or uploads were used.

Official 331684 remains adjusted 3.5844144212297072e-9 and raw
1.949715441540434e-8, 100/100 cases, zero failures. The 4e-9 milestone is
banked; 3e-9, 2e-9 and approximately 1e-9 remain unmet. The full Phase 2
goal remains active. At unchanged raw error, the 0.1 multiplier floor rules
out approximately 1e-9 from cost alone, but does not rule out 2e-9.

The separate covariance/readout release is unchanged. Automatic approval
review previously rejected its AIcrowd upload because it could not verify
trusted authorization for that exact private payload and destination. The
existing exact-upload approval question remains unanswered; there was no
retry, bypass or new upload attempt in this study.
'''
with (HERE/'RESULT.md').open('x') as f:f.write(report)
external=[p for p in read(HERE/'sources_sealed.json')['pins'] if not Path(p['path']).is_relative_to(HERE)]
for p in external:verify(p)
write(HERE/'summary.json',dict(completed=True,diagnostic_qualified=True,both_pass_planning_gate=False,
    savings=[r['optimistic_saving_fraction'] for r in result['rows']],zero_term_allowances=[r['extra_zero_term_allowance'] for r in result['rows']],
    candidate=None,official_changed=False,goal_achieved=False,resources=resources,report=pin(HERE/'RESULT.md'),
    result=pin(HERE/'result.json'),audit=pin(HERE/'audit_result.json'),aggregate_replay=pin(HERE/'aggregate_replay.json'),
    pending_archive=prior['pending_archive'],upload_approval_still_pending=True,upload_attempted_this_turn=False))
write(HERE/'closeout.json',dict(completed=True,diagnostic_qualified=True,both_pass_planning_gate=False,
    created_at_utc=datetime.now(timezone.utc).isoformat(),candidate=None,official_changed=False,goal_achieved=False,
    artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()],external_pins=external,resources=resources,
    pending_archive=prior['pending_archive'],upload_approval_still_pending=True,upload_attempted_this_turn=False))
print(dict(completed=True,diagnostic_qualified=True,resources=resources,artifacts=len(read(HERE/'closeout.json')['artifacts'])),flush=True)
