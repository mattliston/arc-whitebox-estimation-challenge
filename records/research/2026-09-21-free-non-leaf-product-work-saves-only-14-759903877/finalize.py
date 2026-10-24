from datetime import datetime,timezone
import os,shutil,subprocess
from common import *
seal_check()
for p in read(HERE/'finalizer_sources_sealed.json')['pins']:verify(p)
r=read(HERE/'result.json');audit=read(HERE/'audit_result.json');c=read(HERE/'completed.json')
assert r['completed'] and audit['passed'] and c['completed'] and not r['both_pass_planning_gate']
guards=[done(n) for n in c['guards']];assert len(guards)==2
groups=sorted(g['pid'] for g in guards);active=[]
for line in subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,rss=,command='],text=True).splitlines():
    t=line.strip().split(None,3)
    if len(t)==4 and (int(t[0]) in groups or int(t[1]) in groups or
        (int(t[0])!=os.getpid() and any(s in t[3] for s in ('openblas_runtime_env_v1/bin/python',str(D/'run_guarded.py'),str(HERE/'run.py'))))):active.append(line)
assert not active,active
prior=read(D/'continuation_checkpoint_v85.json')
for k in ('study','pending_release','pending_candidate','pending_archive','rejection','official_bank'):verify(prior[k])
pending=D/'frozen_covariance_readout_composition_v1'
for name in ('submission_intent.json','submission.json','submission_attempt.json'):assert not (pending/name).exists()
resources=dict(guarded_groups=2,groups=groups,all_groups_retired=True,controller_terminal_session=3037,
    full_native_calls=0,full_native_flops=0,arrays_decoded=0,targets_read=False,
    peak_sampled_group_rss_bytes=max(g['max_rss_bytes'] for g in guards),
    peak_self_rss_bytes=max(r['peak_rss_bytes'],audit['peak_rss_bytes']),
    sampled_peak_is_sparse_for_subsecond_stages=True,max_swap_growth_bytes=0,
    new_fit_steps=0,new_samples=0,new_downloads=0,GPU_calls=0,
    artifact_bytes_before_closeout=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),
    disk_free_bytes=shutil.disk_usage(HERE).free,
    swap=subprocess.check_output(['/usr/sbin/sysctl','-n','vm.swapusage'],text=True).strip())
assert resources['artifact_bytes_before_closeout']<5_000_000
write(HERE/'resources.json',resources)
report='''# Free overhead around the current leaf products still misses the planning cut

Granting every charged operation inside the current outermost matrix-product
routines for free, except their existing leaf matrix multiplications, saves
at most **14.759903877% on35 and13.690445234% on100**. Both are below the
fixed **16.304320666%** uniform planning cut associated with3e-9 at unchanged
raw error. The standalone fixed-leaf caching route does not advance under
this test. No kernel or candidate was built.

| Network | Original complete FLOPs | Retained leaf FLOPs inside products | All other product work granted free | Optimistic complete remainder |
| --- | ---: | ---: | ---: | ---: |
'''
for x in r['rows']:
    report+=f"| {x['rank']} | {x['complete_cold_flops']} | {x['retained']} | {x['removed']} | {x['optimistic_remaining_complete_flops']} |\n"
report+='''
The allowance is deliberately stronger than input-transform caching. It
removes transforms of unique inputs as well as repeated inputs, output
recombination, packing, geometry binding and other non-leaf work inside
_mm. All computation outside _mm remains paid. The remainder is an
arithmetic bound for this specified deletion family, not an executable
estimator, measured new bill or official-score prediction. The two reused
development traces do not certify the cost distribution on official cases.

The saved identity traces contain590 input occurrences each:428 exact
classes,162 occurrences after the first; granting transpose equivalence
gives383 classes and207 subsequent occurrences. Empty operands are included.
This establishes input repetition even though all295 complete products are
distinct. It does not establish compatible cached transforms or their net
saving. The observed input-transform arithmetic/copy subtotals are
26,911,117,884 and25,120,389,238 FLOPs; even those include unique-input work.
The existing W5 reuse is already reflected in the recorded bills.

All590 ordered product records agree between the original cost census and
the later exact identity census. Stack charges partition to the complete
bills and independent outer-product totals exactly. Four active source AST
statements certify the retained leaf operations. Installed matmul dispatch
charges each such operation once. Seven synthetic stack records cover nested
_mm frames, direct outside products, zero charges and mismatching sites.
An independent integer subtraction audit reproduces all totals and gates;
pairwise comparison independently reproduces both operand equivalence counts.
No source, meter or estimator arithmetic was changed.

Different leaf algorithms/depths, shared leaf products, changed matrix
structure/ranks, outside-work savings, numerical approximations and combined
accuracy/cost changes are outside the result. The earlier finite-catalog
allowance cannot be added to this one: their operation schedules overlap
and differ. Smaller transform-cache gains remain possible and could matter
in a separately justified composition; this result is not a universal
obstruction or a claim that every incremental cost improvement is useless.

The source review also confirms that antithetic first-layer linear controls,
first-layer quadratic RQMC and the direct Gaussian-forecast terminal-skew
model were previously executed and failed their stated gates. Their exact
scopes and partial/fidelity failures remain unchanged; no such study was
restarted. The initial preparation script encountered checkpoint85's null
study-candidate field before producing any output. Its original source and
the schema correction are preserved before the experiment seal.
'''
report+=f'''
Both guarded groups retired; controller3037 is terminal. No new model calls,
arrays, targets, fits, samples, GPU work, downloads or uploads. Peak process
RSS was{resources['peak_self_rss_bytes']} bytes; the sparse0.5s supervisor
samples reached{resources['peak_sampled_group_rss_bytes']} bytes. Zero added
swap. About{resources['artifact_bytes_before_closeout']/1e6:.3f} MB of new
artifacts before closeout.

Official331684 remains adjusted3.5844144212297072e-9,raw1.949715441540434e-8,
100/100 complete,zero failures.4e-9 is banked;3/2/~1e-9 remain unmet and the
full goal is active. The qualified covariance/readout composition remains
ready and unchanged. Automatic review rejected its external upload before
process creation because it could not verify trusted authorization for this
exact private payload and AIcrowd destination. The already requested exact
approval remains unanswered; no retry, bypass or new question occurred.
'''
with (HERE/'RESULT.md').open('x') as f:f.write(report)
write(HERE/'summary.json',dict(completed=True,diagnostic_qualified=True,both_pass_planning_gate=False,
    savings=[x['optimistic_saving_fraction'] for x in r['rows']],candidate=None,official_changed=False,
    goal_achieved=False,resources=resources,report=pin(HERE/'RESULT.md'),result=pin(HERE/'result.json'),audit=pin(HERE/'audit_result.json'),
    pending_archive=prior['pending_archive'],upload_approval_still_pending=True,upload_attempted_this_turn=False))
external=[p for p in read(HERE/'sources_sealed.json')['pins'] if not Path(p['path']).is_relative_to(HERE)]
for p in external:verify(p)
write(HERE/'closeout.json',dict(completed=True,diagnostic_qualified=True,both_pass_planning_gate=False,
    created_at_utc=datetime.now(timezone.utc).isoformat(),candidate=None,official_changed=False,goal_achieved=False,
    artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()],external_pins=external,resources=resources,
    pending_archive=prior['pending_archive'],upload_approval_still_pending=True,upload_attempted_this_turn=False))
print(dict(completed=True,diagnostic_qualified=True,resources=resources),flush=True)
