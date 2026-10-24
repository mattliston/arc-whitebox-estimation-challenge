from datetime import datetime,timezone
import os,shutil,subprocess
from common import *
seal_check();completed=read(HERE/'completed.json');result=read(HERE/'result.json');audit=read(HERE/'fingerprint_audit.json')
assert completed['completed'] and result['completed'] and not result['implementation_gate_passed'] and audit['passed']
for p in read(HERE/'fingerprint_audit_sources_sealed.json')['pins']:verify(p)
for p in read(HERE/'traces_sealed.json')['pins']:verify(p)
guards=[done(name) for name in completed['guards']+['fingerprint_audit']];assert len(guards)==5
groups=sorted(g['pid'] for g in guards);active=[]
for line in subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,rss=,command='],text=True).splitlines():
    t=line.strip().split(None,3)
    if len(t)==4 and (int(t[0]) in groups or int(t[1]) in groups or (int(t[0])!=os.getpid() and any(s in t[3] for s in ('openblas_runtime_env_v1/bin/python',str(D/'run_guarded.py'),str(HERE/'run.py'))))):active.append(line)
assert not active,active
native=[read(HERE/f'rank{rank:04d}/result.json') for rank in (35,100)]
for r in native:
    assert r['all16_bytes_and_complete_bill_exact'] and r['all295_shape_dtype_caller_bills_exact'] and r['numeric_cache_retired'] and r['observer_restored']
for r in result['rows']:
    for name in ('exact','transpose_and_zero_union'):
        x=r[name];assert x['unique_classes']==295 and not x['duplicates'] and x['zero_indices']==[3,6] and x['optimistic_union_removed_flops']==0
    assert r['optimistic_complete_saving_fraction']==0 and r['exact_unweighted_Gram_indices']==[0,2]
previous=read(D/'continuation_checkpoint_v84.json')
for k in ('study','candidate','archive','rejection','official_bank'):verify(previous[k])
pending=D/'frozen_covariance_readout_composition_v1'
for name in ('submission_intent.json','submission.json','submission_attempt.json'):assert not (pending/name).exists()
resources=dict(guarded_groups=len(guards),groups=groups,all_groups_retired=True,controller_terminal_session=31126,
    fingerprint_audit_terminal_session=9062,full_native_calls=2,full_native_flops=sum(r['full_flops'] for r in native),
    peak_group_rss_bytes=max(g['max_rss_bytes'] for g in guards),peak_native_self_rss_bytes=max(r['peak_rss_bytes'] for r in native),
    max_swap_growth_bytes=0,total_fingerprinted_payload_bytes=sum(r['hashed_payload_bytes'] for r in result['rows']),
    max_payload_block_bytes=1024*1024,new_fit_steps=0,new_samples=0,new_downloads=0,GPU_calls=0,targets_read=False,
    artifact_bytes_before_closeout=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),disk_free_bytes=shutil.disk_usage(HERE).free,
    swap=subprocess.check_output(['/usr/sbin/sysctl','-n','vm.swapusage'],text=True).strip())
write(HERE/'resources.json',resources)
report=f'''# Whole-product reuse has zero opportunity on the two current traces

All **295 outermost _mm products** on each of networks35 and100 have distinct
canonical operand-pair fingerprints. Including transposed equivalents does
not merge any classes. The two zero-operand products are already-free empty
products at layer0/1. Consequently the most favorable exact/transpose/zero
whole-product reuse ceiling is **0 FLOPs, 0% of either complete cold bill**.
The fixed5% implementation gate fails, and no cache candidate is created.

| Network | Complete cold FLOPs | Outermost _mm FLOPs | Classes, including transposes | Optimistic removable FLOPs |
| --- | ---: | ---: | ---: | ---: |
'''
for r in result['rows']:report+=f"| {r['rank']} | {r['full_flops']} | {r['outer_mm_flops']} | 295 | 0 |\n"
report+='''
Both observed forwards reproduce all16 original output bytes and complete
cold bills. Every one of the295 ordered shape/dtype/caller/per-call bill
records matches the previous qualified current-bank census. The _mm wrapper
was restored, original meter and arithmetic were unchanged, and native
caches retired. No targets or reference arrays were read.

Streaming hashes match independent one-shot serialization for contiguous,
Fortran, transposed, strided, reversed, empty and signed-zero arrays. Three
nonfinite fixtures are rejected. Synthetic product groups match independent
pairwise enumeration. An additive audit strengthens the first unit's
preservation check with BEFORE-call copies, read-only arrays and five real
multi-block fixtures; all pass. Independent exhaustive pairwise comparison
certifies all native groups and zero-product costs. Signed-zero bits remain
part of identity, except for completely zero operands.

This closes only whole-product identity/transpose/zero reuse on these two
traces. It does not cover partial transforms, intermediate subproducts,
matrix-chain reassociation, low-rank or approximate structure, direct
non-_mm operations, other inputs or other algorithms. Research hashing time
is not deployment runtime. No production cache or changed cost is claimed.
The two unweighted Gram identities are already inside _resident_gram512 at
layer0; their full combined bill is only859975168 FLOPs (about0.208%/0.207%).
That post-result observation selects no further numerical study.
'''
report+=f'''
Five guarded groups retired after two full calls totaling
{resources['full_native_flops']} FLOPs. Peak sampled group RSS was
{resources['peak_group_rss_bytes']} bytes; self-reported native peak was
{resources['peak_native_self_rss_bytes']} bytes. Zero added swap. About
{resources['total_fingerprinted_payload_bytes']/1e9:.3f} GB was read through
1MiB payload blocks; no wide arrays or dataset copies were saved. No fitting,
samples, GPU, downloads, candidate or upload.

Official331684 remains adjusted3.5844144212297072e-9,raw1.949715441540434e-8,
100/100 complete,zero failures. The4e-9 milestone is banked;3/2/~1e-9 remain
unmet. The full goal is active. The qualified0.41%-better local composition
and its exact archive are unchanged. Its prior automatic upload-review
rejection still awaits the already requested approval; this turn made no
upload attempt or alternate external write.
'''
with (HERE/'RESULT.md').open('x') as f:f.write(report)
write(HERE/'summary.json',dict(completed=True,diagnostic_qualified=True,implementation_gate_passed=False,
    distinct_classes_per_trace=295,zero_products_per_trace=2,removable_flops_per_trace=0,
    candidate=None,official_changed=False,goal_achieved=False,resources=resources,
    pending_composition_archive=previous['archive'],upload_approval_still_pending=True,upload_attempted_this_turn=False,
    report=pin(HERE/'RESULT.md'),result=pin(HERE/'result.json'),fingerprint_audit=pin(HERE/'fingerprint_audit.json')))
external=[p for p in read(HERE/'sources_sealed.json')['pins'] if not Path(p['path']).is_relative_to(HERE)]
for p in external:verify(p)
write(HERE/'closeout.json',dict(completed=True,diagnostic_qualified=True,implementation_gate_passed=False,
    created_at_utc=datetime.now(timezone.utc).isoformat(),candidate=None,official_changed=False,goal_achieved=False,
    artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()],external_pins=external,resources=resources,
    pending_composition_archive=previous['archive'],upload_approval_still_pending=True,upload_attempted_this_turn=False))
print(dict(completed=True,diagnostic_qualified=True,removable_flops=0,resources=resources),flush=True)
