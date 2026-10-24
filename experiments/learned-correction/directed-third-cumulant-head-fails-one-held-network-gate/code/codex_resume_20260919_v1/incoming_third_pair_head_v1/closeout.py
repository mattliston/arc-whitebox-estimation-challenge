"""Close only a failed fixed information screen; never promote a proxy score."""
import ast,os,re,resource,shutil
from datetime import datetime,timezone
from common import *
for n in ('sources_sealed.json','runtime_repair_sealed.json','fit_sources_sealed.json','replay_sources_sealed.json','directed_audit_sealed.json','audit_controller_sealed.json'):verify(n)
r=read(HERE/'information_result.json');v=read(HERE/'judgment_replay.json');t=read(HERE/'training_result.json');u=read(HERE/'unit_result.json')
d=read(HERE/'directed_response_audit.json');assert d['passed'] and not d['primary_gate_passed'] and d['no_refit_or_selection']
assert [x['original_primary_passed'] for x in d['records']]==[x['directed_sensitivity_would_pass'] for x in d['records']]
assert r['completed'] and not r['passed'] and v['passed'] and not v['information_gate_passed']
assert t['completed'] and t['updates']==1008 and len(t['epochs'])==24 and u['passed']
assert read(HERE/'fit_completed.json')['completed'] and read(HERE/'inputs_completed.json')['full_native_parent_calls']==10
assert not (HERE/'candidate.py').exists()
records=[];self_peaks=[u['peak_rss_bytes'],t['peak_rss_bytes'],r['peak_rss_bytes'],v['peak_rss_bytes'],d['peak_rss_bytes']]
for ordinal in TRAIN+HELDOUT:
    c=read(HERE/f'capture_mini{ordinal:04d}.json');q=read(HERE/f'targets_mini{ordinal:04d}.json')
    assert c['passed'] and q['passed'];records.extend(c['records'])
    self_peaks.extend([c['peak_rss_bytes'],q['peak_rss_bytes'],read(HERE/f'inputs_mini{ordinal:04d}.json')['peak_rss_bytes'],read(HERE/f'weights_mini{ordinal:04d}.json')['peak_rss_bytes']])
assert len(records)==10
c57=read(HERE/'capture_mini0057.json');assert all(x['all16_and_bill_exact'] for x in c57['records'])
guards=[read(p) for p in sorted(HERE.glob('*_watch.json'))];assert len(guards)==46,len(guards)
assert sorted(g['exit_code'] for g in guards)==[0]*45+[1]
assert all(g['whole_group_retired'] and g['max_swap_growth_bytes']==0 and g['error'] is None for g in guards)
assert read(HERE/'unit_watch.json')['exit_code']==1 and all(g['completed']==(g['exit_code']==0) for g in guards)
f=read(HERE/'model_frozen.json')
for k in ('model','normalizers','training'):check(f[k])
frozen_time=datetime.fromisoformat(f['utc']).timestamp()
for ordinal in HELDOUT:
    assert (HERE/'public_moments'/f'mlp_{ordinal:05d}.npz').stat().st_mtime>frozen_time
    assert (HERE/f'targets_mini{ordinal:04d}.json').stat().st_mtime>frozen_time
assert (HERE/'frozen_inputs_completed.json').stat().st_mtime<(HERE/'targets_mini0057.json').stat().st_mtime
downloads=[read(HERE/f'download_mini{o:04d}.json') for o in TRAIN+HELDOUT]
new_bytes=sum(x['new_bytes'] for x in downloads);assert new_bytes==2_356_640_712
all_checks=[x for o in TRAIN+HELDOUT for x in read(HERE/f'targets_mini{o:04d}.json')['checks'].values()];assert all(x['passed'] for x in all_checks)
groups=sorted(g['pid'] for g in guards)
rows=[]
for line in subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,rss=,command='],text=True).splitlines():
    a=line.strip().split(None,3)
    if len(a)==4:rows.append(dict(pid=int(a[0]),pgid=int(a[1]),rss_kib=int(a[2]),command=a[3]))
remaining=[x for x in rows if x['pid'] in groups or x['pgid'] in groups];assert not remaining,remaining
now=datetime.now(timezone.utc).isoformat();swap=subprocess.check_output(['/usr/sbin/sysctl','-n','vm.swapusage'],text=True).strip()
resource_floor()
resources=dict(utc=now,guard_count=len(guards),completed_guards=45,failed_import_guards=1,groups=groups,all_groups_retired=True,
      remaining_owned_rows=[],nice=10,blas_threads=1,max_sampled_group_rss_bytes=max(g['max_rss_bytes'] for g in guards),
      max_successful_process_rss_bytes=max(self_peaks),failed_import_process_self_peak_unknown=True,max_added_swap_bytes=0,
      total_guard_wall_s=sum(g['elapsed_s'] for g in guards),full_native_parent_calls=10,
      full_native_parent_flops=sum(x['flops'] for x in records),offline_training_not_flop_metered=True,
      new_download_bytes=new_bytes,artifact_bytes_before_closeout=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),
      disk_free_bytes=shutil.disk_usage(HERE).free,system_swap=swap)
write(HERE/'resources.json',resources)
summary=dict(completed=True,information_gate_passed=False,heldout=[dict(ordinal=x['ordinal'],ratios=x['ratios'],passed=x['passed']) for x in r['records']],
      final_training=t['final_training'],component_flops=u['conservative_seven_layer_flops'],full_candidate_created=False,
      official_changed=False,goal_achieved=False,independent_replay_passed=True,model=f['model'],resources=resources,
      response_interpretation='Original fit and gates use the frozen symmetric covariance-response surrogate; separately certified directed-source replay leaves the joint verdict unchanged.',
      directed_response=[{k:v for k,v in x.items() if k!='layers'} for x in d['records']])
write(HERE/'summary.json',summary)
table='\n'.join(f"| {x['ordinal']} | {x['ratios'][0]:.9f} | {x['ratios'][1]:.9f} | {x['ratios'][2]:.9f} |" for x in r['records'])
report=f'''# Directed incoming-third head fails its fixed information gate

The single24-epoch model is complete. At least one of the three required
consumer ratios fails on a held-out network, so this fixed model stops.
No native candidate, development-score panel, package or upload follows.
No extra epochs, amplitude, features, architecture or threshold were chosen
after observing the result.

| MINI ordinal | Weighted third-plane error ratio | Covariance-response ratio | Next-mean-response ratio |
| --- | ---: | ---: | ---: |
{table}

Each held-out network had to meet<=.85,<=.85,<=.90 respectively. Network14
passes all three. Network71 fails its weighted-third and next-mean gates.
These are frozen symmetric-surrogate response losses against public100M
moment targets, not actual
final-layer mean MSE, FLOP-adjusted scores or an official bank. Ratios use
the same TRAIN per-layer energy normalizers for numerator and denominator.
The independent target-centering and full-congruence replay reproduces the
judgment within its predeclared numerical tolerances.

Source review after model freeze caught a description error: the effective
pair series uses the same directed rho_ij for both third terms, whereas
the frozen half-plus-transpose loss uses rho_ji in the second term. Its
formula therefore describes a symmetric surrogate, not the exact directed
real-arithmetic consumer. All original sources, model, normalizers and gates
remain intact. The additive directed audit was sealed before root read the
held-out results. It certifies the actual source formula on symmetric,
asymmetric and active-floor fixtures, then measures all56 captured states.
Maximum covariance asymmetry is{max(x['relative'] for x in d['asymmetries']):.12g}.
Directed covariance and next-mean ratios differ from the original by at most
{max(abs(x['ratio_differences'][1]) for x in d['records']):.12g} and
{max(abs(x['ratio_differences'][2]) for x in d['records']):.12g}, respectively.
The joint pass/fail outcome stays unchanged. No retraining follows.

The audit also reports a DIFFERENT, exact directed-entry weight for the
third-plane loss. That changes the first metric by about.133/.134 because
the weighting definition differs; it is not a covariance-roundoff effect
or an alternate acceptance route. The original symmetric third-plane
weight and its gate stay authoritative. Network71's next-mean ratio still
exceeds.90 under the exact directed response.

The model is a directed32-hidden tanh head using140 causal node features,
including the previous parent hidden state transported by W and W squared,
plus current correlation and the two directed standardized third entries.
It predicts only offdiagonal pre-K21 corrections at layers8..14. The
existing CP carrier and marginal third moments would remain unchanged.
This is a scoped negative for this model, loss and fixed optimizer, not a
bound on other third-moment suppliers or all nonlinear models.

The prospective six/two split from the earlier metadata audit was retained:
TRAIN57,68,45,59,3,12; held-out14,71. All eight current-parent trajectories
were frozen before new target admission. The three57 qualification calls
match all16 output rows and complete bills exactly, and the observed
capture repeats byte-for-byte. Both held-out moment files were downloaded
after the final model, normalizers and training result were hash-frozen.
All eight files passed identity, Gaussian-input, affine, diagonal-storage
and rounding-bound checks. The1B final-mean target fields stayed unopened.
Late-layer sampling error is still present in the100M labels.

Equation, permutation, zero-output, repeat and analytic-gradient checks
passed. The1024-wide component measured1,189,302,272 weighted FLOPs for one
layer including cold parameter ingress; the deliberately conservative
seven-layer sum is8,325,115,904. It passed the15-billion component cap.
There is no measured full changed-estimator bill or runtime claim.

The first unit process stopped at an unavailable SciPy import before
numerical work. Its source and failure remain. An additive repair uses
Python math.erfc for the independent CDF and runs the unchanged projected
weight extractor in the existing PyArrow environment. Native calls and
training use the pinned official-compatible NumPy/flopscope environment.
This changes dependencies, not the frozen equations or fit design.

All46 guarded groups retired:45 successful and the original import failure.
The ten unchanged-parent calls consumed{resources['full_native_parent_flops']:,}
metered FLOPs. Maximum sampled group RSS was
{resources['max_sampled_group_rss_bytes']:,} bytes; maximum successful
process peak RSS was{resources['max_successful_process_rss_bytes']:,} bytes.
Added swap was zero. New downloads total2,356,640,712 bytes, and all new
artifacts total about{resources['artifact_bytes_before_closeout']/1e9:.3f}GB
before closeout, within the8GB cap. No GPU or new sampling was used.
See DATA_PROVENANCE.md for the source credits and pinned revisions.

Official331684 remains adjusted3.5844144212297072e-9 and raw
1.949715441540434e-8,100/100 cases, zero failures. The4e-9 milestone is
banked;3e-9,2e-9 and approximately1e-9 remain unmet. The overall goal is
active. This study changed no official submission.

Automatic approval review previously rejected the separate qualified
covariance/readout archive upload because it could not verify trusted
authorization for the exact private payload and AIcrowd destination. The
existing exact-upload approval question remains unanswered. This local
study did not retry or bypass that rejection.
'''
with (HERE/'RESULT.md').open('x') as f:f.write(report)
external={}
for seal in ('sources_sealed.json','runtime_repair_sealed.json','fit_sources_sealed.json','replay_sources_sealed.json','directed_audit_sealed.json','audit_controller_sealed.json'):
    for p in read(HERE/seal)['pins']:
        if not Path(p['path']).is_relative_to(HERE):check(p);external[p['path']]=p
for d in downloads:
    p=d['payload']
    if not Path(p['path']).is_relative_to(HERE):check(p);external[p['path']]=p
artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()]
assert sum(p['bytes'] for p in artifacts)<8_000_000_000
write(HERE/'closeout.json',dict(completed=True,utc=now,information_gate_passed=False,native_qualified=False,candidate=None,
      independent_replay_passed=True,official_changed=False,goal_achieved=False,summary=pin(HERE/'summary.json'),
      model=read(HERE/'model_frozen.json')['model'],resources=resources,artifacts=artifacts,external_pins=list(external.values())))
print(dict(completed=True,closeout=pin(HERE/'closeout.json'),summary=summary),flush=True)
