"""Close a completed diagnostic without reopening any failed candidate."""
from datetime import datetime,timezone
import ast,math,os,shutil,subprocess
from common import *
seal_check();completed=read(HERE/'completed.json');r=read(HERE/'result.json');audit=read(HERE/'audit.json');pred=read(HERE/'predictions_sealed.json')
assert completed['completed'] and completed['audit_passed'] and audit['passed'] and r['completed'] and pred['completed']
assert completed['guarded_names']==['unit','predict','measure','audit'] and not completed['uploaded']
check(audit['result']);check(r['labels']);check(r['predictions']);check(pred['predictions'])
for p in pred['models'].values():check(p)
copies=read(HERE/'source_copies.json')
for row in copies.values():check(row['old']);check(row['new'])
assert Path(copies['model']['old']['path']).read_bytes()==Path(copies['model']['new']['path']).read_bytes()
source=Path(copies['tangent']['old']['path']).read_text();node=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='tangent_cov')
assert ast.get_source_segment(source,node)+'\n'==Path(copies['tangent']['new']['path']).read_text()
assert len(pred['checks'])==14 and all(x['source_relative']<=3e-5 and x['delta_relative']<=1e-4 for x in pred['checks'])
assert r['rank']==35 and len(r['checks'])==7 and [x['layer'] for x in r['layers']]==list(range(8,15))
assert r['parent_and_corrections_have_stream_A_training_dependence'] and not r['official_score']
assert r['new_fit_steps']==r['new_samples']==r['new_native_calls']==0
assert all(x['original_FP32_U_byte_exact'] and x['original_covariance_energy_relative']<=2e-10 and x['original_response_energy_relative']<=2e-10 and x['replica_linearity_relative']<=2e-10 and x['independent_response_relative']<=2e-10 for x in r['checks'])
assert all(max(v.values())<=1e-4 for v in r['affine_checks'].values())
for channel,x in r['pooled'].items():
    b=x['baseline']
    for key in ('energy_A','energy_B','observed_energy','cross_energy','dispersion_energy'):
        ref=math.fsum(row['baseline'][channel][key] for row in r['layers'])
        assert abs(ref-b[key])<=2e-10*max(b['observed_energy'],1e-30)
    assert b['dispersion_fraction']==b['dispersion_energy']/b['observed_energy']
    for arm,z in x['arms'].items():
        assert z['ratio_A']==z['energy_A']/b['energy_A'] and z['ratio_B']==z['energy_B']/b['energy_B']
        assert abs(z['gain_A']-z['gain_B']-z['gain_gap'])<=2e-10*b['observed_energy']
attention=any(r['pooled'][k]['baseline']['dispersion_fraction']>=.20 for k in ('covariance','response'))
assert attention==r['further_label_acquisition_merits_separate_design'] and r['threshold']==.20
guards=[done(n) for n in completed['guarded_names']];groups=sorted(g['pid'] for g in guards);remaining=[];workers=[]
for line in subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,rss=,command='],text=True).splitlines():
    t=line.strip().split(None,3)
    if len(t)!=4:continue
    if int(t[0]) in groups or int(t[1]) in groups:remaining.append(line)
    if int(t[0])!=os.getpid() and any(s in t[3] for s in (str(RUNTIME),str(D/'run_guarded.py'),str(HERE/'run.py'))):workers.append(line)
assert not remaining and not workers,(remaining,workers)
bank=read(D/'official_best_331684.json');bank_pins=[v for v in bank.values() if isinstance(v,dict) and {'path','sha256','bytes'}<=v.keys()]
assert len(bank_pins)==5
for p in bank_pins:check(p)
previous=read(D/'continuation_checkpoint_v82.json')
for key in ('study','native_study','official_bank'):check(previous[key])
artifact_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file());assert artifact_bytes<256_000_000
resources=dict(guarded_groups=4,controller_terminal_session=28123,preparation_terminal_session=51931,all_groups_retired=True,
    peak_group_rss_bytes=max(g['max_rss_bytes'] for g in guards),
    peak_self_reported_rss_bytes=max(read(HERE/f'{name}.json')['peak_rss_bytes'] for name in ('unit','result','audit')),
    max_swap_growth_bytes=0,new_fit_steps=0,new_samples=0,new_native_calls=0,new_downloads=0,GPU_calls=0,
    existing_reference_streams=2,samples_per_existing_stream=8_000_000,artifact_bytes_before_closeout=artifact_bytes,
    disk_free_bytes=shutil.disk_usage(HERE).free)
resources['peak_self_reported_rss_bytes']=max(resources['peak_self_reported_rss_bytes'],pred['peak_rss_bytes'])
write(HERE/'resources.json',resources)
summary=dict(completed=True,diagnostic_qualified=True,rank=35,pooled=r['pooled'],resources=resources,
    further_label_acquisition_merits_separate_design=attention,no_automatic_followup=True,
    old_oracle_replica_result_unchanged=True,current_parent_label_target_measured=True,
    original_U_FP32_bytes_and_wide_energies_reproduced=True,models_unchanged=True,rank35_is_training=True,
    stream_A_training_dependence_explicit=True,unbiased_or_confidence_claim=False,
    current_nonlinear_native_panel_still_failed=True,official_changed=False,goal_achieved=False,candidate=None)
write(HERE/'summary.json',summary)
report='''# Current covariance-innovation replica audit completed

This measures the exact current331684 Gaussian-gain innovation target used by
checkpoint70/81, on rank35 layers8..14, using two existing independent8M
sample streams. It is distinct from the older oracle-state replica audit.
There is no new fit, sampling, estimator call or candidate.

|Channel|Pooled dispersion / observed|Stream cosine|
|---|---:|---:|
'''
for channel in ('covariance','diagonal','off_diagonal','response'):
    x=r['pooled'][channel]['baseline'];report+=f"|{channel}|{x['dispersion_fraction']:.12f}|{x['cosine']:.12f}|\n"
report+='\nDispersion is half the mean squared stream difference; observed energy is\nthe average of the two squared energies. The cross product plus dispersion\nreproduces observed energy. Figures pool the seven equally weighted layer\nenergies before taking ratios. They are not averages of layer ratios.\n\n'
report+='|Fixed correction|Covariance residual / A|Covariance residual / B|Response residual / A|Response residual / B|\n|---|---:|---:|---:|---:|\n'
for arm in ARMS:
    c=r['pooled']['covariance']['arms'][arm];v=r['pooled']['response']['arms'][arm]
    report+=f"|{arm}|{c['ratio_A']:.12f}|{c['ratio_B']:.12f}|{v['ratio_A']:.12f}|{v['ratio_B']:.12f}|\n"
report+='''
The original final16-output and528-parameter models are unchanged. Their
fourteen full current-state deltas were sealed before reference decoding in
this study and match independent FP64 formulas. The bank reconstruction is
byte-exact. These modest proxy gains also appear against streamB on this
training network; that does not establish native or cross-network benefit.
The nonlinear model's broader native panel remains failed at1.009582713004.
Its diagonal covariance residual worsens on both streams, despite improvement
in the full covariance measure. Complete per-layer results are in result.json.

The fixed decision threshold was at least20% pooled dispersion fraction in
covariance OR next response. Reaching it would support designing a separate
independent-label acquisition study; it never authorized an automatic new
stream or fit. Threshold reached: '''+str(attention)+'.\n\n'
if not attention:report+='This audit does not support sample noise as the main explanation for the\nfailed current fits. It supplies no reason to expand samples or rerun them.\n'
report+='''
These are two-stream point diagnostics, not a confidence-certified noise
ceiling. Both streams share the MPS/FP32 numerical implementation and its
unmeasured common bias. Rank35 is a TRAIN network; the parent and fitted
corrections have used streamA. Their dependence prevents an unbiased
fitted-arm cross-risk or held-out interpretation. No global sampling or
learning impossibility is asserted.

Both reference affine checks pass1e-4. Every original FP32 U byte and wide
covariance/response energy reproduces. The replica difference obeys the
independent linear-transport identity;16 fixed response columns per layer
and stream pass vector-matrix-vector checks. Synthetic indexed tangent and
noise/gain/partition identities pass, including zero and negative-cross cases.
An independent flattened-dot implementation and fsum verify all saved full
labels, both model losses, pooling and gain-gap identities within2e-10.

'''
report+=f"Four guarded groups retired; controller28123 terminal. Peak sampled RSS{resources['peak_group_rss_bytes']} B,self-report{resources['peak_self_reported_rss_bytes']} B,zero added swap. Artifacts {artifact_bytes/1e6:.3f} MB before closeout; no new sampling,fit,native call,download or GPU. All five bank pins and checkpoint82 studies verified.\n\n"
report+='Official331684 remains adjusted3.5844144212297072e-9,raw1.949715441540434e-8,100/100 complete,zero failures.4e-9 banked;3/2/~1e-9 unmet. The full goal remains active; qualified uploads remain authorized.\n'
with (HERE/'RESULT.md').open('x') as f:f.write(report)
external=[p for p in read(HERE/'sources_sealed.json')['pins'] if not Path(p['path']).is_relative_to(HERE)]+bank_pins+[previous['study'],previous['native_study']]
external=list({p['path']:p for p in external}.values())
for p in external:check(p)
write(HERE/'closeout.json',dict(completed=True,diagnostic_qualified=True,created_at_utc=datetime.now(timezone.utc).isoformat(),
    artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()],external_pins=external,resources=resources,
    further_label_acquisition_merits_separate_design=attention,candidate=None,official_changed=False,goal_achieved=False))
print(dict(closeout=pin(HERE/'closeout.json'),attention=attention,resources=resources),flush=True)
