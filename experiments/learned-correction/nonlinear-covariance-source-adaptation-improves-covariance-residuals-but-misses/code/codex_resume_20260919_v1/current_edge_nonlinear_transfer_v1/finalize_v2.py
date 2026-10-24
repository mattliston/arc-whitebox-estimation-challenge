"""Preserve the two fixed fits and their held-network proxy verdict."""
from datetime import datetime,timezone
import os,shutil,subprocess
from common import *
import math
def producer_sum(items):
    value=0.0
    for item in items:value+=item
    return value
verify("finalize_v2_sources_sealed.json")
verify();verify('data_sealed.json');verify('models_sealed.json');verify('predictions_sealed.json');verify('finalize_sources_sealed.json')
completed=read(HERE/'completed.json');result=read(HERE/'result.json');unit=read(HERE/'unit_result.json')
assert completed['completed'] and result['completed'] and unit['passed'] and completed['passed']==result['passed']
assert not result['native_candidate'] and not result['complete_adjusted_score'] and not result['ordinary_1B_targets_opened']
assert result['new_samples']==result['full_native_calls']==0 and result['fit_count']==2
assert len(unit['fixtures'])==3 and all(x['parameters_checked']==528 and x['max_complex_step_gradient_relative']<=2e-10 for x in unit['fixtures'])
fits={arm:read(HERE/f'{arm}_fit.json') for arm in ('output_only','nonlinear')}
for arm,f in fits.items():
    assert f['completed'] and f['epochs']==40 and f['updates']==1680 and f['finite'] and f['final_epoch_only']
    assert f['train']==list(TRAIN) and f['valid_not_decoded'] and len(f['history'])==40
    assert [x['epoch'] for x in f['history']]==list(range(1,41))
    assert [x['step'] for x in f['history']]==list(range(42,1681,42))
    check(f['parameters']);check(f['bank_parameters'])
assert fits['output_only']['trained_parameters']==16 and fits['nonlinear']['trained_parameters']==528
assert fits['output_only']['covariance_norm']==fits['nonlinear']['covariance_norm']
assert fits['output_only']['response_norm']==fits['nonlinear']['response_norm']
prep=[read(HERE/f'data_{rank:04d}.json') for rank in TRAIN+VALID]
for rank,row in zip(TRAIN+VALID,prep):
    assert row['completed'] and row['rank']==rank and len(row['layers'])==len(row['reconstruction'])==7
    for k in ('arrays','capture','truth','weights','old_statistics'):check(row[k])
    assert max(row['affine_mean_relative'],row['affine_covariance_relative'])<=1e-4
    assert all(x['relative']<=3e-6 and x['zero_delta_exact'] for x in row['reconstruction'])
    assert all(max(x['covariance_energy_match_relative'],x['response_energy_match_relative'])<=2e-10 for x in row['layers'])
    assert all(max(x['target_rounding_relative'],x['response_rounding_relative'])<=3e-6 for x in row['layers'])
pred=read(HERE/'predictions_sealed.json');assert pred['completed'] and pred['held_targets_not_decoded']
assert {(r['arm'],r['rank']) for r in pred['rows']}=={(a,r) for a in fits for r in VALID}
for r in pred['rows']:
    assert len(r['checks'])==7
    assert all(c['source_relative']<=3e-5 and c['delta_relative']<=1e-4 for c in r['checks'])
assert len(result['rows'])==4 and len(result['gates'])==2
for rank in VALID:
    control=next(x for x in result['rows'] if x['rank']==rank and x['arm']=='output_only')
    full=next(x for x in result['rows'] if x['rank']==rank and x['arm']=='nonlinear')
    for row in (control,full):
        assert len(row['layers'])==7
        for prefix in ('covariance','response'):
            expected=producer_sum(l[prefix+'_residual'] for l in row['layers'])/producer_sum(l[prefix+'_base'] for l in row['layers'])
            high_precision=math.fsum(l[prefix+'_residual'] for l in row['layers'])/math.fsum(l[prefix+'_base'] for l in row['layers'])
            assert abs(expected-high_precision)<=2e-10*abs(expected)
            assert expected==row[prefix+'_ratio']
        assert all(l['covariance_identity_relative']<=2e-10 and l['response_identity_relative']<=2e-10 for l in row['layers'])
    gate=next(g for g in result['gates'] if g['rank']==rank)
    ratio=full['response_ratio']/control['response_ratio']
    assert gate['response_over_control']==ratio
    assert gate['passed']==(full['covariance_ratio']<=.98 and full['response_ratio']<=.95 and ratio<=.98)
assert result['passed']==all(g['passed'] for g in result['gates'])
names=completed['guarded_names'];assert len(names)==13
guards=[done(n) for n in names];groups=sorted(g['pid'] for g in guards);remaining=[];workers=[]
for line in subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,rss=,command='],text=True).splitlines():
    t=line.strip().split(None,3)
    if len(t)!=4:continue
    if int(t[0]) in groups or int(t[1]) in groups:remaining.append(line)
    if int(t[0])!=os.getpid() and any(s in t[3] for s in [str(RUNTIME),str(D/'run_guarded.py'),str(HERE/'run.py')]):workers.append(line)
assert not remaining and not workers,(remaining,workers)
bank=read(D/'official_best_331684.json');bank_pins=[v for v in bank.values() if isinstance(v,dict) and {'path','sha256','bytes'}<=v.keys()]
assert len(bank_pins)==5
for p in bank_pins:check(p)
previous=read(D/'continuation_checkpoint_v80.json');check(previous['study']);check(previous['official_bank'])
artifact_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file());assert artifact_bytes<2_000_000_000
resources=dict(guarded_groups=13,all_groups_retired=True,controller_terminal_session=46086,
    peak_group_rss_bytes=max(g['max_rss_bytes'] for g in guards),
    peak_self_reported_rss_bytes=max(x['peak_rss_bytes'] for x in [unit,result,pred]+prep+list(fits.values())),
    max_swap_growth_bytes=0,full_native_calls=0,fit_count=2,epochs_per_arm=40,updates_total=3360,closeout_verifier_failures=1,
    new_samples=0,new_downloads=0,GPU_calls=0,ordinary_1B_targets_opened=False,existing_covariance_reference_networks=8,
    artifact_bytes_before_closeout=artifact_bytes,disk_free_bytes=shutil.disk_usage(HERE).free)
write(HERE/'resources.json',resources)
summary=dict(completed=True,passed=result['passed'],negative=not result['passed'],candidate=False,
    scope='One fixed current-input nonlinear covariance-innovation adaptation versus matched output-only control; Gaussian-gain proxy on reused held networks, not a complete adjusted objective.',
    train=list(TRAIN),valid=list(VALID),rows=[{k:v for k,v in r.items() if k!='layers'} for r in result['rows']],gates=result['gates'],
    trained_parameters=dict(output_only=16,nonlinear=528),unchanged_deployment_architecture=True,
    source_normalizations_unchanged=True,previous_head_only_failure_preserved=True,
    derivative_fixtures=3,derivative_coordinates_per_fixture=528,wide_source_cells=56,independent_held_predictions=28,
    resources=resources,official_changed=False,goal_achieved=False,selected_next_numerical_study=None)
write(HERE/'summary.json',summary)
verdict=('PASS: fixed information gate' if result['passed'] else 'NEGATIVE: fixed information gate failed')
report=f'''# Current nonlinear edge adaptation: {verdict}

Both fits used the exact current official331684 frozen states, the same
coherent8M Gaussian-gain covariance-innovation labels, six TRAIN networks
and seven layers per network. VALID400/500 were excluded from both fits;
they have earlier research history and are not pristine holdouts. Forty
epochs and1680 updates completed for each arm, using identical cell orders,
optimizer, schedule, normalization and regularizer. Only final parameters
were evaluated; no validation checkpoint selection or fit extension.

The matched control trained16 output weights. The nonlinear arm trained
all528 parameters in the existing16-channel edge source. Input features,
normalizations, envelope and deployment architecture are unchanged. This
explicitly tests the encoder surface excluded by the earlier ridge study;
that study's source, failed gate and result remain unchanged.

|VALID rank|Arm|Covariance residual / baseline|Next-response residual / baseline|
|---|---|---:|---:|
'''
for row in result['rows']:
    report+=f"|{row['rank']}|{row['arm']}|{row['covariance_ratio']:.12f}|{row['response_ratio']:.12f}|\n"
report+='\nEach nonlinear held network requires covariance<=.98, response<=.95, and\nresponse<=.98 of the matched output-only control.\n\n'
for g in result['gates']:
    report+=f"- Rank{g['rank']}: response/control={g['response_over_control']:.12f}; passed={g['passed']}.\n"
report+=f'''
The proxy labels subtract the same declared Gaussian-gain transport of
previous mean/covariance errors. They are not exact full-program innovations
or derivatives, and retain coherent8M sampling/numerical error. The reverse
checks establish only the smooth frozen-input proxy's derivative, not a
native gradient or an autonomous improvement. No1B final-mean reference,
native candidate, complete bill, runtime or official score is measured.

All528 derivative coordinates pass independent scalar complex-step checks
on three fixed small fixtures, together with independent forward, symmetry,
permutation, null, directional-descent and FP32 checks. All56 wide bank
source reconstructions pass3e-6 before each network's targets are decoded.
Starting deltas are byte-zero. The copied innovation-transport function is
byte-identical to the old source, and all old covariance/response energies
reproduce within2e-10. FP32 target/response roundoff passes3e-6.
Both final models were sealed before VALID prediction. All28 held layer
deltas were sealed before scoring and pass an independent explicit-channel
FP64 reference. Direct and expanded quadratic losses agree within2e-10.

The first closeout verifier stopped on exact pooled-ratio equality: Python3.13
uses a different floating-point sum from the Python3.11 scoring runtime.
The preserved diagnostic shows differences of0--4 rounding units. Additive
finalize_v2 reproduces the producer's sequential sums exactly and independently
checks high-precision fsum within the existing2e-10 arithmetic tolerance.
No data, source model, fit, prediction, loss or scientific gate changed.
The original verifier, diagnostic and complete source inverse are retained.

All13 guarded groups retired; controller46086 terminal. Peak sampled RSS
{resources['peak_group_rss_bytes']} B, self-report
{resources['peak_self_reported_rss_bytes']} B,zero added swap. Artifacts before
closeout total{artifact_bytes/1e6:.3f} MB. No new estimator calls, samples,
downloads,GPU or ordinary1B targets. Existing captured states and references
were reused, with all old and official source pins retained.

Official331684 stays adjusted3.5844144212297072e-9,raw1.949715441540434e-8,
100/100 complete,zero failures.4e-9 is banked;3/2/~1e-9 remain unmet.
The full goal remains active; qualified uploads remain authorized.
'''
if result['passed']:
    report+='\nA separate prospective native finite-endpoint study of this one final payload\nis required next; this information pass does not qualify a submission.\n'
else:
    report+='\nThis fixed adaptation stops here. No alternate epoch, optimizer, penalty,\nlayer, label, feature or roster rescue, native candidate, panel or upload\nfollows this failed information test.\n'
with (HERE/'RESULT.md').open('x') as f:f.write(report)
external=[p for p in read(HERE/'sources_sealed.json')['pins'] if not Path(p['path']).is_relative_to(HERE)]
external+=bank_pins+[previous['study']];external=list({p['path']:p for p in external}.values())
for p in external:check(p)
write(HERE/'closeout.json',dict(completed=True,negative=not result['passed'],created_at_utc=datetime.now(timezone.utc).isoformat(),
    artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()],external_pins=external,resources=resources,
    candidate=None,official_changed=False,goal_achieved=False,selected_next_numerical_study=None))
print(dict(closeout=pin(HERE/'closeout.json'),resources=resources),flush=True)
