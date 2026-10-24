"""Authenticate a complete fixed panel/confirmation without changing decisions."""
from datetime import datetime,timezone
import ast,math,os,shutil,subprocess
from common import *
assert len(sys.argv)==2;controller_session=int(sys.argv[1]);seal_check()
for p in read(HERE/'finalize_sources_sealed.json')['pins']:verify(p)
native=read(NATIVE/'closeout.json')
for p in native['artifacts']+native['external_pins']:verify(p)
assert read(NATIVE/'score.json')['passed'] and not read(FIT/'result.json')['passed']
copies=read(HERE/'source_copies.json')
for row in copies['copies']:
    verify(row['old']);verify(row['new'])
    assert row['byte_exact'] and Path(row['old']['path']).read_bytes()==Path(row['new']['path']).read_bytes()
completed=read(HERE/'completed.json');assert completed['completed'] and not completed['uploaded'] and not completed['release_qualified']
stages=completed['completed_stages'];assert stages in (['panel'],['panel','confirmation'])
assert completed['confirmation_executed']==('confirmation' in stages)
if stages==['panel']:assert not read(HERE/'panel_score.json')['passed']
else:assert read(HERE/'panel_score.json')['passed']
expected_names=[];scores={};calls=[];source_cells=0
for stage in stages:
    roster=PANEL if stage=='panel' else CONFIRM
    expected_names += [f'{stage}_{r}' for r in roster]+[f'{stage}_qualify',f'{stage}_score']
    s=read(HERE/f'{stage}_score.json');q=read(HERE/f'{stage}_qualified.json')
    assert s['completed'] and q['passed'] and not s['official_score'] and not s['pristine_holdout']
    assert s['new_fit_steps']==s['new_samples']==0 and s['old_proxy_gate_unchanged']
    for p in (s['predictions'],s['qualification'],q['predictions']):verify(p)
    assert s['predictions']==q['predictions']
    sealed=read(s['predictions']['path']);assert sealed['before_targets'] and sealed['all_cold'] and sealed['no_new_fit']
    assert sealed['candidate']==pin(CANDIDATE)
    assert [r['rank'] for r in sealed['rows']]==[r['rank'] for r in s['rows']]==roster
    assert all(q[k] for k in ('first8_exact_all','all16_unfloored_variances_positive','cold_initialization_all','targets_not_decoded'))
    assert len(q['source_fidelity'])==7*len(roster)
    assert all(max(r['numpy_relative'],r['fp64_relative'])<=3e-5 for r in q['source_fidelity'])
    source_cells+=len(q['source_fidelity'])
    for row,scored in zip(sealed['rows'],s['rows']):
        for k in ('truth','weights','result','watch'):verify(row[k])
        result=read(row['result']['path']);c=row['call'];p=row['parent']
        assert result['completed'] and result['candidate']==pin(CANDIDATE) and result['records']==[c]
        assert result['full_estimator_calls']==1 and result['capture_additional_fnp_operations']==0 and not result['truth_accessed']
        assert c['includes_initialization'] and p['includes_initialization'] and not c['bank_parameters']
        assert c['weights']==row['weights'] and c['weights']['sha256']==p['weights_sha256']
        assert c['numeric_cache_retired'] and not c['truth_accessed']
        assert len(c['source_checks'])==7 and all(x['full_source_finite_and_symmetric'] for x in c['source_checks'])
        assert len(c['unfloored_variance_minima'])==16 and min(c['unfloored_variance_minima'])>=1e-10
        for item in (c['prediction'],c['capture'],p['prediction'],p['receipt'],p['source']):verify(item)
        for arm,record in (('candidate',c),('parent',p)):
            a=scored[arm]
            assert a['flops']==record['flops'] and a['multiplier']==record['multiplier']==max(.1,a['flops']/2**41)
            assert a['raw']>0 and a['adjusted']==a['raw']*a['multiplier']
        assert scored['raw_ratio']==scored['candidate']['raw']/scored['parent']['raw']
        assert scored['cost_ratio']==c['flops']/p['flops']
        assert scored['adjusted_ratio']==scored['candidate']['adjusted']/scored['parent']['adjusted']
        calls.append(c)
    totals={}
    for arm in ('candidate','parent'):
        total=0.0
        for row in s['rows']:total+=row[arm]['adjusted']
        totals[arm]=total
    assert totals==s['adjusted_totals'] and s['pooled_adjusted_ratio']==totals['candidate']/totals['parent']
    independent=math.fsum(r['candidate']['adjusted'] for r in s['rows'])/math.fsum(r['parent']['adjusted'] for r in s['rows'])
    assert abs(independent-s['pooled_adjusted_ratio'])<=2e-10*independent
    wins=sum(r['adjusted_ratio']<1 for r in s['rows']);worst=max(r['adjusted_ratio'] for r in s['rows']);minimum_wins=8 if stage=='panel' else 5
    assert s['wins']==wins and s['worst_adjusted_ratio']==worst
    assert s['gate']==dict(pooled_adjusted_ratio_less_than=1,minimum_wins=minimum_wins,worst_adjusted_ratio_at_most=1.10)
    assert s['passed']==(s['pooled_adjusted_ratio']<1 and wins>=minimum_wins and worst<=1.10)
    scores[stage]=s
assert completed['guarded_names']==expected_names
passed=stages==['panel','confirmation'] and scores['confirmation']['passed'];assert completed['passed']==passed
guards=[done(n) for n in expected_names];groups=sorted(g['pid'] for g in guards);remaining=[];workers=[]
for line in subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,rss=,command='],text=True).splitlines():
    t=line.strip().split(None,3)
    if len(t)!=4:continue
    if int(t[0]) in groups or int(t[1]) in groups:remaining.append(line)
    if int(t[0])!=os.getpid() and any(s in t[3] for s in (str(RUNTIME),str(D/'run_guarded.py'),str(HERE/'run.py'))):workers.append(line)
assert not remaining and not workers,(remaining,workers)
bank=read(D/'official_best_331684.json');bank_pins=[v for v in bank.values() if isinstance(v,dict) and {'path','sha256','bytes'}<=v.keys()]
assert len(bank_pins)==5
for p in bank_pins:verify(p)
artifact_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file());assert artifact_bytes<20_000_000
resources=dict(guarded_groups=len(guards),all_groups_retired=True,controller_terminal_session=controller_session,
    full_native_calls=len(calls),full_native_flops=sum(c['flops'] for c in calls),
    peak_group_rss_bytes=max(g['max_rss_bytes'] for g in guards),peak_self_reported_rss_bytes=max(c['peak_rss_bytes'] for c in calls),
    max_swap_growth_bytes=0,new_fit_steps=0,new_samples=0,new_downloads=0,GPU_calls=0,
    existing_1B_reference_networks=len(calls),artifact_bytes_before_closeout=artifact_bytes,disk_free_bytes=shutil.disk_usage(HERE).free)
write(HERE/'resources.json',resources)
summary=dict(completed=True,passed=passed,negative=not passed,completed_stages=stages,candidate=pin(CANDIDATE),
    development=pin(NATIVE/'closeout.json'),development_pooled_adjusted_ratio=read(NATIVE/'score.json')['pooled_adjusted_ratio'],
    scores={k:{x:v[x] for x in ('pooled_adjusted_ratio','wins','worst_adjusted_ratio','passed','rows')} for k,v in scores.items()},
    source_cells_verified=source_cells,resources=resources,old_proxy_gate_still_failed=True,no_parameter_change_or_new_fit=True,
    release_qualified=False,official_changed=False,goal_achieved=False,
    next_stage='real_client_prediction_only_runtime_then_contract_and_package' if passed else None)
write(HERE/'summary.json',summary)
report='# Unchanged nonlinear edge payload: fixed panel results\n\n'
report+='The native development ratio was .9935549509494914. The identical final528-parameter payload was carried into the predeclared panel without fitting or tuning. The old checkpoint81 proxy gates remain failed. These networks were excluded from that fit but have prior research use; they are not pristine holdouts or official evaluation.\n\n'
for stage,s in scores.items():
    report+=f"{stage.capitalize()}: pooled adjusted ratio **{s['pooled_adjusted_ratio']:.12f}**, {s['wins']}/{len(s['rows'])} wins, worst {s['worst_adjusted_ratio']:.12f}; passed={s['passed']}.\n\n"
    report+='|Network|Raw / bank|Full cold cost / bank|Adjusted / bank|\n|---|---:|---:|---:|\n'
    for row in s['rows']:report+=f"|{row['rank']}|{row['raw_ratio']:.12f}|{row['cost_ratio']:.12f}|{row['adjusted_ratio']:.12f}|\n"
    report+='\n'
report+='Panel gate: pooled<1, at least8/14 wins, worst<=1.10. Confirmation gate: pooled<1, at least5/8 wins, worst<=1.10. The complete roster predictions and each arm\'s own cold cost were sealed before ordinary1B targets. Pooled scores average actual adjusted losses. Every fresh call includes module/static setup. The first8 outputs match the bank exactly; all16 unfloored variance vectors are positive and every full source is finite and symmetric. Saved pair-local sources pass independent FP32/FP64 references. Numerical caches retire.\n\n'
report+='Four parent records use the exact packed official source. Other bank records use the accepted pre-packing source with the authenticated lossless packing build/unit/native certificates. The four overlapping outputs and complete bills are exact, and both source identities are explicit. The native harness and its tiny-unit reference/qualification were copied byte-for-byte from development. A preparation-only missing receipt key was corrected by reading and checking the direct run receipts; the original preparation script is preserved. No native call or score preceded that correction.\n\n'
report+=f"{len(guards)} guarded groups and {len(calls)} full native calls retired; controller{controller_session} terminal. Total {resources['full_native_flops']} F, peak sampled RSS{resources['peak_group_rss_bytes']} B, self-report{resources['peak_self_reported_rss_bytes']} B, zero added swap. {source_cells} source cells verified. Artifacts {artifact_bytes/1e6:.3f} MB before closeout. No new fit, sampling, download or GPU work.\n\n"
if passed:report+='Both accuracy stages passed. Real-client prediction-only runtime, contract and package qualification remain required before release. No upload has occurred in this study.\n\n'
else:report+='The fixed payload is closed at its failed validation gate. No further confirmation after a panel failure, fit, amplitude/epoch/layer/precision alternative, package or upload follows this arm. Its positive development result remains preserved; it does not qualify release.\n\n'
report+='Official331684 remains adjusted3.5844144212297072e-9, raw1.949715441540434e-8, 100/100 complete and zero failures.4e-9 is banked;3/2/~1e-9 remain unmet. The overall goal remains active and qualified uploads remain authorized.\n'
with (HERE/'RESULT.md').open('x') as f:f.write(report)
external=[p for p in read(HERE/'sources_sealed.json')['pins'] if not Path(p['path']).is_relative_to(HERE)]+bank_pins
external=list({p['path']:p for p in external}.values())
for p in external:verify(p)
write(HERE/'closeout.json',dict(completed=True,negative=not passed,created_at_utc=datetime.now(timezone.utc).isoformat(),
    artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()],external_pins=external,resources=resources,
    candidate=pin(CANDIDATE),development=pin(NATIVE/'closeout.json'),release_qualified=False,official_changed=False,goal_achieved=False,
    old_proxy_gate_still_failed=True,no_parameter_change_or_new_fit=True,next_stage=summary['next_stage']))
print(dict(closeout=pin(HERE/'closeout.json'),passed=passed,resources=resources),flush=True)
