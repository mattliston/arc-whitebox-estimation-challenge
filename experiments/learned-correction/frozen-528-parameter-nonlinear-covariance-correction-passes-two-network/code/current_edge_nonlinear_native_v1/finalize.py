"""Authenticate the completed native development endpoint without changing it."""
from datetime import datetime,timezone
import ast,math,os,shutil,subprocess
from common import *
seal_check()
for p in read(HERE/'finalize_sources_sealed.json')['pins']:verify(p)
completed=read(HERE/'completed.json');score=read(HERE/'score.json');q=read(HERE/'qualification.json');build=read(HERE/'build.json')
assert completed['completed'] and score['completed'] and q['passed'] and completed['passed']==score['passed']
assert score['old_proxy_gate_unchanged'] and score['stopping_decision_explicitly_revised']
assert score['new_fit_steps']==score['new_samples']==0 and not score['official_score']
assert not read(FIT/'result.json')['passed'] and read(FIT/'closeout.json')['negative']
for k in ('parent','candidate','parameters','bank_literal','fitted_parameters','fit_receipt'):verify(build[k])
candidate=Path(build['candidate']['path']).read_text();original=build['replaced_assignment'];new=build['new_assignment']
assert candidate.count(new)==1 and candidate.replace(new,original,1)==PARENT.read_text()
assert build['whole_source_inverse_exact'] and build['all_normalizers_bit_exact'] and build['changed_parameter_scalars']==528 and build['total_edge_scalars']==545
for c in read(HERE/'source_copies.json')['copies']:
    verify(c['old']);verify(c['new']);source=Path(c['new']['path']).read_text()
    for a,b in reversed(c['edits']):assert source.count(b)==1;source=source.replace(b,a,1)
    assert source==Path(c['old']['path']).read_text()
assert all(q[k] for k in ('bank_triple_all16_bytes_and_bills_exact','enabled_first8_exact','repeat36_exact','warm_cold38_exact','all_source_repeats_exact','all16_unfloored_variances_positive'))
assert q['full_native_calls']==7 and len(q['fidelity'])==49
assert all(max(r['numpy_relative'],r['fp64_relative'])<=3e-5 for r in q['fidelity'])
names=completed['guarded_names'];assert names==['build','unit','bank_triple','enabled','cold38','qualify','score']
guards=[done(n) for n in names];calls=[]
for name in ('bank_triple','enabled','cold38'):
    record=read(HERE/name/'result.json');assert record['completed'] and record['candidate']==build['candidate']
    assert record['capture_additional_fnp_operations']==0 and not record['truth_accessed']
    calls+=record['records']
assert len(calls)==7 and sum(c['flops'] for c in calls)==q['full_native_flops']
for c in calls:
    verify(c['prediction']);verify(c['capture']);verify(c['weights'])
    assert c['numeric_cache_retired'] and len(c['unfloored_variance_minima'])==16 and min(c['unfloored_variance_minima'])>=1e-10
assert [r['rank'] for r in score['rows']]==[36,38]
for r in score['rows']:
    for a in ('candidate','parent'):
        x=r[a];assert x['multiplier']==max(.1,x['flops']/2**41) and x['adjusted']==x['raw']*x['multiplier']
    assert r['raw_ratio']==r['candidate']['raw']/r['parent']['raw']
    assert r['cost_ratio']==r['candidate']['flops']/r['parent']['flops']
    assert r['adjusted_ratio']==r['candidate']['adjusted']/r['parent']['adjusted']
numerator=denominator=0.
for r in score['rows']:numerator+=r['candidate']['adjusted'];denominator+=r['parent']['adjusted']
assert score['pooled_adjusted_ratio']==numerator/denominator
independent=math.fsum(r['candidate']['adjusted'] for r in score['rows'])/math.fsum(r['parent']['adjusted'] for r in score['rows'])
assert abs(independent-score['pooled_adjusted_ratio'])<=2e-10*independent
assert score['passed']==(score['pooled_adjusted_ratio']<1 and all(r['adjusted_ratio']<=1.05 for r in score['rows']))
groups=sorted(g['pid'] for g in guards);remaining=[];workers=[]
for line in subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,rss=,command='],text=True).splitlines():
    t=line.strip().split(None,3)
    if len(t)!=4:continue
    if int(t[0]) in groups or int(t[1]) in groups:remaining.append(line)
    if int(t[0])!=os.getpid() and any(s in t[3] for s in (str(RUNTIME),str(D/'run_guarded.py'),str(HERE/'run.py'))):workers.append(line)
assert not remaining and not workers,(remaining,workers)
previous=read(D/'continuation_checkpoint_v81.json');verify(previous['study']);verify(previous['official_bank'])
bank=read(D/'official_best_331684.json');bank_pins=[v for v in bank.values() if isinstance(v,dict) and {'path','sha256','bytes'}<=v.keys()]
assert len(bank_pins)==5
for p in bank_pins:verify(p)
unit=read(HERE/'unit.json');assert unit['passed']
artifact_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file());assert artifact_bytes<20_000_000
resources=dict(guarded_groups=7,all_groups_retired=True,controller_terminal_session=19177,full_native_calls=7,full_native_flops=q['full_native_flops'],
    peak_group_rss_bytes=max(g['max_rss_bytes'] for g in guards),
    peak_self_reported_rss_bytes=max([x['peak_rss_bytes'] for x in calls]+[build['peak_rss_bytes'],unit['peak_rss_bytes'],q['peak_rss_bytes'],score['peak_rss_bytes']]),
    max_swap_growth_bytes=0,new_fit_steps=0,new_samples=0,new_downloads=0,GPU_calls=0,existing_1B_reference_networks=2,
    artifact_bytes_before_closeout=artifact_bytes,disk_free_bytes=shutil.disk_usage(HERE).free)
write(HERE/'resources.json',resources)
summary=dict(completed=True,development_passed=score['passed'],negative=not score['passed'],candidate=build['candidate'],
    old_proxy_gate_still_failed=True,stopping_decision_explicitly_revised=True,unchanged_fitted_payload=build['fitted_parameters'],
    pooled_adjusted_ratio=score['pooled_adjusted_ratio'],rows=score['rows'],parent_parameter_triple_exact=True,
    whole_source_inverse_exact=True,unchanged_edge_helper_code=True,all_normalizers_bit_exact=True,
    native_full_calls=7,source_cells_verified=49,all16_unfloored_variances_positive=True,
    resources=resources,official_changed=False,goal_achieved=False,release_qualified=False,
    next_stage='fixed14_network_panel' if score['passed'] else None)
write(HERE/'summary.json',summary)
verdict='positive development result' if score['passed'] else 'negative development result'
report=f'''# Fixed nonlinear edge payload: {verdict}

The unchanged final528-parameter nonlinear payload from checkpoint81 was
tested through the complete current native estimator. This explicitly
supersedes only the decision not to measure that endpoint. Its earlier
covariance/response/control proxy gates remain failed; no old threshold,
fit, coefficient, source or result changed. No new training or tuning.

|Network|Raw error / bank|Full cold cost / bank|Adjusted error / bank|
|---|---:|---:|---:|
'''
for r in score['rows']:report+=f"|{r['rank']}|{r['raw_ratio']:.12f}|{r['cost_ratio']:.12f}|{r['adjusted_ratio']:.12f}|\n"
report+=f'''
Pooled adjusted ratio is **{score['pooled_adjusted_ratio']:.12f}**. The fixed
development gate requires pooled<1 and each<=1.05; passed={score['passed']}.
Each score uses that arm's actual complete cold bill and its own final mean.
The ordinary1B targets36/38 were decoded only after qualified predictions
and bills were sealed. These networks are outside the payload's six TRAIN
and two VALID networks, but have previous research use. They are not
pristine holdouts or official evaluation scores.

Only the _EDGE_LITERAL assignment changed in the exact official packed
source. Replacing it with the original restores the entire parent byte-for-
byte. All17 normalizer/factor constants and the helper, architecture, layer
range, physical machinery, kernels and other learned heads are unchanged.
Their later states, supports and bill were allowed to respond naturally.
The research-only bank mode restores the original literal and keeps the
edge enabled. It reproduces all16 original outputs and each complete
cold/warm bill on35/100/35. Adapted36/38/36 and separate cold38 preserve
the first8 parent rows and repeat all own outputs and source samples exactly.

Actual-FNP tiny fixtures pass FP32/FP64 references, symmetry/permutation,
zero output-head, repeat and cold-loading checks. All49 saved pair-local
source cells pass independent NumPy/FP64 checks. Every full correction is
finite and symmetric; all16 unfloored prevariance vectors in all seven
calls are finite and>=1e-10. Parent numerical caches retire. Passive capture
adds no FNP arithmetic. These research wall times do not qualify production
residual time. The whole-source inverse and two copied-source inverses pass.

Seven guarded groups and seven full native calls completed; controller19177
terminal. Total native work{resources['full_native_flops']} F. Peak sampled
RSS{resources['peak_group_rss_bytes']} B, self-report
{resources['peak_self_reported_rss_bytes']} B,zero added swap. About
{artifact_bytes/1e6:.3f} MB artifacts before closeout. No new fit, samples,
downloads or GPU. Prior study and all five official bank files reverified.

Official331684 remains adjusted3.5844144212297072e-9,raw1.949715441540434e-8,
100/100 complete,zero failures.4e-9 banked;3/2/~1e-9 unmet. The full goal
remains active and qualified uploads remain authorized.
'''
if score['passed']:
    report+='\nProceed to the prospectively fixed14-network panel, then8-network confirmation\nand real-client/runtime/contract/package qualification if each passes.\nThis development result alone does not qualify release.\n'
else:
    report+='\nThis fixed payload stops at the actual native development gate. No panel,\nconfirmation, scaling, parameter/epoch/layer variant, refit, package or upload\nfollows. The prior positive proxy effects did not qualify this endpoint.\n'
with (HERE/'RESULT.md').open('x') as f:f.write(report)
external=[p for p in read(HERE/'sources_sealed.json')['pins'] if not Path(p['path']).is_relative_to(HERE)]
external+=bank_pins+[previous['study']];external=list({p['path']:p for p in external}.values())
for p in external:verify(p)
write(HERE/'closeout.json',dict(completed=True,negative=not score['passed'],created_at_utc=datetime.now(timezone.utc).isoformat(),
    artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()],external_pins=external,resources=resources,
    candidate=build['candidate'],release_qualified=False,official_changed=False,goal_achieved=False,
    old_proxy_gate_still_failed=True,stopping_decision_explicitly_revised=True,
    next_stage='fixed14_network_panel' if score['passed'] else None))
print(dict(closeout=pin(HERE/'closeout.json'),development_passed=score['passed'],resources=resources),flush=True)
