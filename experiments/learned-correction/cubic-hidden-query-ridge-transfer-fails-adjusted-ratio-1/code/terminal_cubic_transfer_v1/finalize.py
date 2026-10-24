"""Authenticate the frozen transfer result; no further fits or array execution."""
import ast,os,shutil,subprocess
from datetime import datetime,timezone
from common import *
seal_check()
for name in ['repair_sources_sealed.json','finalize_sources_sealed.json']:
    for p in read(HERE/name)['pins']:check(p)
complete=read(HERE/'completed.json');score=read(HERE/'cv_result.json')
assert complete['completed'] and complete['passed']==score['passed']
inputs=read(HERE/'inputs.json');train=inputs['train']
features=read(HERE/'features_sealed.json');oof=read(HERE/'oof_predictions_sealed.json')
for p in features['pins']:check(p)
check(oof['feature_seal'])
for p in oof['predictions'].values():check(p)
assert len(features['records'])==23 and len(oof['models'])==10 and len(train)==20
assert features['all16_parent_bytes_and_full_bill_exact'] and features['repeat35_features_and_bill_exact']
assert features['old35_and100_feature_parity'] and features['targets_not_yet_decoded']
assert oof['all_held_predictions_frozen_before_scoring'] and oof['each_model_uses_only_fit_targets']
for model in oof['models']:
    check(model['parameters']);fold=model['fold']
    assert model['fit']==[r for i,r in enumerate(train) if i%5!=fold]
    assert model['held']==[r for i,r in enumerate(train) if i%5==fold]
    assert model['normal_residual']<=1e-9 and model['cholesky_relative']<=1e-8
for row in inputs['rows']:
    for k in ['weights','truth','parent_prediction','parent_result']:check(row[k])
    if 'old_feature' in row:check(row['old_feature']['file'])
assert score['helper_bill_measured'] and not score['full_native_candidate_bill_measured']
assert not score['pristine_holdout'] and not score['official_score'] and score['independent_fsum_agreed']
records={r['rank']:r for r in features['records'] if r['name'].startswith('train')}
assert [r['rank'] for r in score['rows']]==train
for i,row in enumerate(score['rows']):
    assert row['fold']==i%5
    record=records[row['rank']];s=row['scores']
    assert s['parent']['accounted_flops']==record['parent_flops']
    assert s['affine']['accounted_flops']==record['parent_flops']+2_000_000
    assert s['extended']['accounted_flops']==record['parent_flops']+record['helper_flops']+2_000_000
    for value in s.values():assert value['adjusted']==value['raw']*max(.1,value['accounted_flops']/2**41)
assert score['passed']==(score['pooled_adjusted_parent_ratio']<=.99 and score['pooled_adjusted_affine_ratio']<=.99
    and score['network_wins']>=12 and score['fold_wins']>=4 and score['worst_adjusted_ratio']<=1.20)
proofs=read(HERE/'repair_build.json')['proofs']
assert len(proofs)==5
for proof in proofs:
    check(proof['original']);check(proof['derived']);source=Path(proof['derived']['path']).read_text()
    for before,after,count in reversed(proof['edits']):
        assert source.count(after)==count;source=source.replace(after,before)
    assert source==Path(proof['original']['path']).read_text()
assert (HERE/'helper.py').read_bytes()==(D/'terminal_cubic_hidden_queries_v1/helper.py').read_bytes()
assert (HERE/'unit.py').read_bytes()==(D/'terminal_cubic_hidden_queries_v1/unit.py').read_bytes()
unit=read(HERE/'unit.json');solver_unit=read(HERE/'solver_unit_v2.json')
assert unit['passed'] and len(unit['rows'])==3 and solver_unit['passed']
assert max(r['flops'] for r in solver_unit['wide_readout_bills'])<=2_000_000
guards=[(p.name,read(p)) for p in sorted(HERE.glob('*_watch.json'))]
assert len(guards)==30,len(guards)
for name,g in guards:
    assert g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    if name=='solver_unit_watch.json':assert g['exit_code']==1
    else:assert g['completed'] and g['exit_code']==0,name
groups=sorted(g['pid'] for _,g in guards);remaining=[];workers=[]
markers=[str(RUNTIME),str(D/'run_guarded.py'),str(HERE/'run.py'),str(HERE/'run_v2.py')]
for line in subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,rss=,command='],text=True).splitlines():
    t=line.strip().split(None,3)
    if len(t)!=4:continue
    if int(t[0]) in groups or int(t[1]) in groups:remaining.append(line)
    if int(t[0])!=os.getpid() and any(m in t[3] for m in markers):workers.append(line)
assert not remaining and not workers,(remaining,workers)
bank=read(D/'official_best_331684.json')
bank_pins=[p for p in bank.values() if isinstance(p,dict) and {'path','sha256','bytes'}<=p.keys()]
assert len(bank_pins)==5
for p in bank_pins:check(p)
previous=read(D/'continuation_checkpoint_v78.json');check(previous['study']);check(previous['official_bank'])
artifact_bytes=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file());assert artifact_bytes<50_000_000
resources=dict(guarded_groups=len(guards),failed_fixture_groups=1,groups=groups,all_groups_retired=True,
    terminal_sessions=[65020,88499],full_native_calls=23,
    full_native_flops=sum(r['parent_flops'] for r in features['records']),
    feature_flops=sum(r['helper_flops'] for r in features['records']),
    peak_group_rss_bytes=max(g['max_rss_bytes'] for _,g in guards),
    peak_self_reported_rss_bytes=max([r['peak_rss_bytes'] for r in features['records']]
        +[unit['peak_rss_bytes'],solver_unit['peak_rss_bytes'],oof['peak_rss_bytes'],score['peak_rss_bytes'],read(HERE/'solver_diagnostic.json')['peak_rss_bytes']]),
    max_swap_growth_bytes=0,training_networks=20,qualification_networks=2,cross_validation_fits=10,
    new_samples=0,new_downloads=0,GPU_calls=0,artifact_bytes_before_closeout=artifact_bytes,
    disk_free_bytes=shutil.disk_usage(HERE).free)
assert max(resources['peak_group_rss_bytes'],resources['peak_self_reported_rss_bytes'])<8_000_000_000
write(HERE/'resources.json',resources)
write(HERE/'summary.json',dict(completed=True,negative=not score['passed'],score=score,resources=resources,
    previous_opportunity_gate_still_failed=True,previous_stop_decision_explicitly_superseded=True,
    feature_family_unchanged=True,oracle_coefficients_used=False,all_parent_bytes_and_bills_exact=True,
    official_changed=False,goal_achieved=False,next_numerical_study=None))
status='passes the fixed transfer gate; native validation remains' if score['passed'] else 'fails the fixed transfer gate'
report=f'''# Terminal cubic transfer: {status}

The unchanged65 cubic queries from terminal_cubic_hidden_queries_v1 were
tested with fixed cross-network ridge regression. The older per-network
oracle still fails its .80 opportunity gates. This study explicitly revised
the decision to stop there, because a smaller transferable, paid gain would
still serve the user's objective. It did not alter those results or use
privileged oracle coefficients. No new feature family is claimed.

Twenty existing research networks were split into five fixed network folds,
16 fit and4 held per model. Each fold fitted the65-column affine control and
the130-column extended design with fit-only weighted RMS normalization and
ridge1e-3. Every model's served networks are absent from its training labels.
All held predictions were sealed before the separate scoring stage. These
reused research networks are not pristine holdouts or an official score.

- Pooled adjusted extended/parent: {score['pooled_adjusted_parent_ratio']:.12f} (required<=.99).
- Pooled adjusted extended/affine: {score['pooled_adjusted_affine_ratio']:.12f} (required<=.99).
- Individual parent wins: {score['network_wins']}/20 (required>=12).
- Parent-improving folds: {score['fold_wins']}/5 (required>=4).
- Worst individual adjusted ratio: {score['worst_adjusted_ratio']:.12f} (required<=1.20).

The complete cold parent bill and the actual measured helper cost are paid
for each network, plus a provisional2MF readout allowance. The affine control
pays parent+2MF. The isolated wide extended readout costs1206272 F and the
affine readout673792 F, within that allowance. Full native candidate bills
and runtime remain unqualified. All raw errors agree with independent fsum.

All23 cold parent calls reproduce all16 bank output rows and complete bills
exactly, and retire original caches. Qualification35/100 reproduces the
earlier feature bytes/bills; repeated35 also matches all compact fields and
bills. All wide queries pass independent FP64 source reconstruction at2e-10
relative and formal/native third diagonal comparison at1e-3. Only compact
features and predictions are retained, with no wide trajectory files.
The unchanged three tensor fixtures run twice each and pass. Four ridge
fixtures pass independent augmented least squares; all10 wide fits pass
normal-equation and independent Cholesky coefficient checks. FNP readout
predictions and bills repeat, including zero-coefficient parent identity.

The first solver fixture failed before native work: it compared unscaled
coefficients of an identically zero feature, amplifying harmless augmented-
solve roundoff by the1e-12 scale floor. The preserved diagnostic establishes
scaled agreement<=6.27e-14 and prediction agreement<=1.28e-15. Additive v2
checks both in meaningful coordinates at the original2e-10 tolerance.
No solver, penalty, feature, input, cost or score gate changed. All original
sources, failure and five exact wrapper inverses are preserved.

Resources:23 full parent calls, {resources['full_native_flops']} F, plus
{resources['feature_flops']} F of measured helpers;10 CV fits. All30 guarded
groups, including the failed fixture, are retired. Controllers65020/88499
are terminal. Peak sampled RSS{resources['peak_group_rss_bytes']} B,
self-report{resources['peak_self_reported_rss_bytes']} B; zero added swap.
Fast-stage samples may understate peaks. Artifacts before closeout total
{artifact_bytes/1e6:.3f} MB. No new samples, downloads or GPU calls.
'''
if score['passed']:
    report+='\nThis licenses a separately specified all20 fit, CHECK and actual native integration, followed by panel, confirmation, client/runtime, contract and package qualification. It does not yet authorize a claim of improved official performance.\n'
else:
    report+='\nClose this fixed transfer test. No all-training fit, CHECK, alternate penalty, control-only deployment, shrinkage, feature/layer/nonlinear rescue, native candidate, panel, package or upload follows. This is now an actual cross-network prediction result, with its declared cost allowance, rather than only an oracle opportunity screen.\n'
report+='\nOfficial331684 remains adjusted3.5844144212297072e-9,100/100 complete,zero failures. The4e-9 milestone is banked;3/2/~1e-9 remain unmet and the goal remains active. Qualified future uploads remain user-authorized.\n'
with (HERE/'RESULT.md').open('x') as f:f.write(report)
external=[p for p in read(HERE/'sources_sealed.json')['pins'] if not Path(p['path']).is_relative_to(HERE)]
for row in inputs['rows']:
    external += [row[k] for k in ['weights','truth','parent_prediction','parent_result']]
    if 'old_feature' in row:external.append(row['old_feature']['file'])
external += bank_pins+[previous['study']]
external=list({p['path']:p for p in external}.values())
for p in external:check(p)
artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file()]
write(HERE/'closeout.json',dict(completed=True,negative=not score['passed'],created_at_utc=datetime.now(timezone.utc).isoformat(),
    artifacts=artifacts,external_pins=external,resources=resources,candidate=None,official_changed=False,
    goal_achieved=False,previous_oracle_gate_still_failed=True,stopping_decision_explicitly_revised=True,next_numerical_study=None))
print(dict(closeout=pin(HERE/'closeout.json'),artifacts=len(artifacts),external=len(external),resources=resources),flush=True)
