"""Read-only result audit and additive closeout after the fixed controller exits."""
import datetime
import math
from common import *

verify()
controller=read(HERE/'controller_result.json')
assert controller['completed'] and controller['error'] is None
assert controller['no_upload'] and controller['no_automatic_downstream']
check(controller['score'])
score=read(HERE/'information_result.json')
assert score['completed'] and score['passed']==controller['passed']
assert score['exploratory_network_cv'] and score['earlier_parent_training_overlap']
assert score['actual_dual_capture_cost_included'] and not score['deployment_bill_qualified']
assert not score['official_changed'] and not score['goal_achieved']
sealed=read(HERE/'captures_sealed.json')
assert sealed['train']==TRAIN and sealed['targets_decoded'] is False
for p in sealed['pins']:check(p)
assert [r['rank'] for r in score['rows']]==TRAIN
arms=('unchanged','control','extended')
means={a:math.fsum(r[a]['adjusted'] for r in score['rows'])/20 for a in arms}
for a in arms:assert math.isclose(means[a],score['means'][a],rel_tol=1e-14,abs_tol=0)
folds=[{a:math.fsum(r[a]['adjusted'] for r in score['rows'] if r['fold']==f)/4 for a in arms} for f in range(5)]
for f,actual in zip(folds,score['folds']):
    for a in arms:assert math.isclose(f[a],actual[a],rel_tol=1e-14,abs_tol=0)
wins=sum(f['extended']<min(f['unchanged'],f['control']) for f in folds)
passed=means['extended']<=.95*means['unchanged'] and means['extended']<=.97*means['control'] and wins>=4
assert passed==score['passed'] and wins==score['fold_wins']
check(score['predictions_seal'])
oof=read(HERE/'oof_sealed.json');check(oof['predictions'])
assert oof['before_pooled_scoring'] and not oof['hyperparameter_selection']
for f in score['fits']:
    check(f['payload'])
    assert set(f['train']).isdisjoint(f['validation'])
    assert sorted(f['train']+f['validation'])==TRAIN

names=['qualification','qualify']+[f'capture_{r}' for r in TRAIN[1:]]+[f'parent_{r}' for r in PARENT_NEW]+['seal','information']
guards=[clean_watch(n) for n in names]
assert len(guards)==28
calls=[]
for name in ['qualification']+[f'capture_{r}' for r in TRAIN[1:]]+[f'parent_{r}' for r in PARENT_NEW]:
    result=read(HERE/name/'result.json');assert result['completed']
    for r in result['records']:
        for key in ('prediction','bill','weights','input_bill','input_reset','lifecycle'):check(r[key])
        bill=read(r['bill']['path']);life=read(r['lifecycle']['path'])
        assert bill['complete_counted_predict'] and not bill['initialization_subtracted']
        assert bill['flops']==r['flops']==life['closed_metrics']['flops']
        assert life['complete'] and life['context_released'] and life['closed_context_bill_unchanged_by_reset']
        retirement=read(HERE/name/f"call_{r['index']:02d}_retirement.json")
        assert retirement['all_tracked_call_objects_released'] and not retirement['live_objects']
        assert retirement['after_closed_saved_bill'] and retirement['saved_bill_unchanged']
        capture=name=='qualification' or name.startswith('capture_')
        assert retirement['tracked_objects']==(26 if capture else 18)
        if capture:check(r['packet'])
        calls.append(dict(name=name,index=r['index'],rank=r['rank'],capture=capture,flops=r['flops']))
assert len(calls)==27 and sum(r['capture'] for r in calls)==22
processes=subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,pgid=,rss=,command='],text=True,timeout=5)
owned=[line for line in processes.splitlines() if any(HERE.name+'/'+p in line for p in ('run.py','native.py','guard.py','information.py','seal.py','qualify.py'))]
assert not owned,owned
resources=dict(completed=True,guards=len(guards),new_estimator_calls=len(calls),new_capture_calls=22,
    new_parent_calls=5,reused_cold_parent_calls=15,max_rss_bytes=max(g['max_rss_bytes'] for g in guards),
    max_swap_growth_bytes=max(g['max_swap_growth_bytes'] for g in guards),
    summed_guard_wall_s=math.fsum(g['elapsed_s'] for g in guards),all_groups_retired=True,
    remaining_owned_workers=owned,new_native_flops=sum(c['flops'] for c in calls),calls=calls,
    free_disk_bytes=shutil.disk_usage(HERE).free,
    artifact_bytes_before_closeout=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),
    guards_pins=[pin(HERE/f'{n}_watch.json') for n in names])
write(HERE/'resources.json',resources)
decision='PASSED the fixed exploratory information gate' if passed else 'FAILED the fixed exploratory information gate'
next_action=('A separately sealed full-data fit and actual native validation may follow. No production bill or independent validation is established.' if passed else
    'This fixed study ends here. No full-data fit, CHECK target opening, production candidate or submission follows; no fold, ridge, layer or feature rescue was performed.')
report=f'''# Sign-reversed terminal representation: {decision}

The exact official331760 estimator supplied the unchanged trajectory. A second
final scalar/GNN evaluation with sign-reversed pre-mean, third cumulant and
weights supplied64 extra hidden features. The original outputs were unchanged
and the complete additional computation was counted.

Five-fold network CV on the predeclared reused TRAIN20 roster gave extended /
unchanged adjusted error **{score['extended_to_parent']:.12g}**, extended /
affine-control **{score['extended_to_control']:.12g}**, and **{wins}/5** folds
beating both comparators. Gates remain <=0.95, <=0.97, and >=4/5 respectively.
Mean adjusted errors: unchanged {means['unchanged']:.15g}, affine control
{means['control']:.15g}, extension {means['extended']:.15g}.

These are exploratory results on research networks used by previous parent
training, not pristine validation or official scores. The score includes the
actual dual-capture bill plus the previously specified2MF readout reserve;
this is not a measured deployed readout bill. OOF FP32 predictions and fold
payloads were frozen before pooled scoring. Algebra, duplicate-span and
compiled-fit checks passed; an independent scalar aggregation reproduces the
fixed decision.

All20 cold captures match their exact-current-parent outputs, original variance
domains and selectors. The35/100/35 qualification also matches repeated packet
bytes. All27 new estimator calls and28 guards retired, peak
{resources['max_rss_bytes']/1e9:.6f}GB and zero added swap. The15 reused cold
baselines were authenticated separately. Added feature cost is315710544F
(about0.076% of cold35). New artifacts before closeout occupy
{resources['artifact_bytes_before_closeout']/1e6:.3f}MB; free disk is
{resources['free_disk_bytes']/1e9:.3f}GB.

{next_action}

Official331760 remains **3.507738856975084e-9**,100/100 complete with zero
failures. The4e-9 milestone is banked;3e-9,2e-9 and approximately1e-9 remain
unmet. No upload occurred. The broader Phase2 goal remains active.
'''
with (HERE/'RESULT.md').open('x') as f:f.write(report)
artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
write(HERE/'closeout.json',dict(completed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scientific_passed=passed,score=pin(HERE/'information_result.json'),resources=pin(HERE/'resources.json'),
    source_seal=pin(HERE/'sources_sealed.json'),capture_seal=pin(HERE/'captures_sealed.json'),
    report=pin(HERE/'RESULT.md'),closeout_source=pin(Path(__file__)),artifacts=artifacts,
    official_changed=False,goal_achieved=False,new_uploads=0,next_action=next_action))
print(json.dumps(dict(completed=True,passed=passed,parent_ratio=score['extended_to_parent'],
    control_ratio=score['extended_to_control'],fold_wins=wins,
    resources={k:v for k,v in resources.items() if k not in ('calls','guards_pins')}),indent=2))
