"""Close a retired, independently audited study without relabeling its gates."""
import datetime,shutil
from common import HERE,D,read,write,pin,verify,seal_check

seal_check();outcome=read(HERE/'outcome.json');assert read(HERE/'fit_audit.json')['passed']
controller=read(HERE/'controller_watch.json')
assert controller['completed'] and controller['whole_group_retired']
guards={p.name:read(p) for p in HERE.glob('*_watch.json')}
assert all(g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0 for g in guards.values())
assert all(g['max_rss_bytes']<8_000_000_000 for g in guards.values())
for record in ['release_prospective_sources.json','fit_audit_prospective_source.json']:
    r=read(HERE/record)
    for p in r.get('pins',[r.get('source')]):
        if p:verify(p)
native=[];clients=[]
for p in HERE.glob('*/result.json'):
    r=read(p)
    if 'records' in r:native+=r['records']
for p in HERE.glob('*/completed.json'):
    r=read(p)
    if 'records' in r:clients+=r['records']
assert len(native)+len(clients)<=120
assert all(r['max_rss_bytes']<8_000_000_000 for r in native)
leaf=[g for k,g in guards.items() if k not in ['controller_watch.json','release_controller_watch.json']]
resources=dict(guarded_groups=len(guards),leaf_guarded_groups=len(leaf),all_groups_retired=True,
    peak_group_rss_bytes=max(g['max_rss_bytes'] for g in guards.values()),max_swap_growth_bytes=0,
    leaf_elapsed_s=sum(g['elapsed_s'] for g in leaf),controller_elapsed_s=controller['elapsed_s'],
    native_full_calls=len(native),native_full_flops=sum(r['flops'] for r in native),
    peak_native_self_rss_bytes=max(r['max_rss_bytes'] for r in native),
    client_full_calls=len(clients),client_full_flops=sum(r['summary']['flops_used'] for r in clients),
    fit_iterations=36,loss_evaluations=72,fit_parameter_count=16,new_samples=0,new_dataset_downloads=0,GPU_calls=0,
    disk_free_bytes=shutil.disk_usage(HERE).free,artifact_bytes_before_closeout=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()))
assert resources['artifact_bytes_before_closeout']<150_000_000
write(HERE/'resources.json',resources)
release=read(HERE/'release_outcome.json') if (HERE/'release_outcome.json').exists() else None
uploaded=(HERE/'submission_attempt.json').exists() and read(HERE/'submission_attempt.json')['exit_code']==0
stages={stage:read(HERE/f'{stage}_score.json') for stage in ['train','check','panel','confirmation'] if (HERE/f'{stage}_score.json').exists()}
lines=['# Fixed direct native covariance-head optimization','',
    'Parent331684 is the official3.5844144212297072e-9 bank,100/100 completed.',
    'One36-update projected SPSA run changed only16 FP32 output coefficients.',
    'Every update used an actual full-estimator plus/minus loss pair on the',
    'declared six TRAIN networks, with own cold FLOP bills. Deployment uses',
    'the fixed average of the final18 centers; no checkpoint or gain selection.',
    'Independent scalar replay verifies all72 losses, every update and the',
    'terminal FP32 head. The earlier linear-response failure remains closed.','']
for stage,r in stages.items():
    lines.append(f"- {stage}: adjusted ratio{r['pooled_adjusted_ratio']:.12g},wins{r['wins']}/{len(r['rows'])},worst{r['worst_adjusted_ratio']:.12g},passed={r['passed']}.")
lines+=['',f"Accuracy outcome: {outcome}. Release outcome: {release}.",
    'These reused research networks are not pristine holdouts or official scores.',
    'Failed gates remain failed; no follow-up tuning is authorized by a failure.',
    f"Native calls{len(native)}; client calls{len(clients)}; all{len(guards)} groups retired,zero added swap.",
    f"Peak sampled group RSS{resources['peak_group_rss_bytes']}B. No new samples,datasets or GPU.",
    f"Uploaded={uploaded}; full approximately1e-9 goal remains unachieved.",'']
with (HERE/'RESULT.md').open('x') as f:f.write('\n'.join(lines))
files=sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='finalize.log')
record=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accuracy_outcome=outcome,
    release_outcome=release,uploaded=uploaded,goal_achieved=False,resources=resources,
    artifact_pins=[pin(p) for p in files],external_pins=[pin(D/'official_best_331684.json'),pin(D/'covariance_head_native_response_v1/closeout_after_replay_v1.json')])
for p in record['artifact_pins']+record['external_pins']:verify(p)
write(HERE/'closeout.json',record)
print(dict(artifact_pins=len(files),outcome=outcome,release=release,resources=resources),flush=True)
