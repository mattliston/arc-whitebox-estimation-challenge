import datetime,shutil,subprocess
from common import *
assert read(HERE/'completed.json')['completed'];assert read(HERE/'audit_result.json')['passed']
verify();verify('captures_sealed.json');verify('statistics_sealed.json');verify('audit_sources_sealed.json')
result=read(HERE/'result.json');watch=[(p,read(p)) for p in sorted(HERE.glob('*_watch.json'))]
assert watch and all(r['completed'] and r['whole_group_retired'] and r['max_swap_growth_bytes']==0 for _,r in watch)
ps=subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,rss=,command='],text=True)
groups={r['pid'] for _,r in watch};live=[line for line in ps.splitlines() if len(line.split())>=3 and int(line.split()[1]) in groups]
assert not live,live
calls=[]
for name in ['qualification']+[f'capture_{r:04d}' for r in TRAIN[2:]+VALID]:calls.extend(read(HERE/name/'result.json')['records'])
assert len(calls)==9
resources=dict(guarded_groups=len(watch),all_groups_retired=True,full_native_calls=len(calls),full_native_flops=sum(r['flops'] for r in calls),peak_group_rss_bytes=max(r['max_rss_bytes'] for _,r in watch),max_swap_growth_bytes=max(r['max_swap_growth_bytes'] for _,r in watch),new_samples=0,new_downloads=0,GPU_calls=0,fitted_parameter_vectors=1,fitted_parameters=16,ordinary_1B_target_reads=0,artifact_bytes_before_closeout=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()),disk_free_bytes=shutil.disk_usage(HERE).free)
write(HERE/'resources.json',resources)
valid=[r for r in result['rows'] if r['role']=='VALID']
table='\n'.join(f"|{r['rank']}|{r['pooled']['cov']:.12f}|{r['pooled']['value']:.12f}|{r['passed']}|" for r in valid)
status='PASSED the fixed proxy gate' if result['passed'] else 'CLOSED: fixed proxy gate failed'
decision=('The single payload may proceed only to a separately sealed native finite-endpoint protocol. No gradient, adjusted-score or upload qualification follows from this proxy.' if result['passed'] else 'No native candidate, changed-trajectory replay,1B CHECK/panel scoring, package or upload follows. No penalty, hidden-feature, roster, layer or label rescue is selected.')
text_report=f'''# Current covariance-innovation supervision: {status}

One sixteen-coefficient shared ridge fit used the exact official331684
frozen trajectory and existing coherent8M reference means/covariances.
The target subtracts the declared Gaussian-gain transport of previous
measured errors from each current covariance error. It is an approximate
local innovation, not an exact cumulant source or full-program derivative.

|VALID rank|Covariance residual / baseline|Next-mean response / baseline|Pass|
|---|---:|---:|---|
{table}

Each held network required covariance<=.98 AND response<=.95 across layers8..14.
VALID400/500 is excluded from this fit, but has prior research use. These
point estimates retain8M sampling noise and do not measure actual final
MSE or the complete adjusted objective. {decision}

The original edge model used older oracle-input rows; the old signed matrix
observer used final means. This direct covariance-supervision test therefore
answers a different question. The failed native response, precision and
SPSA studies remain unchanged. No smooth native-gradient claim is made.

Two whole-function source inverses and actual-FNP small tests pass. Native
35/100/35 reproduces all16 parent bytes and full cold/warm bills; every
captured repeated35 array is exact. Six further target-free cold captures
have the same zero-arithmetic taps but no independent uninstrumented replay.
All eight unique captures were sealed before target decoding. Every actual
wide original-head reconstruction passed<=3e-6, and source affine checks
passed<=1e-4. Independent Cholesky solution reconstruction reproduces the
rounded payload exactly and all loss/gate calculations. One fit, no search.

Nine native calls used {resources['full_native_flops']} FLOPs.
{resources['guarded_groups']} guarded groups retired; maximum sampled RSS
{resources['peak_group_rss_bytes']} bytes, zero added swap. New artifacts
before closeout total{resources['artifact_bytes_before_closeout']} bytes.
No new samples, downloads,GPU or ordinary1B means. Official331684 remains
adjusted3.5844144212297072e-9,100/100 complete,zero failures.4e-9 is banked;
3/2/~1e-9 remain unmet and the original goal stays active.
'''
with (HERE/'RESULT.md').open('x') as f:f.write(text_report)
artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='closeout.json']
external=read(HERE/'sources_sealed.json')['pins']
external=[x for x in external if not Path(x['path']).is_relative_to(HERE)]
for p in artifacts+external:assert sha(p['path'])==p['sha256']
write(HERE/'closeout.json',dict(completed=True,closed_negative=not result['passed'],proxy_passed=result['passed'],created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),artifacts=artifacts,external_pins=external,resources=resources,goal_achieved=False,official_changed=False))
print(__import__('json').dumps(dict(proxy_passed=result['passed'],artifacts=len(artifacts),external_pins=len(external),resources=resources)),flush=True)
