"""Close the frozen-model accuracy study after its packed derivative is submitted."""
import datetime,shutil
from common import HERE,D,read,write,pin,verify
C=D/'full_lower_covariance_edge_packed_v1'
child=read(C/'closeout.json');assert child['qualified'] and child['uploaded']
assert read(C/'native_qualified.json')['passed'] and read(C/'boundary_qualified.json')['passed']
for name in ['qualification.json','score.json','panel_score.json','confirmation_score.json']:assert read(HERE/name)['passed']
for p in read(HERE/'accuracy_closeout.json')['artifact_pins']+read(HERE/'accuracy_closeout.json')['external_pins']:verify(p)
guards=[read(p) for p in HERE.glob('*_watch.json')]
assert all(g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0 for g in guards)
records=[]
for p in HERE.glob('*/result.json'):
    j=read(p)
    if 'records' in j:records.extend(j['records'])
assert len(records)==29
unit=read(HERE/'unit.json')
resources=dict(guarded_groups=len(guards),all_groups_retired=True,max_swap_growth_bytes=0,
    max_guarded_rss_bytes=max(g['max_rss_bytes'] for g in guards),elapsed_guarded_s=sum(g['elapsed_s'] for g in guards),
    full_estimator_calls=len(records),full_estimator_flops=sum(r['flops'] for r in records),
    small_source_calls=sum(len(r['bills']) for r in unit['rows'])+3,
    small_metered_flops=unit['setup_flops']+unit['cold_flops']+unit['permutation_bill']+unit['zero_bill']+sum(sum(r['bills']) for r in unit['rows']),
    wide_reference_cells=len(read(HERE/'qualification.json')['fidelity']),new_fit_steps=0,new_samples=0,new_dataset_downloads=0,GPU_calls=0,
    disk_free_bytes=shutil.disk_usage(HERE).free,artifact_bytes_before_closeout=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()))
write(HERE/'resources.json',resources)
dev=read(HERE/'score.json');panel=read(HERE/'panel_score.json');confirm=read(HERE/'confirmation_score.json')
report=f'''# Frozen covariance edge model: positive native endpoint, packed derivative submitted

The fixed amplitude-one correction at layers8..14 passes native fidelity and
all declared complete-objective stages on the full-lower trajectory. Development
ratio{dev['pooled_adjusted_ratio']:.12f}; panel14 ratio{panel['pooled_adjusted_ratio']:.12f},
{panel['wins']}/14 wins; confirmation8 ratio{confirm['pooled_adjusted_ratio']:.12f},
{confirm['wins']}/8 wins, worst{confirm['worst_adjusted_ratio']:.12f}. Confirmation
mean adjusted{confirm['mean_candidate_adjusted']:.12g} versus
{confirm['mean_parent_adjusted']:.12g}. All source/action formula, symmetry,
repeat and disabled-parent checks pass. Predictions and complete cold bills
were sealed before existing targets. These reused public research networks
are not pristine holdouts or official grades.

The old model's strict covariance/response proxy gates remain failed. This is
a separately declared autonomous endpoint with no new fit or amplitude/layer
sweep. It includes no frozen mean-innovation source or newly fitted readout.

Because readout331676 had two official residual-time failures, this uncompressed
source was not uploaded. A separate lossless static-packing derivative matched
all16 output bytes and complete bills, passed real-client residual and matched
wall gates, contract and package checks, and was accepted as submission
{child['submission_id']}. See full_lower_covariance_edge_packed_v1/closeout.json
for exact archive/source identity. Official grade pending; bank331672 remains
3.6316074532884024e-9; approximately1e-9 goal unachieved.

This model study used29 full estimator calls and{len(guards)} guards, all retired;
peak{resources['max_guarded_rss_bytes']}B,zero added swap. No new samples,
training, datasets or GPU work. Earlier accuracy_closeout.json is preserved.
'''
with (HERE/'RESULT.md').open('x') as f:f.write(report)
files=sorted(p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
result=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accuracy_qualified=True,
    packed_derivative_qualified_and_submitted=True,submission_id=child['submission_id'],official_grade_pending=True,
    goal_achieved=False,artifact_pins=[pin(p) for p in files],external_pins=[pin(C/'closeout.json'),pin(C/'native_qualified.json'),pin(C/'boundary_qualified.json')])
for p in result['artifact_pins']+result['external_pins']:verify(p)
write(HERE/'closeout.json',result)
print(dict(artifact_pins=len(files),resources=resources,packed_derivative_submission=child['submission_id']),flush=True)
