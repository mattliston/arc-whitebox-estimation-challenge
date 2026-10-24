"""Close the negative finite endpoint; keep all failed harness evidence."""
import datetime
import shutil
from common import HERE,D,OLD,MODEL,PARENT,read,pin,write,verify,seal_check,guarded

seal_check()
for seal in ['repair_sources_sealed.json','parameter_accounting_sources_sealed.json']:
    for p in read(HERE/seal)['pins']:verify(p)
score=read(HERE/'score.json');qualification=read(HERE/'qualification.json')
assert not score['passed'] and qualification['passed']
assert read(HERE/'development_completed.json')['local_study_completed']
names=['build','unit','unit_v2','unit_v3','disabled','enabled','cold100','qualify','score']
guards={n:read(HERE/f'{n}_watch.json') for n in names}
for n,g in guards.items():
    assert g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    assert g['completed']==(n not in ['unit','unit_v2'])
records=sum([read(HERE/f'{n}/result.json')['records'] for n in ['disabled','enabled','cold100']],[])
unit=read(HERE/'unit_v3.json')
tiny=sum(sum(x['bills']) for x in unit['rows'])+unit['zero_bill']+unit['module_setup_flops']+unit['parameter_setup_flops']+unit['cold_helper_flops']
tiny+=unit['original_two_call_flops_inferred']+unit['v2_two_call_flops']
resources=dict(guarded_groups=len(guards),successful_groups=7,preserved_failed_unit_groups=2,
    all_groups_retired=True,max_swap_growth_bytes=0,elapsed_guarded_s=sum(g['elapsed_s'] for g in guards.values()),
    max_guarded_rss_bytes=max(g['max_rss_bytes'] for g in guards.values()),full_estimator_calls=len(records),
    full_estimator_flops=sum(r['flops'] for r in records),small_source_calls=12,small_metered_flops=tiny,
    wide_reference_cells=32,new_fit_steps=0,new_samples=0,new_dataset_downloads=0,GPU_calls=0,
    disk_free_bytes=shutil.disk_usage(HERE).free,
    artifact_bytes_before_closeout=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()))
write(HERE/'resources.json',resources)
rows=score['rows']
result=f'''# Frozen local mean innovation: native endpoint negative

The fixed amplitude-one correction at layers8..15 passed numerical/native
qualification but worsened the pooled complete own-cold adjusted objective
by {(score['pooled_adjusted_ratio']-1)*100:.9f}%. Ratios35/100 are
{rows[0]['adjusted_ratio']:.15g}/{rows[1]['adjusted_ratio']:.15g}.
Candidate adjusted scores are {rows[0]['candidate']['adjusted']:.15g} and
{rows[1]['candidate']['adjusted']:.15g}, versus the exact full-lower parent
{rows[0]['parent']['adjusted']:.15g}/{rows[1]['parent']['adjusted']:.15g}.
Compute increased only about0.014%; the failure is primarily raw accuracy.
These reused public research scores are not official or pristine holdouts.

The frozen original model and parent source are authenticated. Inverse source
editing restores the parent byte for byte. Disabled35/100/35 has all16 output
rows and complete bills exact. Enabled35/100/35+cold100 has first8 rows exact,
exact own repeats and warm/cold100 output/feature/source bytes. All32 wide
local-source evaluations match the original FP32 NumPy model exactly; maximum
independent FP64 relative discrepancy is
{max(r['fp64_relative'] for r in qualification['fidelity']):.12g}.

Two small unit harness failures are preserved. Both mistakenly demanded equal
cold and warm helper costs. Separate module initialization measured0F; frozen
parameter materialization measured7036F. The v3 test explicitly proves cold
bill = warm bill + measured parameter setup, with exact outputs. Candidate
source and numerical/objective tolerances never changed. All actual native
calls include cold setup when applicable.

The old TRAIN loss.876624 failed its original.50 gate and remains failed. This
was a distinct autonomous native intervention, with current live features and
actual state propagation, not the old unexecuted Gaussian-tangent forecast.
No coefficient, layer, amplitude, readout combination, fit or other rescue was
selected. The development gate failed; no broader panel, confirmation, real
client qualification, package or upload is warranted for this arm.

Nine guarded groups are retired (seven successful, two preserved harness
failures), peak{resources['max_guarded_rss_bytes']}B, zero added swap. Seven
full calls/{resources['full_estimator_flops']}F, twelve small source calls/
{tiny}F,32 wide reference cells; no fit/new samples/download/GPU work. The
official bank331672 remains3.6316074532884024e-9; readout331676 is separate
and pending at the latest observed poll. The full goal remains active.
'''
with (HERE/'RESULT.md').open('x') as f:f.write(result)
external=[pin(PARENT),pin(MODEL),pin(OLD/'fit_result.json'),pin(D/'official_best_331672.json')]
for p in external:verify(p)
artifacts=[pin(p) for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]
for p in artifacts:verify(p)
closeout=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    development_study_complete=True,qualification_passed=True,objective_gate_passed=False,
    no_candidate_upload=True,artifacts=artifacts,external_pins=external,resources=resources,
    prior_failed_proxy_gate_unchanged=True,goal_achieved=False)
write(HERE/'closeout.json',closeout)
print(dict(closeout=pin(HERE/'closeout.json'),artifacts=len(artifacts),resources=resources),flush=True)
