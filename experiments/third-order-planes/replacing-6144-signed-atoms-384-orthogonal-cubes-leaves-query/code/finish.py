"""Post-result stdlib verification and immutable research closeout."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    with Path(p).open('x') as stream:json.dump(value,stream,indent=2)

seal=read(HERE/'sources_sealed.json')
for item in seal['sources']+seal['external_pins']:
    assert sha(item['path'])==item['sha256'],item['path']
guards=[read(HERE/name) for name in ('unit_watch.json','information_watch.json')]
assert all(g['completed'] and g['whole_group_retired'] and g['exit_code']==0
           and g['error'] is None and g['max_swap_growth_bytes']==0 for g in guards)
unit=read(HERE/'unit.json');result=read(HERE/'result.json')
assert unit['passed'] and len(unit['rows'])==12
assert max(max(row['errors'].values()) for row in unit['rows'])<=1e-11
assert result['completed'] and result['numerical_checks_passed']
assert not result['information_gate_passed']
assert result['query_relative_squared_error']>.01 and result['plane_relative_squared_error']>.01
for key in ('query','plane'):
    measured=result[key+'_residual_squared']/result[key+'_signal_squared']
    assert abs(measured-result[key+'_relative_squared_error'])<1e-14
assert abs((result['old_tensor_frobenius_squared']-result['retained_orthogonal_cube_energy'])/result['old_tensor_frobenius_squared']-result['tensor_relative_squared_error'])<1e-14
assert sha(HERE/'fixed_merge.npz')==result['merge_sha256']==read(HERE/'merge_sealed_before_verification.json')['merge_sha256']
assert result['source_sha256']==sha(HERE/'information.py')
assert result['algebra_sha256']==sha(HERE/'algebra.py')
subtotals={}
for record in result['products']:
    left,right=record['shapes'];assert left[1]==right[0]
    assert record['fp64_leading_billed_flops']==4*left[0]*left[1]*right[1]
    group=record['tag'].split('_')[0]
    subtotals[group]=subtotals.get(group,0)+record['fp64_leading_billed_flops']
assert subtotals==result['fp64_gemm_leading_subtotals']
assert not any(result[key] for key in ('complete_fnp_deployment_bill','actual_accuracy_score_measured','truth_accessed','model_calls','new_samples','new_training'))
peak=max([g['max_rss_bytes'] for g in guards]+[unit['peak_rss_bytes'],result['peak_rss_bytes']])
assert peak<8_000_000_000
resources=dict(guarded_stages=2,successful_guarded_stages=2,
               total_guarded_wall_s=sum(g['elapsed_s'] for g in guards),
               peak_rss_bytes=peak,peak_includes_worker_self_reports=True,
               additional_swap_bytes=0,all_owned_groups_retired=True,
               terminal_sessions=[50460],full_estimator_calls=0,
               new_samples=0,new_training=False,truth_arrays_accessed=False,
               new_archive=False,new_upload=False,
               fp64_gemm_leading_subtotals=subtotals,
               complete_fnp_deployment_bill=False,
               remaining_disk_GiB=shutil.disk_usage(HERE).free/2**30)
write(HERE/'resources.json',resources)
files=[p for p in sorted(HERE.rglob('*')) if p.is_file()]
size=sum(p.stat().st_size for p in files)
assert size<12_000_000
closeout=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              status='closed negative at fixed modeled-tensor fidelity gate',
              source_seal_sha256=sha(HERE/'sources_sealed.json'),
              artifacts=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in files],
              external_pins=seal['external_pins'],artifact_bytes_before_closeout=size,
              resources=resources,no_active_worker=True,
              goal_status='active',goal_achieved=False)
write(HERE/'closeout.json',closeout)
print(json.dumps(dict(closeout_sha256=sha(HERE/'closeout.json'),
                     artifact_count=len(files),external_pin_count=len(seal['external_pins']),
                     artifact_bytes_before_closeout=size,resources=resources)),flush=True)
