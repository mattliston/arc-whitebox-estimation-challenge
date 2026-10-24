"""Audit the complete trajectory and export only the sole terminal P payload."""
import math
import os
from iso_common import *
import state_io

verify()
assert os.getpriority(os.PRIO_PROCESS,0)>=10
assert not (HERE/'fit_result.json').exists() and not (HERE/'terminal_payload.npz').exists()
import numpy as np
assert np.__version__=='1.26.4'
order=read(HERE/'training_order.json')
assert order==read(FIT/'training_order.json')
qualified=read(QUAL/'reference/result.json')
initial=qualified['initial_parameter_hashes'];pkeys=qualified['trainable_keys']
template=read(QUAL/'reference/checkpoint_02.json')
records=[];updates=[];checkpoints=[];worker_pins=[];max_retired=0;watches=[]
old_graph_count=old_update_count=qualified_graph_count=qualified_update_count=0
previous_checkpoint=None
for update in range(40):
    arm=f'batch_{update:02d}';watches.append(clean_watch(arm));folder=HERE/arm
    worker=read(folder/'result.json');worker_pins.append(pin(folder/'result.json'))
    assert worker['completed'] and worker['passed'] and worker['arm']==arm
    assert worker['start_update']==update and worker['stop_update']==update+1
    assert worker['training_graphs']==4 and worker['updates']==1
    assert len(worker['records'])==4 and len(worker['update_receipts'])==len(worker['checkpoints'])==1
    assert worker['initial_parameter_hashes']==initial and worker['trainable_keys']==pkeys
    assert worker['source_seal']==pin(HERE/'sources_sealed.json')
    assert worker['fixed_parameters_unchanged'] and worker['inactive_parameters_unchanged']
    assert not worker['validation_targets_accessed'] and not worker['checkpoint_selection']
    assert not worker['native_score_measured'] and not worker['candidate_created']
    assert worker['restored_checkpoint']==previous_checkpoint
    check(worker['training_order']);assert read(worker['training_order']['path'])==order
    for within,row in enumerate(worker['records']):
        check(row['record']);record=read(row['record']['path']);check(record['prediction'])
        index=4*update+within
        assert record['index']==row['index']==index and record['rank']==row['rank']==order['ranks'][index]
        assert record['update']==update+1 and record['within_batch']==within
        assert record['fixed_parameters_unchanged'] and record['card_retired'] and not record['validation_targets_accessed']
        assert set(record['accumulated_gradient_hashes'])==set(pkeys)
        assert set(record['domain'])==set(record['states'])=={str(i) for i in range(16)}
        assert len(record['selectors'])==45
        retired=record['retirement_memory']['after']['rss_bytes']
        assert retired<2_000_000_000;max_retired=max(max_retired,retired)
        checks=prefix_graph(record);assert checks==record['prefix_checks']
        old_graph_count+=checks['old_completed_graph_exact'];qualified_graph_count+=checks['qualified_gradient_graph_exact']
        records.append(row)
    entry=worker['update_receipts'][0];check(entry);record=read(entry['path'])
    assert record['update']==update+1 and record['ranks']==order['ranks'][4*update:4*update+4]
    assert record['learning_rate']==5e-5+.5*(5e-4-5e-5)*(1+math.cos(math.pi*update/39))
    assert record['fixed_parameters_unchanged'] and set(record['clipped_gradient_hashes'])==set(pkeys)
    checks=prefix_update(record);assert checks==record['prefix_checks']
    old_update_count+=checks['old_completed_update_exact'];qualified_update_count+=checks['qualified_state_boundary_exact']
    checkpoint=worker['checkpoints'][0];check(checkpoint)
    assert record['checkpoint']==checkpoint
    state=read(checkpoint['path']);check(state['payload'])
    assert state['complete'] and state['completed_updates']==update+1 and state['mandatory_boundary_only']
    assert state['round_trip_exact'] and state['live_values_unchanged'] and not state['candidate_eligible']
    assert state['parameter_keys']==list(initial) and state['trainable_keys']==pkeys
    assert set(state['tensors'])==set(template['tensors'])
    assert {k:projection(v,['dtype','shape']) for k,v in state['tensors'].items()}=={k:projection(v,['dtype','shape']) for k,v in template['tensors'].items()}
    with np.load(state['payload']['path'],allow_pickle=False) as source:
        values={k:source[k].copy() for k in source.files}
    assert state_io.descriptors(values)==state['tensors']
    assert all(np.isfinite(v).all() for v in values.values())
    assert all(float(values['step__'+k])==update+1 for k in pkeys)
    expected_groups=json.loads(json.dumps(template['optimizer_groups']))
    expected_groups[0]['lr']=record['learning_rate']
    assert state['optimizer_groups']==expected_groups
    parameter_hashes={k:state['tensors']['p__'+k]['sha256'] for k in initial}
    assert parameter_hashes==record['parameter_hashes']==worker['terminal_parameter_hashes']
    assert all(parameter_hashes[k]==initial[k] for k in initial if k not in pkeys)
    previous_checkpoint=checkpoint;updates.append(entry);checkpoints.append(checkpoint)
assert len(records)==160 and len(updates)==len(checkpoints)==40
assert (old_graph_count,old_update_count,qualified_graph_count,qualified_update_count)==(134,33,8,2)
changed=[k for k in pkeys if parameter_hashes[k]!=initial[k]]
assert 'ungated_w' in changed
payload={k:values['p__'+k].copy() for k in initial}
assert {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in payload.items()}==parameter_hashes
payload_path=HERE/'terminal_payload.npz'
with payload_path.open('xb') as stream:np.savez(stream,**payload)
with np.load(payload_path,allow_pickle=False) as source:
    assert set(source.files)==set(initial)
    for k in initial:assert source[k].dtype==payload[k].dtype and source[k].shape==payload[k].shape and source[k].tobytes()==payload[k].tobytes()
verify()
resource_floor()
write(HERE/'fit_result.json',dict(completed=True,passed=True,model_fitted=True,training_graphs=160,updates=40,epoch=1,
    records=records,update_receipts=updates,checkpoints=checkpoints,worker_results=worker_pins,
    terminal_payload=pin(payload_path),initial_parameter_hashes=initial,terminal_parameter_hashes=parameter_hashes,
    changed_trainable_keys=changed,fixed_parameters_unchanged=True,inactive_parameters_unchanged=True,
    validation_targets_accessed=False,training_targets_accessed=True,checkpoint_selection=False,
    native_derivative_certified=False,native_score_measured=False,candidate_created=False,
    old_completed_graphs_exact=134,old_completed_updates_exact=33,qualified_gradient_graphs_exact=8,
    qualified_state_boundaries_exact=2,all_checkpoint_arrays_reverified=True,
    all_worker_groups_retired=True,max_retired_graph_rss_bytes=max_retired,
    max_batch_group_rss_bytes=max(w['max_rss_bytes'] for w in watches),max_added_swap_bytes=0,
    training_order=pin(HERE/'training_order.json'),source_seal=pin(HERE/'sources_sealed.json')))
print(dict(completed=True,fit_result=pin(HERE/'fit_result.json')),flush=True)
