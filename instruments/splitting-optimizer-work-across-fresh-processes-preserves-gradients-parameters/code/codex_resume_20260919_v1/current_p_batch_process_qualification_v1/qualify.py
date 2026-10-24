"""Require exact predictions, gradients and complete optimizer-state transport."""
from iso_common import *

verify(data=True)
assert not (HERE/'qualification.json').exists()
first=[116,74,96,120,161,177,207,154]
graph_keys=['index','rank','update','within_batch','weights','target','final_mse','earlier_mse',
    'batch_scaled_loss','parameter_hashes_during_graph','domain','selectors','states','visits',
    'fixed_parameters_unchanged','validation_targets_accessed','card_retired']
update_keys=['update','ranks','learning_rate','pre_clip_gradient_norm',
    'fixed_parameters_unchanged','parameter_hashes']
checkpoint_keys=['complete','completed_updates','tensors','parameter_keys','trainable_keys',
    'optimizer_groups','round_trip_exact','live_values_unchanged','mandatory_boundary_only','candidate_eligible']
def projection(value,keys):return {k:value[k] for k in keys}
def contents(value):return {k:value[k] for k in ('bytes','sha256')}

arms={};watches={};graph_pins=[];update_pins=[];checkpoint_pins=[]
for arm,start,stop in [('reference',0,2),('split0',0,1),('split1',1,2)]:
    watches[arm]=clean_watch(arm)
    result=read(HERE/arm/'result.json')
    assert result['completed'] and result['passed'] and result['arm']==arm
    assert result['start_update']==start and result['stop_update']==stop
    assert result['training_graphs']==4*(stop-start) and result['updates']==stop-start
    assert len(result['records'])==4*(stop-start) and len(result['update_receipts'])==stop-start
    assert len(result['checkpoints'])==stop-start and len(result['trainable_keys'])==12
    assert result['fixed_parameters_unchanged'] and result['inactive_parameters_unchanged']
    assert not result['validation_targets_accessed'] and not result['checkpoint_selection']
    assert not result['native_score_measured'] and not result['candidate_created']
    assert result['source_seal']==pin(HERE/'sources_sealed.json')
    check(result['training_order'])
    assert read(result['training_order']['path'])==read(FIT/'training_order.json')
    graph_records={}
    for row in result['records']:
        check(row['record']);graph_pins.append(row['record'])
        record=read(row['record']['path']);check(record['prediction'])
        assert record['index']==row['index'] and record['rank']==row['rank']==first[row['index']]
        assert record['card_retired'] and record['retirement_memory']['after']['rss_bytes']<2_000_000_000
        assert set(record['accumulated_gradient_hashes'])==set(result['trainable_keys'])
        assert set(record['domain'])=={str(i) for i in range(16)} and len(record['selectors'])==45
        assert set(record['states'])=={str(i) for i in range(16)}
        assert record['index'] not in graph_records
        graph_records[record['index']]=record
    assert set(graph_records)==set(range(4*start,4*stop))
    update_records={}
    for entry in result['update_receipts']:
        check(entry);update_pins.append(entry);record=read(entry['path'])
        assert set(record['clipped_gradient_hashes'])==set(result['trainable_keys'])
        assert record['update'] not in update_records
        update_records[record['update']]=record
    assert set(update_records)==set(range(start+1,stop+1))
    checkpoints={}
    for entry in result['checkpoints']:
        check(entry);checkpoint_pins.append(entry);record=read(entry['path']);check(record['payload'])
        assert record['complete'] and record['round_trip_exact'] and record['live_values_unchanged']
        assert record['mandatory_boundary_only'] and not record['candidate_eligible']
        expected={'p__'+k for k in record['parameter_keys']}|{'rng__cpu'}
        expected|={name+'__'+k for name in ('step','exp_avg','exp_avg_sq') for k in result['trainable_keys']}
        assert set(record['tensors'])==expected
        assert record['completed_updates'] not in checkpoints
        checkpoints[record['completed_updates']]=record
        assert update_records[record['completed_updates']]['checkpoint']==entry
    assert set(checkpoints)==set(range(start+1,stop+1))
    assert result['terminal_parameter_hashes']==update_records[stop]['parameter_hashes']
    arms[arm]=dict(result=result,graphs=graph_records,updates=update_records,checkpoints=checkpoints)

initial=read(FIT/'train_000_result.json')['parameter_hashes_during_graph']
assert all(a['result']['initial_parameter_hashes']==initial for a in arms.values())
assert arms['reference']['result']['restored_checkpoint'] is None and arms['split0']['result']['restored_checkpoint'] is None
assert arms['split1']['result']['restored_checkpoint']==pin(HERE/'split0/checkpoint_01.json')
for index in range(8):
    reference=arms['reference']['graphs'][index]
    split=arms['split0' if index<4 else 'split1']['graphs'][index]
    old=read(FIT/f'train_{index:03d}_result.json')
    assert projection(reference,graph_keys)==projection(old,graph_keys),(index,'old_graph')
    assert contents(reference['prediction'])==contents(old['prediction']),(index,'old_prediction')
    assert projection(reference,graph_keys+['accumulated_gradient_hashes'])==projection(split,graph_keys+['accumulated_gradient_hashes']),(index,'split_graph')
    assert contents(reference['prediction'])==contents(split['prediction']),(index,'split_prediction')
for number,arm in [(1,'split0'),(2,'split1')]:
    reference=arms['reference']['updates'][number]
    split=arms[arm]['updates'][number]
    old=read(FIT/f'update_{number:02d}.json')
    assert projection(reference,update_keys)==projection(old,update_keys),(number,'old_update')
    assert projection(reference,update_keys+['clipped_gradient_hashes'])==projection(split,update_keys+['clipped_gradient_hashes']),(number,'split_update')
    assert projection(arms['reference']['checkpoints'][number],checkpoint_keys)==projection(arms[arm]['checkpoints'][number],checkpoint_keys),(number,'checkpoint')
check(read(HERE/'initial_rng.json')['payload'])
resource_floor()
write(HERE/'qualification.json',dict(passed=True,backward_graphs=16,optimizer_updates=4,
    unique_training_networks=8,old_first_eight_graphs_exact=True,old_first_two_updates_exact=True,
    split_predictions_gradients_parameters_domains_selectors_states_visits_exact=True,
    all_parameter_adam_group_and_rng_states_exact=True,all_round_trips_exact=True,
    resume_used_exact_split0_boundary=True,initial_rng=pin(HERE/'initial_rng.json'),
    graph_receipts=graph_pins,update_receipts=update_pins,checkpoint_receipts=checkpoint_pins,
    arm_results=[pin(HERE/arm/'result.json') for arm in arms],
    watches=[pin(HERE/f'{arm}_watch.json') for arm in arms],
    max_group_rss_bytes=max(w['max_rss_bytes'] for w in watches.values()),
    max_retired_graph_rss_bytes=max(r['retirement_memory']['after']['rss_bytes'] for a in arms.values() for r in a['graphs'].values()),
    max_added_swap_bytes=max(w['max_swap_growth_bytes'] for w in watches.values()),
    every_worker_group_retired=True,validation_targets_accessed=False,native_calls=0,
    full_fit_resource_success_proven=False,native_derivatives_certified=False,
    accuracy_measured=False,candidate_created=False,full_fit_launched=False,
    full_fit_requires_new_seal_and_original_zero_start=True,sources=pin(HERE/'sources_sealed.json')))
print(dict(passed=True,qualification=pin(HERE/'qualification.json')),flush=True)
