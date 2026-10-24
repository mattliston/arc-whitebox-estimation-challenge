from fit_common import *
verify()
import numpy as np
model,_=qualified_modules()
p,a,c,initial_edge=model.parameters(np)
digest=lambda v:hashlib.sha256(v.tobytes()).hexdigest()
fixed_p={k:digest(v) for k,v in p.items()};fixed_ac=[digest(a),digest(c)]
fixed_edge={k:digest(v) for k,v in initial_edge.items() if k not in EKEYS}
inputs=read(HERE/'inputs.json');records=[];updates=[];guards=[];order=None;last=None
for batch in range(40):
    name=f'batch_{batch:02d}';folder=HERE/name;guards.append(clean_watch(name))
    r=read(folder/'result.json')
    assert r['completed'] and r['passed'] and r['update']==batch+1 and r['training_graphs']==4
    assert r['fixed_p_hashes']==fixed_p and r['fixed_ac_hashes']==fixed_ac and r['fixed_edge_hashes']==fixed_edge
    assert r['edge_trainable_keys']==list(EKEYS) and r['edge_trainable_values']==528
    assert r['fixed_parameters_unchanged'] and not r['validation_targets_accessed'] and not r['checkpoint_selection']
    assert r['source_seal']==pin(HERE/'sources_sealed.json')
    current=read(folder/'training_order.json');assert current['seed']==9409213
    if order is None:order=current['ranks']
    assert order==current['ranks'] and len(order)==160 and sorted(order)==sorted(inputs['train'])
    assert r['restored_checkpoint']==last
    check(r['checkpoint']);checkpoint=read(r['checkpoint']['path']);check(checkpoint['payload'])
    assert checkpoint['completed_updates']==batch+1 and checkpoint['complete']
    assert checkpoint['round_trip_exact'] and checkpoint['live_values_unchanged']
    assert checkpoint['mandatory_boundary_only'] and not checkpoint['candidate_eligible']
    assert checkpoint['trainable_keys']==list(EKEYS) and checkpoint['parameter_keys']==list(initial_edge)
    assert r['terminal_edge_hashes']=={k:checkpoint['tensors']['p__'+k]['sha256'] for k in initial_edge}
    for within,rpin in enumerate(r['records']):
        check(rpin);row=read(rpin['path']);index=batch*4+within;rank=order[index]
        assert (row['index'],row['rank'],row['update'],row['within_batch'])==(index,rank,batch+1,within)
        assert row['weights']==inputs['weights'][str(rank)] and row['target']==inputs['targets'][str(rank)]
        check(row['prediction'])
        assert row['fixed_parameters_unchanged'] and row['card_retired'] and not row['validation_targets_accessed']
        assert row['retirement_memory']['after']['rss_bytes']<2_000_000_000
        assert set(row['accumulated_gradient_hashes'])==set(EKEYS)
        assert len(row['domain'])==16 and all(d['finite'] and d['minimum']>=1e-10 for d in row['domain'].values())
        assert len(row['selectors'])==45 and len(row['states'])==16
        records.append(rpin)
    check(r['update_receipt']);updates.append(r['update_receipt']);last=r['checkpoint']
assert len(records)==160 and len(updates)==40
with np.load(checkpoint['payload']['path'],allow_pickle=False) as z:
    terminal={k:z['p__'+k].copy() for k in initial_edge}
    assert all(z['step__'+k].item()==40 for k in EKEYS)
for k,v in terminal.items():
    assert v.dtype==initial_edge[k].dtype and v.shape==initial_edge[k].shape and np.isfinite(v).all()
    if k not in EKEYS:assert v.tobytes()==initial_edge[k].tobytes()
changed=[k for k in EKEYS if terminal[k].tobytes()!=initial_edge[k].tobytes()]
assert changed
path=HERE/'terminal_edge_payload.npz'
with path.open('xb') as f:np.savez(f,**terminal)
with np.load(path,allow_pickle=False) as z:
    assert set(z.files)==set(terminal) and all(z[k].tobytes()==v.tobytes() for k,v in terminal.items())
verify()
write(HERE/'fit_result.json',dict(completed=True,passed=True,model_fitted=True,training_graphs=160,updates=40,
    terminal_payload=pin(path),changed_trainable_keys=changed,edge_trainable_values=528,
    fixed_p_hashes=fixed_p,fixed_ac_hashes=fixed_ac,fixed_edge_hashes=fixed_edge,
    terminal_edge_hashes={k:digest(v) for k,v in terminal.items()},records=records,update_receipts=updates,
    fixed_parameters_unchanged=True,validation_targets_accessed=False,checkpoint_selection=False,
    native_derivative_certified=False,native_accuracy_measured=False,production_candidate_created=False,
    worker_guards=40,worker_peak_rss_bytes=max(g['max_rss_bytes'] for g in guards),zero_added_swap=True,
    all_worker_groups_retired=True,source_seal=pin(HERE/'sources_sealed.json'),qualification=pin(Q/'result.json')))
print(dict(completed=True,graphs=160,updates=40,terminal_payload=pin(path),changed=changed),flush=True)
