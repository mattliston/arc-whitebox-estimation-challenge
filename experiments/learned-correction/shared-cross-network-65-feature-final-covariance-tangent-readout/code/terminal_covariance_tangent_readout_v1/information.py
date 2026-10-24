"""Fixed cross-network value test of the formal covariance tangent readout."""
import json,resource,time
import numpy as np
from threadpoolctl import threadpool_info
from common import HERE,D,P2,TRAIN,sha,read,seal_check,write
from fit_library import load_library
seal_check();assert all(x['num_threads']==1 for x in threadpool_info())
started=time.monotonic();manifest=read(HERE/'train_capture/manifest.json')
assert manifest['completed'] and not manifest['truth_accessed'] and manifest['train']==TRAIN
assert manifest['source_seal_sha256']==sha(HERE/'sources_sealed.json')
out=HERE/'information';out.mkdir(exist_ok=False);lib,equations=load_library();certificate=lib.certificate();data=[]
for row in manifest['rows']:
    guard=read(row['watch']);assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
    assert sha(row['completion'])==row['completion_sha256']
    rec=row['record'];assert sha(rec['capture'])==rec['capture_sha256'] and rec['includes_initialization']
    with np.load(rec['capture'],allow_pickle=False) as z:a={k:z[k].astype(np.float64) for k in z.files}
    B=a['sigma'][:,None]*np.column_stack([np.ones(1024),a['hidden']]);Z=a['features']
    assert B.shape==(1024,65) and Z.shape==(1024,65) and np.isfinite(B).all() and np.isfinite(Z).all()
    data.append(dict(row=row,prediction=a['prediction'][-1],B=B,Z=Z))
target_manifest=P2/'astra_phase2_reexamination_v1/joint_prefix_final_fit_v1/manifest.json'
assert sha(target_manifest)=='d61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971'
targets={r['rank']:r for r in read(target_manifest)['target_sources']}
write(out/'inputs_sealed_before_targets.json',dict(capture_manifest_sha256=sha(HERE/'train_capture/manifest.json'),source_seal_sha256=sha(HERE/'sources_sealed.json'),target_manifest_sha256=sha(target_manifest),train=TRAIN,targets_opened=False,fit_certificate=certificate))
target_hashes={}
for item in data:
    rank=item['row']['rank'];source=targets[rank];path=P2/f'public_g22_operator_features_v1/truthpost/truthpost_rank{rank:04d}.npz'
    assert str(path)==source['target_file'] and sha(path)==source['target_sha256']
    assert sha(source['weight_file'])==item['row']['record']['weights_sha256']
    with np.load(path,allow_pickle=False) as z:truth=z['mc_post'][-1].astype(np.float64)
    item.update(truth=truth,residual=truth-item['prediction']);target_hashes[str(rank)]=sha(path)
predictions={arm:[None]*20 for arm in ['control','extended']};fits=[]
for fold in range(5):
    fit_ids=[i for i in range(20) if i%5!=fold];held_ids=[i for i in range(20) if i%5==fold]
    x=np.concatenate([data[i]['B'] for i in fit_ids]);z=np.concatenate([data[i]['Z'] for i in fit_ids]);y=np.concatenate([data[i]['residual'] for i in fit_ids]);w=np.concatenate([np.full(1024,data[i]['row']['record']['parent_multiplier']) for i in fit_ids])
    before=len(equations);model=lib.fit(x,z,y,w)
    for i in held_ids:
        for arm,delta in zip(['control','extended'],lib.predict(model,data[i]['B'],data[i]['Z'])):
            predictions[arm][i]=np.maximum(data[i]['prediction']+delta,0).astype(np.float32)
    path=out/f'fold{fold}_fit.npz';np.savez(path,**model)
    record=dict(fold=fold,fit_ranks=[TRAIN[i] for i in fit_ids],held_ranks=[TRAIN[i] for i in held_ids],control_rank=model['rank'],retained_extension_columns=int(model['keep'].sum()),orthogonality=model['orthogonality'],equations=equations[before:],fit_path=str(path),fit_sha256=sha(path))
    fits.append(record);print(json.dumps(record),flush=True)
    del model,x,z,y,w
arrays={arm:np.stack(values) for arm,values in predictions.items()}
assert all(a.shape==(20,1024) and np.isfinite(a).all() and (a>=0).all() for a in arrays.values())
np.savez(out/'oof_predictions.npz',**arrays)
write(out/'oof_sealed_before_aggregate.json',dict(prediction_sha256=sha(out/'oof_predictions.npz'),fits=fits,target_hashes=target_hashes,selection_used=False))
rows=[]
for i,item in enumerate(data):
    rec=item['row']['record'];row=dict(rank=TRAIN[i],fold=i%5)
    for arm,added in [('unchanged',0),('control',2_000_000),('extended',10_000_000_000)]:
        p=item['prediction'] if arm=='unchanged' else arrays[arm][i].astype(np.float64)
        raw=float(np.mean((p-item['truth'])**2));F=rec['parent_flops']+added
        row[arm]=dict(raw=raw,flops=F,adjusted=raw*max(.1,F/2**41))
    rows.append(row)
arms=['unchanged','control','extended'];pooled={a:float(np.mean([r[a]['adjusted'] for r in rows])) for a in arms}
folds=[{a:float(np.mean([r[a]['adjusted'] for r in rows if r['fold']==f])) for a in arms} for f in range(5)]
wins=sum(f['extended']<min(f['unchanged'],f['control']) for f in folds)
ratio=pooled['extended']/pooled['unchanged'];control_ratio=pooled['extended']/pooled['control']
result=dict(completed=True,gate_passed=ratio<=.98 and control_ratio<=.99 and wins>=4,exploratory_train_network_cv=True,pooled_adjusted=pooled,extended_to_unchanged=ratio,extended_to_control=control_ratio,fold_wins=wins,fold_adjusted=folds,rows=rows,coefficients_fit_only=True,actual_feature_bills_charged=False,provisional_extended_allowance_flops=10_000_000_000,deployment_bill_qualified=False,source_seal_sha256=sha(HERE/'sources_sealed.json'),capture_manifest_sha256=sha(HERE/'train_capture/manifest.json'),elapsed_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,new_deployable_fit=False,new_archive=False,new_upload=False,additive_tangent_readout_only=True,old_terminal_covariance_oracle_gate_unchanged_failed=True)
write(out/'result.json',result)
print(json.dumps({k:v for k,v in result.items() if k not in ['rows','fold_adjusted']}),flush=True)
