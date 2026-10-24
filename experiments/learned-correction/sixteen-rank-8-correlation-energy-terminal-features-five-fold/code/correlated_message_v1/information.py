"""Fixed five-network-fold information test after complete feature acquisition."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np

HERE=Path(__file__).resolve().parent
P2=HERE.parent.parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    started=time.monotonic()
    manifest=read(HERE/'train_capture/manifest.json')
    assert manifest['completed'] and not manifest['truth_accessed'] and len(manifest['rows'])==20
    assert manifest['feature_source_sha256']==sha(HERE/'feature.py')
    assert manifest['protocol_sha256']==sha(HERE/'PROTOCOL.md')
    out=HERE/'information'
    out.mkdir(exist_ok=False)
    data=[]
    # Authenticate and load every target-free capture before opening any target.
    for row in manifest['rows']:
        r=row['record']
        w=read(row['watch'])
        assert w['completed'] and w['whole_group_retired'] and w['max_swap_growth_bytes']==0
        assert sha(row['completion'])==row['completion_sha256']
        assert sha(r['capture'])==r['capture_sha256']
        with np.load(r['capture']) as f:
            d={k:f[k].astype(np.float64) for k in f.files}
        h,extra,sigma=d['hidden'],d['correlated'],d['sigma']
        assert h.shape==(1024,64) and extra.shape==(1024,16) and sigma.shape==(1024,)
        x_control=sigma[:,None]*np.concatenate([np.ones((1024,1)),h],axis=1)
        x_extended=np.concatenate([x_control,sigma[:,None]*extra],axis=1)
        data.append(dict(row=row,prediction=d['prediction'][-1],control=x_control,extended=x_extended))
    inputs_seal=dict(manifest_sha256=sha(HERE/'train_capture/manifest.json'),ranks=manifest['train'],
                     source_sha256=sha(__file__),targets_opened=False)
    target_manifest=P2/'astra_phase2_reexamination_v1/joint_prefix_final_fit_v1/manifest.json'
    assert sha(target_manifest)=='d61757d28236e3d35d87415962d1a57d18ff38ae189651df52df5e3412b9a971'
    target_index={r['rank']:r for r in read(target_manifest)['target_sources']}
    inputs_seal['original_target_manifest_sha256']=sha(target_manifest)
    with (out/'inputs_sealed.json').open('x') as f:json.dump(inputs_seal,f,indent=2)
    target_hashes={}
    for d in data:
        rank=d['row']['rank']
        target=P2/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{rank:04d}.npz'
        target_hashes[str(rank)]=sha(target)
        assert str(target)==target_index[rank]['target_file']
        assert target_hashes[str(rank)]==target_index[rank]['target_sha256']
        assert sha(target_index[rank]['weight_file'])==d['row']['record']['weights_sha256']
        truth=np.load(target)['mc_post'][-1].astype(np.float64)
        d.update(truth=truth,residual=truth-d['prediction'])
    predictions={'control':[None]*20,'extended':[None]*20}
    fits=[]
    for fold in range(5):
        train=[i for i in range(20) if i%5!=fold]
        check=[i for i in range(20) if i%5==fold]
        for arm in ('control','extended'):
            x=np.concatenate([data[i][arm] for i in train])
            y=np.concatenate([data[i]['residual'] for i in train])
            weight=np.concatenate([np.full(1024,data[i]['row']['record']['parent_multiplier']) for i in train])
            rms=np.sqrt(np.sum(x*x*weight[:,None],axis=0)/np.sum(weight))
            rms=np.maximum(rms,1e-12)
            xn=x/rms
            gram=xn.T@(xn*weight[:,None])
            gram=(gram+gram.T)*.5
            gram.flat[::gram.shape[0]+1] += 1e-4*np.sum(weight)
            rhs=xn.T@(weight*y)
            chol=np.linalg.cholesky(gram)
            coefficient=np.linalg.solve(chol.T,np.linalg.solve(chol,rhs))
            for i in check:
                correction=(data[i][arm]/rms)@coefficient
                predictions[arm][i]=np.maximum(data[i]['prediction']+correction,0).astype(np.float32)
            fits.append(dict(fold=fold,arm=arm,train_ranks=[data[i]['row']['rank'] for i in train],
                             check_ranks=[data[i]['row']['rank'] for i in check],rms=rms.tolist(),coefficient=coefficient.tolist()))
    # Finish and seal all OOF outputs before computing their aggregate errors.
    arrays={a:np.stack(p) for a,p in predictions.items()}
    assert all(x.shape==(20,1024) and np.isfinite(x).all() for x in arrays.values())
    np.savez(out/'oof_predictions.npz',**arrays)
    with (out/'oof_sealed.json').open('x') as f:
        json.dump(dict(predictions_sha256=sha(out/'oof_predictions.npz'),target_sha256=target_hashes,
                       fits=fits,selection_used=False),f,indent=2)
    rows=[]
    for i,d in enumerate(data):
        r=d['row']['record']
        row=dict(rank=d['row']['rank'],fold=i%5)
        for arm in ('unchanged','control','extended'):
            pred=d['prediction'] if arm=='unchanged' else arrays[arm][i].astype(np.float64)
            raw=float(np.mean((pred-d['truth'])**2))
            overhead=0 if arm=='unchanged' else (1_000_000 if arm=='control' else r['feature_flops']+2_000_000)
            flops=r['parent_flops']+overhead
            row[arm]=dict(raw=raw,flops=flops,adjusted=raw*max(.1,flops/2**41))
        rows.append(row)
    pooled={a:float(np.mean([r[a]['adjusted'] for r in rows])) for a in ('unchanged','control','extended')}
    folds=[{a:float(np.mean([r[a]['adjusted'] for r in rows if r['fold']==f]))
            for a in ('unchanged','control','extended')} for f in range(5)]
    fold_wins=sum(f['extended']<min(f['unchanged'],f['control']) for f in folds)
    passed=pooled['extended']<=.99*pooled['unchanged'] and pooled['extended']<pooled['control'] and fold_wins>=4
    result=dict(schema='correlated_terminal_message_information_v1',exploratory_train_network_cv=True,
        rows=rows,pooled_adjusted=pooled,fold_adjusted=folds,fold_wins=fold_wins,
        extended_to_unchanged=pooled['extended']/pooled['unchanged'],
        extended_to_control=pooled['extended']/pooled['control'],gate_passed=bool(passed),
        source_sha256=sha(__file__),input_manifest_sha256=inputs_seal['manifest_sha256'],
        feature_bill_min=min(d['row']['record']['feature_flops'] for d in data),
        feature_bill_max=max(d['row']['record']['feature_flops'] for d in data),
        deployment_bill_qualified=False,elapsed_s=time.monotonic()-started,
        max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with (out/'result.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','fold_adjusted')}),flush=True)


if __name__=='__main__':
    main()
