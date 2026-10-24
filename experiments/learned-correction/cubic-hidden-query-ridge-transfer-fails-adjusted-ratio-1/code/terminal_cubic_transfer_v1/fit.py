import resource
import numpy as np
from common import *
from solver import fit
seal_check();done('qualify')
sealed=read(HERE/'features_sealed.json')
for p in sealed['pins']:check(p)
inputs=read(HERE/'inputs.json');train=inputs['train'];meta={r['rank']:r for r in inputs['rows']}
records={r['rank']:r for r in sealed['records'] if r['name'].startswith('train')}
designs={};parents={}
for rank in train:
    with np.load(records[rank]['file']['path'],allow_pickle=False) as z:
        designs[rank]=dict(affine=z['affine'],extended=np.column_stack([z['affine'],z['feature']]))
        parents[rank]=z['prediction'][-1]
oof={k:np.zeros((20,1024),np.float32) for k in ['affine','extended']};models=[]
for fold in range(5):
    fit_ranks=[r for i,r in enumerate(train) if i%5!=fold];held=[r for i,r in enumerate(train) if i%5==fold]
    targets=[];weights=[]
    for rank in fit_ranks:
        check(meta[rank]['truth'])
        with np.load(meta[rank]['truth']['path'],allow_pickle=False) as z:truth=z['mc_post'][-1].astype(float)
        targets.append(truth-parents[rank].astype(float))
        weights.append(np.full(1024,max(.1,records[rank]['parent_flops']/2**41)))
    y=np.concatenate(targets);w=np.concatenate(weights)
    for kind in ['affine','extended']:
        X=np.concatenate([designs[r][kind] for r in fit_ranks]);beta,cert=fit(np,X,y,w)
        model_path=HERE/f'fold{fold}_{kind}.npz'
        np.savez(model_path,beta=beta,scale=cert['scale'],scaled_beta=cert['scaled_beta'])
        for rank in held:
            oof[kind][train.index(rank)]=np.maximum(parents[rank].astype(float)+designs[rank][kind]@beta,0).astype(np.float32)
        models.append(dict(fold=fold,kind=kind,fit=fit_ranks,held=held,parameters=pin(model_path),
            normal_residual=cert['residual'],cholesky_relative=cert['cholesky_relative']))
    print(dict(fold=fold,held=held,models_frozen=True),flush=True)
files={}
for kind,array in oof.items():
    p=HERE/f'oof_{kind}.npy';np.save(p,array);files[kind]=pin(p)
write(HERE/'oof_predictions_sealed.json',dict(models=models,predictions=files,train=train,
    feature_seal=pin(HERE/'features_sealed.json'),all_held_predictions_frozen_before_scoring=True,
    each_model_uses_only_fit_targets=True,reference_usage='network CV; targets may be training in another fold',
    fits=10,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
