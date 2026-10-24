"""Qualify on synthetic data, then one network-separated fit per process."""
import json
import resource
import sys
import time
import numpy as np
from common import D, bind, sha, once, arrays, block_system, response, load


def qualify():
    cfg, _ = bind()
    rng = np.random.default_rng(2026092701)
    z = np.c_[np.ones(8), rng.normal(size=(8,3))]
    b = rng.normal(size=(8,13,3)); y = rng.normal(size=(8,13))
    coefficient, residual = block_system(z,b,y)
    design = np.einsum('np,nma->nmpa',z,b).reshape(104,12)
    penalty = np.r_[np.zeros(3),np.ones(9)]
    explicit = np.linalg.solve(design.T@design/104+np.diag(penalty),design.T@y.ravel()/104)
    error = float(np.max(np.abs(coefficient.ravel()-explicit)))
    assert error < 1e-12
    perm = rng.permutation(13)
    permcoef, _ = block_system(z,b[:,perm],y[:,perm])
    assert np.allclose(coefficient,permcoef,rtol=0,atol=1e-12)
    zero, _ = block_system(z,b,np.zeros_like(y))
    assert np.count_nonzero(zero) == 0
    # Check that the five-mode whitening actually produces unit pooled Gram.
    q = rng.normal(size=(8,13,5))
    gram = np.einsum('nma,nmb->ab',q,q)/(8*13)
    inv = np.linalg.solve(np.linalg.cholesky(gram).T,np.eye(5))
    white = q@inv
    assert np.allclose(np.einsum('nma,nmb->ab',white,white)/104,np.eye(5),rtol=0,atol=2e-15)
    once(D/'qualification.json',dict(passed=True,explicit_design_error=error,
        block_solve_residual=residual,neuron_permutation=True,zero_target=True,
        whitening=True,synthetic_only=True,source_seal_sha256=sha(D/'source_seal.json')))
    print(json.dumps(dict(qualified=True,explicit_design_error=error)),flush=True)


def run(fold):
    cfg, records = bind()
    for name in ('qualification.json','qualification_guard.json'):
        record = json.loads((D/name).read_text())
        assert record['passed'] if name.startswith('qualification.json') else (record['completed'] and record['whole_group_retired'] and record['max_swap_growth_bytes']==0)
    test = cfg['folds'][fold]; train = sorted(set(cfg['ranks'])-set(test))
    assert len(train)==96 and len(test)==32 and set(train).isdisjoint(test)
    out = D/f'fold_{fold}';out.mkdir();started=time.monotonic()
    x,sigma,base,truth,provenance = load(train,records,cfg,True,train)
    summary=x.mean(1)
    mean=summary.mean(0);scale=np.maximum(summary.std(0),1e-6)
    z=np.c_[np.ones(len(train)),np.clip((summary-mean)/scale,-8,8)]
    output_scale=float(np.sqrt(np.mean((sigma*.001)**2)))
    raw=response(x,sigma)/output_scale
    gram=np.einsum('nma,nmb->ab',raw,raw,optimize=True)/(96*1024)
    spectrum=np.linalg.eigvalsh(gram)
    assert spectrum[0]>0 and spectrum[-1]/spectrum[0]<1e10
    inverse=np.linalg.solve(np.linalg.cholesky(gram).T,np.eye(5))
    b=raw@inverse;y=(truth-base)/output_scale
    models={}
    for mode in ('intercept','network'):
        features=z[:,:1] if mode=='intercept' else z
        coeff,residual=block_system(features,b,y)
        arrays(out/f'{mode}_model.npz',mean=mean,scale=scale,inverse=inverse,
               coefficient=coeff,output_scale=np.array(output_scale))
        models[mode]=dict(sha256=sha(out/f'{mode}_model.npz'),solve_residual=residual)
    once(out/'models_frozen.json',dict(models=models,train_ranks=train,test_ranks=test,
        test_labels_accessed=False,source_seal_sha256=sha(D/'source_seal.json'),
        gram_spectrum=spectrum.tolist(),target_files=provenance))
    tx,ts,tb,_,_=load(test,records,cfg,False)
    summary=tx.mean(1);tz=np.c_[np.ones(len(test)),np.clip((summary-mean)/scale,-8,8)]
    rb=response(tx,ts)/output_scale@inverse
    for mode in models:
        with np.load(out/f'{mode}_model.npz') as saved:coeff=saved['coefficient']
        features=tz[:,:1] if mode=='intercept' else tz
        delta=output_scale*np.einsum('nma,na->nm',rb,features@coeff,optimize=True)
        prediction=np.maximum(tb+delta,0).astype(np.float32).astype(np.float64)
        assert prediction.shape==(32,1024) and np.isfinite(prediction).all()
        with (out/f'{mode}_prediction.npy').open('xb') as stream:np.save(stream,prediction)
        models[mode]['prediction_sha256']=sha(out/f'{mode}_prediction.npy')
    once(out/'result.json',dict(fold=fold,models=models,train_ranks=train,test_ranks=test,
        test_labels_accessed=False,check32_accessed=False,valid40_accessed=False,
        source_seal_sha256=sha(D/'source_seal.json'),elapsed_s=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(json.dumps(dict(fold=fold,models_frozen=True,elapsed_s=time.monotonic()-started)),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='qualify':qualify()
    else:run(int(sys.argv[1]))
