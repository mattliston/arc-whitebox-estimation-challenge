"""Fixed width6 hidden-feature distillation of sampled msg8 edge; no network truth."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import numpy as np

TRAIN=(35,150,200,250)
VALID=(259,277,287)
LAYERS=tuple(range(8,15))
WIDTH=6
STEPS=5000
BATCH=8192
RATE=.001
SEED=20261020
TEACHER_SHA='777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()

def write(p,x):
    with Path(p).open('x') as f:
        json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

def activation(z,a,b):
    s=np.sqrt(b+z*z)
    d=a+s
    raw=z/d
    derivative=(a+b/s)/(d*d)
    return np.clip(raw,-1,1),np.where(np.abs(raw)<1,derivative,0)

def read_array(directory,record):
    path=directory/record['path']
    assert sha(path)==record['sha256'],path
    return np.fromfile(path,dtype=record['dtype']).reshape(record['shape'])

def physical_record(directory,rec,meta):
    z=read_array(directory,rec['pre']).astype(np.float64)
    h=read_array(directory,rec['h']).astype(np.float64)
    assert z.shape==h.shape and z.shape[1]==14
    a,b=(float(np.float32(meta['activation'][k])) for k in ('a','b'))
    replay,_=activation(z,a,b)
    replay_error=float(np.max(np.abs(replay-h)))
    assert replay_error<2e-6,('pre/post mismatch',rec['rank'],rec['layer'],replay_error)
    pf,gf,sigma=read_array(directory,rec['vectors']).astype(np.float64)
    ij=np.asarray(rec['pairs'],dtype=np.int64); i,j=ij.T
    f,r=np.asarray(rec['forward']),np.asarray(rec['reverse'])
    vf,vr=f>=0,r>=0
    f=np.maximum(f,0);r=np.maximum(r,0)
    scale=.5*sigma[i]*sigma[j]*float(np.float32(rec['cov_factor']))*float(np.float32(rec['gn_multiplier']))
    af=scale*pf[i]*gf[j]*vf
    ar=scale*pf[j]*gf[i]*vr
    X=h[f]*af[:,None]+h[r]*ar[:,None]
    output=np.asarray(rec['output'],dtype=np.float32).astype(np.float64)
    target=X@output
    off=rec['offdiag_count']
    weights=np.ones(len(i),dtype=np.float64)
    if off:weights[:off]=2*rec['total_supported_offdiag']/off
    energy=float(weights@(target*target));assert energy>0
    weights/=energy
    return dict(rank=rec['rank'],layer=rec['layer'],zf=z[f],zr=z[r],af=af,ar=ar,
                target=target,weights=weights,G=X.T@(X*weights[:,None]),
                output=output,replay_error=replay_error)

def collect(root):
    records=[];metadata=None;provenance=[];seen=set()
    for path in sorted(root.rglob('MANIFEST.json')):
        manifest=json.loads(path.read_text())
        if 'records' not in manifest or 'metadata_sha256' not in manifest:continue
        assert manifest['uses_network_truth'] is False
        assert manifest['arithmetic_AST_identical_after_removing_capture'] is True
        directory=path.parent;mp=directory/'METADATA_v1.json'
        assert sha(mp)==manifest['metadata_sha256']
        meta=json.loads(mp.read_text())
        assert meta['uses_network_truth'] is False and meta['hidden']==14 and meta['message_width']==8
        assert meta['teacher_source_sha256']==TEACHER_SHA
        assert tuple(meta['recipe']['train_ranks'])==TRAIN
        assert tuple(meta['recipe']['validation_ranks'])==VALID
        if metadata is None:metadata=meta
        else:assert metadata==meta
        for item in manifest['records']:
            rp=directory/item['path'];assert sha(rp)==item['sha256']
            rec=json.loads(rp.read_text());key=rec['rank'],rec['layer']
            assert key not in seen;seen.add(key)
            assert rec['teacher_source_sha256']==meta['teacher_source_sha256']
            records.append(physical_record(directory,rec,meta))
        provenance.append(dict(path=str(path),sha256=sha(path),metadata_sha256=sha(mp)))
    assert seen=={(r,l) for r in TRAIN+VALID for l in LAYERS},('incomplete capture',sorted(seen))
    return records,metadata,provenance

def pool(records):
    keys=('zf','zr','af','ar','target','weights')
    d={k:np.concatenate([r[k] for r in records]) for k in keys}
    d['weights']/=len(records)
    return d

def unpack(v):
    return v[:14*WIDTH].reshape(14,WIDTH),v[14*WIDTH:15*WIDTH],v[15*WIDTH:16*WIDTH]

def forward(v,d,a,b):
    Q,c,o=unpack(v)
    hf,df=activation(d['zf']@Q+c,a,b)
    hr,dr=activation(d['zr']@Q+c,a,b)
    return (hf@o)*d['af']+(hr@o)*d['ar'],hf,hr,df,dr

def objective(v,d,a,b,anchor,gradient=True,factor=1.):
    pred,hf,hr,df,dr=forward(v,d,a,b)
    err=pred-d['target']
    loss=float(factor*(d['weights']@(err*err)))
    Q,c,o=unpack(v)
    scales=np.concatenate((np.full(14*WIDTH,1e-5/(14*WIDTH)),np.full(WIDTH,1e-5/WIDTH),
                           np.full(WIDTH,1e-4/(WIDTH*max(float(np.mean(anchor[15*WIDTH:]**2)),1e-12)))))
    delta=v-anchor
    loss+=float(scales@(delta*delta))
    if not gradient:return loss
    e=2*factor*d['weights']*err
    ef=e*d['af'];er=e*d['ar']
    gf=ef[:,None]*o[None,:]*df
    gr=er[:,None]*o[None,:]*dr
    dq=d['zf'].T@gf+d['zr'].T@gr
    dc=gf.sum(axis=0)+gr.sum(axis=0)
    do=hf.T@ef+hr.T@er
    g=np.concatenate((dq.ravel(),dc,do))+2*scales*delta
    assert np.isfinite(g).all() and np.isfinite(loss)
    return loss,g

def metrics(v,records,a,b):
    result=[]
    for r in records:
        pred,*_=forward(v,r,a,b)
        result.append(dict(rank=r['rank'],layer=r['layer'],
             relative_mse=float(r['weights']@((pred-r['target'])**2))))
    return dict(mean_relative_mse=float(np.mean([x['relative_mse'] for x in result])),
                per_rank={str(k):float(np.mean([x['relative_mse'] for x in result if x['rank']==k]))
                          for k in sorted({x['rank'] for x in result})},per_layer=result)

def selftest():
    rng=np.random.default_rng(17)
    v=np.concatenate((np.eye(14)[:,:WIDTH].ravel(),np.zeros(WIDTH),rng.normal(0,.3,WIDTH)))
    d=dict(zf=rng.normal(0,.5,(40,14)),zr=rng.normal(0,.5,(40,14)),
           af=rng.normal(0,.3,40),ar=rng.normal(0,.3,40),
           target=rng.normal(0,.2,40),weights=np.full(40,1/40))
    a,b=-.2684473340048872,1.5215732663431163
    anchor=v.copy();v=v+rng.normal(0,.01,len(v))
    loss,g=objective(v,d,a,b,anchor)
    errors=[]
    for k in (0,15,7*WIDTH,14*WIDTH-1,14*WIDTH,15*WIDTH-1,15*WIDTH,16*WIDTH-1):
        plus=v.copy();minus=v.copy();plus[k]+=1e-6;minus[k]-=1e-6
        fd=(objective(plus,d,a,b,anchor,False)-objective(minus,d,a,b,anchor,False))/2e-6
        errors.append(abs(fd-g[k]))
    assert max(errors)<1e-7,errors
    assert objective(v-.01*g,d,a,b,anchor,False)<loss
    raw=rng.normal(size=(21,14));Q,c,_=unpack(v);inputs=rng.normal(size=(11,21))
    np.testing.assert_allclose((inputs@raw)@Q+c,inputs@(raw@Q)+c,rtol=1e-11,atol=1e-11)
    return dict(status='ANALYTIC_GRADIENT_AND_FOLD_PASS',max_gradient_abs_error=max(errors))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--captures',type=Path)
    p.add_argument('--out',type=Path)
    p.add_argument('--ridge-fitter',type=Path)
    a=p.parse_args()
    check=selftest()
    if a.self_test:
        print(json.dumps(check));return
    assert a.captures and a.out and a.ridge_fitter and not a.out.exists()
    a.out.mkdir()
    write(a.out/'PREREG_v1.json',dict(utc=utc(),width=WIDTH,steps=STEPS,batch=BATCH,
          learning_rate=RATE,seed=SEED,train=TRAIN,validation=VALID,
          objective='equal network/layer relative physical covariance MSE; frozen anchored Adam',
          checkpoint_selection='minimum complete training objective every250steps; validation never used',
          source_sha256=sha(__file__),ridge_source_sha256=sha(a.ridge_fitter),
          uses_network_truth=False,check=check))
    records,meta,provenance=collect(a.captures)
    tr=[r for r in records if r['rank'] in TRAIN];va=[r for r in records if r['rank'] in VALID]
    assert len(tr)==28 and len(va)==21
    spec=importlib.util.spec_from_file_location('frozen_ridge',a.ridge_fitter)
    ridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(ridge)
    G=sum(r['G'] for r in tr)/len(tr);original=tr[0]['output']
    assert all(np.array_equal(r['output'],original) for r in records)
    init=ridge.fit_width(G,original,WIDTH,1e-6)
    Q=np.eye(14)[:,init['keep']]
    anchor=np.concatenate((Q.ravel(),np.zeros(WIDTH),np.asarray(init['output'])))
    v=anchor.copy();d=pool(tr);N=len(d['target'])
    aa,bb=(float(np.float32(meta['activation'][k])) for k in ('a','b'))
    best=v.copy();bestloss=objective(v,d,aa,bb,anchor,False)
    m=np.zeros_like(v);s=np.zeros_like(v);rng=np.random.default_rng(SEED)
    history=[];start=time.monotonic()
    for step in range(1,STEPS+1):
        ix=rng.integers(0,N,size=min(BATCH,N))
        batch={k:val[ix] for k,val in d.items()}
        _,g=objective(v,batch,aa,bb,anchor,factor=N/len(ix))
        norm=float(np.linalg.norm(g))
        if norm>10:g*=10/norm
        m=.9*m+.1*g;s=.999*s+.001*g*g
        v-=RATE*(m/(1-.9**step))/(np.sqrt(s/(1-.999**step))+1e-8)
        if step%250==0 or step==1:
            loss=objective(v,d,aa,bb,anchor,False)
            if loss<bestloss:best,bestloss=v.copy(),loss
            row=dict(utc=utc(),step=step,train_objective=loss,best_train_objective=bestloss,
                     elapsed_s=time.monotonic()-start)
            history.append(row);print(json.dumps(row),flush=True)
    best=best.astype(np.float32).astype(np.float64)
    Q,c,o=unpack(best)
    edge={k:np.asarray(v,dtype=np.float32) if isinstance(v,list) else v
          for k,v in meta['edge_literal'].items()}
    folded={}
    for key,value in edge.items():
        if key in ('left','right','edge'):
            assert value.shape[-1]==14,(key,value.shape)
            folded[key]=(value@Q.astype(np.float32)).astype(np.float32).tolist()
        elif key=='bias':
            folded[key]=(value@Q.astype(np.float32)+c.astype(np.float32)).astype(np.float32).tolist()
        elif key=='output':folded[key]=o.astype(np.float32).tolist()
        else:folded[key]=value.tolist() if isinstance(value,np.ndarray) else value
    context={key:(np.asarray(value,dtype=np.float32)@Q.astype(np.float32)).astype(np.float32).tolist()
             for key,value in meta['message_edge_literal'].items()}
    result=dict(utc=utc(),status='FIXED_HIDDEN_MIX_FIT_NO_NATIVE_SCORE',width=WIDTH,
                teacher_source_sha256=meta['teacher_source_sha256'],fit_source_sha256=sha(__file__),
                captures=provenance,recipe=meta['recipe'],activation=meta['activation'],
                initialize=init,Q=Q.tolist(),offset=c.tolist(),output=o.tolist(),
                edge_literal=folded,message_edge_literal=context,
                train_before=metrics(anchor,tr,aa,bb),train_after=metrics(best,tr,aa,bb),
                validation_before=metrics(anchor,va,aa,bb),validation_after=metrics(best,va,aa,bb),
                history=history,uses_network_truth=False,network_score_gain_claimed=False,
                deployment='Fold Q and offset into existing literals; hidden6; no extra inference operations.',
                caveat='Approximate fit on teacher trajectories; full native32 and held25 required.')
    write(a.out/'FITTED_v1.json',result)
    print(json.dumps({k:result[k] for k in ('status','train_before','train_after','validation_before','validation_after')}),flush=True)

if __name__=='__main__':
    main()

