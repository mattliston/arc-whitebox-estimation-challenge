import time
from common import *
verify();clean_watch('seal')
sealed=read(HERE/'captures_sealed.json')
assert sealed['completed'] and sealed['targets_decoded']==False and sealed['train']==TRAIN
for p in sealed['pins']:check(p)
import numpy as np
import span

def compile_fit(model):
    ctrl=model['beta']/model['sx']
    weights=model['gamma']/model['sr'][model['keep']]
    pos=(model['beta']-model['projection'][:,model['keep']]@weights)/model['sx']
    neg=np.zeros(len(model['sz']))
    neg[model['keep']]=weights/model['sz'][model['keep']]
    return ctrl,pos,neg

def compiled_certificate():
    t=np.linspace(-1,1,40)
    x=np.stack((np.ones_like(t),t,t*t),axis=1)
    z=np.stack((np.sin(t),np.cos(t),t**3),axis=1)
    model=span.fit(x,z,np.sin(3*t),1.2+.2*t)
    c,p,n=compile_fit(model)
    a,b=span.predict(model,x,z)
    errors=[float(np.max(np.abs(a-x@c))),float(np.max(np.abs(b-(x@p+z@n))))]
    assert max(errors)<1e-11,errors
    return dict(compilation_errors=errors,original_algebra=span.certificate())

def serve(packet,positive,negative=None):
    hp=packet['hplus'];s=packet['sigma'].astype(np.float32)
    term=hp@positive[1:]+positive[0]
    if negative is not None:term=term+packet['hminus']@negative
    correction=(s*term).astype(np.float64)
    return np.maximum(packet['host']+(packet['delta'].astype(np.float64)+correction),0).astype(np.float32)

started=time.monotonic();cert=compiled_certificate();meta=read(HERE/'inputs.json');data=[]
for r in sealed['rows']:
    with np.load(r['capture']['packet']['path'],allow_pickle=False) as z:packet={k:z[k] for k in z.files}
    s=packet['sigma'].astype(np.float64)
    x=s[:,None]*np.concatenate((np.ones((1024,1)),packet['hplus'].astype(np.float64)),axis=1)
    z=s[:,None]*packet['hminus'].astype(np.float64)
    parent=np.load(r['baseline']['record']['prediction']['path'],allow_pickle=False)[15].astype(np.float64)
    assert np.array_equal(serve(packet,np.zeros(65,np.float32)),parent.astype(np.float32))
    data.append(dict(row=r,packet=packet,x=x,z=z,parent=parent))
write(HERE/'before_targets.json',dict(captures=pin(HERE/'captures_sealed.json'),certificate=cert,
    train=TRAIN,targets_decoded=False,source=pin(HERE/'information.py')))
for d in data:
    tp=meta['targets'][str(d['row']['rank'])];check(tp)
    with np.load(tp['path'],allow_pickle=False) as z:truth=z['mc_post'][15].astype(np.float64)
    assert truth.shape==(1024,) and np.isfinite(truth).all()
    d['truth']=truth;d['y']=truth-d['parent']
predictions={k:[None]*20 for k in ('control','extended')};fits=[]
for fold in range(5):
    tr=[i for i in range(20) if i%5!=fold];va=[i for i in range(20) if i%5==fold]
    x=np.concatenate([data[i]['x'] for i in tr]);z=np.concatenate([data[i]['z'] for i in tr]);y=np.concatenate([data[i]['y'] for i in tr])
    w=np.concatenate([np.full(1024,data[i]['row']['baseline']['record']['multiplier']) for i in tr])
    model=span.fit(x,z,y,w);c,p,n=compile_fit(model)
    for i in va:
        predictions['control'][i]=serve(data[i]['packet'],c.astype(np.float32))
        predictions['extended'][i]=serve(data[i]['packet'],p.astype(np.float32),n.astype(np.float32))
    path=HERE/f'fold_{fold}.npz'
    with path.open('xb') as f:np.savez(f,**model,compiled_control=c.astype(np.float32),compiled_positive=p.astype(np.float32),compiled_negative=n.astype(np.float32))
    fits.append(dict(fold=fold,train=[TRAIN[i] for i in tr],validation=[TRAIN[i] for i in va],
        control_rank=model['rank'],extension_columns=int(model['keep'].sum()),orthogonality=model['orthogonality'],payload=pin(path)))
arrays={k:np.stack(v) for k,v in predictions.items()}
assert all(a.shape==(20,1024) and a.dtype==np.float32 and np.isfinite(a).all() for a in arrays.values())
with (HERE/'oof_predictions.npz').open('xb') as f:np.savez(f,**arrays)
write(HERE/'oof_sealed.json',dict(predictions=pin(HERE/'oof_predictions.npz'),fits=fits,hyperparameter_selection=False,
    targets=meta['targets'],before_pooled_scoring=True))
rows=[]
for i,d in enumerate(data):
    r=d['row'];row=dict(rank=r['rank'],fold=i%5)
    base=r['baseline']['record']['flops'];extra=r['capture']['flops']
    for arm,flops in [('unchanged',base),('control',base+1_000_000),('extended',extra+2_000_000)]:
        pred=d['parent'] if arm=='unchanged' else arrays[arm][i].astype(np.float64)
        raw=float(np.mean((pred-d['truth'])**2))
        row[arm]=dict(raw=raw,flops=flops,adjusted=raw*max(.1,flops/2**41))
    rows.append(row)
arms=('unchanged','control','extended')
means={a:float(np.mean([r[a]['adjusted'] for r in rows])) for a in arms}
folds=[{a:float(np.mean([r[a]['adjusted'] for r in rows if r['fold']==f])) for a in arms} for f in range(5)]
wins=sum(f['extended']<min(f['unchanged'],f['control']) for f in folds)
passed=means['extended']<=.95*means['unchanged'] and means['extended']<=.97*means['control'] and wins>=4
write(HERE/'information_result.json',dict(completed=True,passed=bool(passed),rows=rows,means=means,folds=folds,fold_wins=wins,
    extended_to_parent=means['extended']/means['unchanged'],extended_to_control=means['extended']/means['control'],
    exploratory_network_cv=True,earlier_parent_training_overlap=True,actual_dual_capture_cost_included=True,
    deployment_bill_qualified=False,readout_reserves=dict(control=1_000_000,extended=2_000_000),
    fits=fits,certificate=cert,predictions_seal=pin(HERE/'oof_sealed.json'),wall_s=time.monotonic()-started,
    official_changed=False,goal_achieved=False))
print(dict(completed=True,passed=bool(passed),means=means,fold_wins=wins,
    ratio=means['extended']/means['unchanged'],control_ratio=means['extended']/means['control']),flush=True)
