import resource,time
from datetime import datetime,timezone
from common import *
import numpy as np
from threadpoolctl import threadpool_info
from math_core import initialize,forward,backward,losses
verify();verify('fit_sources_sealed.json');assert all(x['num_threads']==1 for x in threadpool_info())
assert read(HERE/'train_targets_completed.json')['completed'];start=time.monotonic();cells=[];provenance=[]
for ordinal in TRAIN:
    ir=read(HERE/f'inputs_mini{ordinal:04d}.json');tr=read(HERE/f'targets_mini{ordinal:04d}.json');assert ir['passed'] and tr['passed']
    assert tr['role']=='train';provenance.extend([pin(HERE/f'inputs_mini{ordinal:04d}.json'),pin(HERE/f'targets_mini{ordinal:04d}.json')])
    for ri,rt in zip(ir['records'],tr['targets']):
        assert ri['layer']==rt['layer'];check(ri['input']);check(rt['target'])
        with np.load(ri['input']['path'],allow_pickle=False) as z:c={k:z[k] for k in ('F','R','S','L','weight','Wnext','gnext')}
        c.update(target=np.load(rt['target']['path'],allow_pickle=False),ordinal=ordinal,layer=ri['layer']);cells.append(c)
assert len(cells)==42
fs=np.zeros(140,np.float64);rs=ss=ts=0.;n=1024;off=n*(n-1)
energies=np.zeros((7,3),np.float64)
for c in cells:
    fs+=np.mean(c['F'].astype(np.float64)**2,axis=0)
    for key in ('R','S'):
        v=c[key].astype(np.float64);value=(np.sum(v*v)-np.sum(np.diag(v)**2))/off
        if key=='R':rs+=value
        else:ss+=value
    ts+=np.sum(c['target']**2)/off
    energies[c['layer']-8]+=losses(c,np.zeros((n,n)))
energies/=6;assert np.isfinite(energies).all() and np.min(energies)>0
norm=dict(feature_scale=np.maximum(np.sqrt(fs/42),1e-5).astype(np.float32),rho_scale=np.array(max(np.sqrt(rs/42),1e-5),np.float32),
          third_scale=np.array(max(np.sqrt(ss/42),1e-5),np.float32),target_scale=np.array(np.sqrt(ts/42),np.float32))
assert all(np.isfinite(v).all() and np.min(v)>0 for v in norm.values())
for c in cells:
    c['F']=np.clip(c['F']/norm['feature_scale'],-8,8);c['R']=c['R']/norm['rho_scale'];c['S']=c['S']/norm['third_scale']
np.savez(HERE/'normalizers.npz',**norm,energies=energies)
t=initialize();m={k:np.zeros_like(t[k],dtype=np.float64) for k in KEYS};v={k:np.zeros_like(t[k],dtype=np.float64) for k in KEYS}
rng=np.random.default_rng(712031);updates=0;epochs=[]
for epoch in range(24):
    summed=np.zeros(3);maxgrad=0.;clipped=0;lr=.002*.98**epoch
    for index in rng.permutation(42):
        c=cells[index];e=energies[c['layer']-8];pred=forward(c['F'],c['R'],c['S'],t,norm['target_scale'])
        value,terms,d=losses(c,pred,e,True);assert np.isfinite(value) and np.isfinite(d).all()
        grads=backward(c['F'],c['R'],c['S'],t,norm['target_scale'],d)
        gn=float(np.sqrt(sum(np.sum(g*g) for g in grads.values())));assert np.isfinite(gn)
        maxgrad=max(maxgrad,gn);scale=min(1.,5/max(gn,1e-30));clipped+=int(scale<1);updates+=1
        for k in KEYS:
            g=grads[k]*scale;m[k]=.9*m[k]+.1*g;v[k]=.999*v[k]+.001*g*g
            update=(m[k]/(1-.9**updates))/(np.sqrt(v[k]/(1-.999**updates))+1e-8)
            t[k]=(t[k].astype(np.float64)-lr*update).astype(np.float32);assert np.isfinite(t[k]).all()
        summed+=terms/e
    rec=dict(epoch=epoch+1,updates=updates,learning_rate=lr,online_pre_update_normalized_terms=(summed/42).tolist(),max_gradient=maxgrad,
             clipped_steps=clipped,elapsed_s=time.monotonic()-start)
    epochs.append(rec);print(rec,flush=True);resource_floor()
assert updates==1008
np.savez(HERE/'model.npz',**t,**norm)
train=[]
for ordinal in TRAIN:
    numerator=np.zeros(3);denominator=np.zeros(3)
    for c in cells:
        if c['ordinal']!=ordinal:continue
        e=energies[c['layer']-8];pred=forward(c['F'],c['R'],c['S'],t,norm['target_scale'])
        numerator+=losses(c,pred)/e;denominator+=losses(c,np.zeros((n,n)))/e
    train.append(dict(ordinal=ordinal,ratios=(numerator/denominator).tolist()))
write(HERE/'training_result.json',dict(completed=True,epochs=epochs,updates=updates,final_training=train,model=pin(HERE/'model.npz'),
      normalizers=pin(HERE/'normalizers.npz'),provenance=provenance,heldout_payloads_opened=0,public1B_labels_opened=0,
      wall_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
for ordinal in HELDOUT:assert not (HERE/f'download_mini{ordinal:04d}.json').exists() and not (HERE/f'targets_mini{ordinal:04d}.json').exists()
write(HERE/'model_frozen.json',dict(training_completed=True,utc=datetime.now(timezone.utc).isoformat(),model=pin(HERE/'model.npz'),
      normalizers=pin(HERE/'normalizers.npz'),training=pin(HERE/'training_result.json'),heldout_decoded=False,final_epoch=24,
      architecture_or_amplitude_selection=False))
print(dict(model_frozen=True,train_ratios=train,wall_s=time.monotonic()-start),flush=True)
