import resource,time
from datetime import datetime,timezone
from common import *
import numpy as np
from threadpoolctl import threadpool_info
from data import load_cell
from math_core import initialize,forward,backward,losses
verify();verify('fit_sources_sealed.json');assert all(x['num_threads']==1 for x in threadpool_info())
assert read(HERE/'train_targets_completed.json')['completed'];start=time.monotonic();cells=[load_cell(o,l) for o in TRAIN for l in LAYERS]
fs=np.zeros(140,np.float64);rs=ss=0.;ts=np.zeros(3,np.float64);energy=np.zeros((7,5),np.float64);off=1024*1023
for c in cells:
    fs+=np.mean(c['F'].astype(np.float64)**2,axis=0)
    for k in ('R','S'):
        x=c[k].astype(np.float64);value=(np.sum(x*x)-np.sum(np.diag(x)**2))/off
        if k=='R':rs+=value
        else:ss+=value
    ts+=np.sum(c['target']**2,axis=(1,2))/off;energy[c['layer']-8]+=losses(c,np.zeros((3,1024,1024)))
energy/=6;assert np.isfinite(energy).all() and np.min(energy)>0
norm=dict(feature_scale=np.maximum(np.sqrt(fs/42),1e-5).astype(np.float32),rho_scale=np.array(max(np.sqrt(rs/42),1e-5),np.float32),
          third_scale=np.array(max(np.sqrt(ss/42),1e-5),np.float32),target_scale=np.sqrt(ts/42).astype(np.float32))
assert all(np.isfinite(x).all() and np.min(x)>0 for x in norm.values())
for c in cells:c['F']=np.clip(c['F']/norm['feature_scale'],-8,8);c['R']=c['R']/norm['rho_scale'];c['S']=c['S']/norm['third_scale']
np.savez(HERE/'normalizers.npz',**norm,energies=energy)
t=initialize();m={k:np.zeros_like(t[k],dtype=np.float64) for k in KEYS};v={k:np.zeros_like(t[k],dtype=np.float64) for k in KEYS}
rng=np.random.default_rng(923071);epochs=[];updates=0
for epoch in range(32):
    terms_sum=np.zeros(5);maxgrad=0.;clipped=0;lr=.002*.98**epoch
    for index in rng.permutation(42):
        c=cells[index];e=energy[c['layer']-8];pred=forward(c['F'],c['R'],c['S'],t,norm['target_scale'])
        value,terms,d=losses(c,pred,e,True);assert np.isfinite(value) and np.isfinite(d).all()
        grads=backward(c['F'],c['R'],c['S'],t,norm['target_scale'],d);gn=float(np.sqrt(sum(np.sum(g*g) for g in grads.values())))
        assert np.isfinite(gn);scale=min(1.,5/max(gn,1e-30));maxgrad=max(maxgrad,gn);clipped+=int(scale<1);updates+=1
        for k in KEYS:
            g=grads[k]*scale;m[k]=.9*m[k]+.1*g;v[k]=.999*v[k]+.001*g*g
            update=(m[k]/(1-.9**updates))/(np.sqrt(v[k]/(1-.999**updates))+1e-8)
            t[k]=(t[k].astype(np.float64)-lr*update).astype(np.float32);assert np.isfinite(t[k]).all()
        terms_sum+=terms/e
    record=dict(epoch=epoch+1,updates=updates,learning_rate=lr,online_pre_update_normalized_terms=(terms_sum/42).tolist(),max_gradient=maxgrad,clipped_steps=clipped,elapsed_s=time.monotonic()-start)
    epochs.append(record);print(record,flush=True);resource_floor()
assert updates==1344;np.savez(HERE/'model.npz',**t,**norm);train=[]
for ordinal in TRAIN:
    numerator=np.zeros(5);denominator=np.zeros(5)
    for c in cells:
        if c['ordinal']!=ordinal:continue
        e=energy[c['layer']-8];pred=forward(c['F'],c['R'],c['S'],t,norm['target_scale'])
        numerator+=losses(c,pred)/e;denominator+=losses(c,np.zeros((3,1024,1024)))/e
    train.append(dict(ordinal=ordinal,ratios=(numerator/denominator).tolist()))
write(HERE/'training_result.json',dict(completed=True,epochs=epochs,updates=updates,final_training=train,model=pin(HERE/'model.npz'),normalizers=pin(HERE/'normalizers.npz'),
      new_validation_fourth_fields_decoded=False,public1B_labels_decoded=False,wall_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
for ordinal in VALID:assert not (HERE/f'targets_mini{ordinal:04d}.json').exists()
write(HERE/'model_frozen.json',dict(training_completed=True,utc=datetime.now(timezone.utc).isoformat(),model=pin(HERE/'model.npz'),normalizers=pin(HERE/'normalizers.npz'),
      training=pin(HERE/'training_result.json'),final_epoch=32,new_validation_fourth_fields_decoded=False,reused_validation=True))
print(dict(model_frozen=True,final_training=train,wall_s=time.monotonic()-start),flush=True)
