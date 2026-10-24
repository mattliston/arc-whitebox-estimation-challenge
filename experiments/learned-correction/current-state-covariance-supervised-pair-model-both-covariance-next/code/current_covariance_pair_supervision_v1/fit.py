import sys,gc,time,math
import numpy as np
import torch
from common import *
from projection import project
from pair_base import Pair,inputs
from data import packet,response,truth_covariance
sys.path.insert(1,str(OLDPAIR))
from consumer_v4 import tensors,evaluate

verify();verify('fit_seal.json');torch.set_num_threads(1);torch.manual_seed(2026092743)
for name in ('baseline','capture'):
    g=read(HERE/f'{name}_guard.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
captures=read(HERE/'capture/complete.json')
assert len(captures['rows'])==8 and all(r['all16_bytes_exact'] for r in captures['rows'])
for row in captures['rows']:assert pin(row['packet']['path'])==row['packet']
with np.load(OLDPAIR/'parent_parameters.npz') as z:P=tensors({k:z[k] for k in z.files})
start=time.monotonic();features=[];rhos=[];thirds=[];zero=[]
for mini in ALL:
    a=packet(mini)
    with torch.no_grad():base,_=evaluate(a,P,a['v'])
    actual=a['prediction'][-1].numpy();gap=np.abs(base.numpy()-actual);bound=2*np.abs(np.spacing(actual.astype(np.float32))).astype(float)+1e-9
    assert np.all(gap<=bound),(mini,float(gap.max()),float((gap/bound).max()))
    zero.append(dict(mini=mini,maximum_absolute=float(gap.max()),maximum_fraction_bound=float((gap/bound).max())))
    m=a['prediction'][14];assert float((m*m).max()/(m*m).sum())<.5
    if mini in TRAIN:
        s=a['s14'];R=a['C']/torch.outer(s,s);S=a['K']/torch.outer(s*s,s);mask=~torch.eye(len(s),dtype=torch.bool)
        features.append(a['F'].double().square().mean(0));rhos.append(R[mask].square().mean());thirds.append(S[mask].square().mean())
    del a;gc.collect()
norm=dict(feature_scale=torch.stack(features).mean(0).sqrt().clamp_min(1e-5).float(),rho_scale=torch.stack(rhos).mean().sqrt().clamp_min(1e-5).float(),third_scale=torch.stack(thirds).mean().sqrt().clamp_min(1e-5).float())
with (HERE/'normalizers.npz').open('xb') as f:np.savez(f,**{k:v.numpy() for k,v in norm.items()})
once(HERE/'before_targets.json',dict(normalizers=pin(HERE/'normalizers.npz'),zero_replays=zero,targets_accessed=False))
targetdir=HERE/'targets';targetdir.mkdir();targetrows=[]
for mini in TRAIN:
    a=packet(mini);T,v,vr,info=truth_covariance(mini,a)
    p=targetdir/f'mini{mini:04d}.npz'
    with p.open('xb') as f:np.savez(f,T=T.numpy(),variance=v.numpy(),response=vr.numpy())
    info['target']=pin(p);targetrows.append(info)
    print(json.dumps(dict(event='target',mini=mini,affine_covariance_relative=info['affine_covariance_relative'],covariance_energy=info['baseline_covariance_energy'])),flush=True)
    del a,T,v,vr;gc.collect()
once(HERE/'train_targets.json',dict(rows=targetrows,check_targets_accessed=False,final_mean_targets_accessed=False))
def target(mini):
    with np.load(targetdir/f'mini{mini:04d}.npz') as z:return {k:torch.from_numpy(z[k].copy()) for k in z.files}
model=Pair();assert sum(p.numel() for p in model.parameters())==6177
optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
rng=np.random.default_rng(2026092743);step=0;updates=360
for epoch in range(60):
    losses=[];maxgrad=0.
    for mini in rng.permutation(TRAIN):
        mini=int(mini);a=packet(mini);t=target(mini)
        lr=.00001+(.001-.00001)*.5*(1+math.cos(math.pi*step/(updates-1)))
        for group in optimizer.param_groups:group['lr']=lr
        D=project(model(*inputs(a,norm)),a['prediction'][14]);dv,vr=response(a,D)
        eC=(D-t['T']).square().mean()/t['T'].square().mean()
        eV=(vr-t['response']).square().mean()/t['response'].square().mean()
        loss=.5*eC+.5*eV+.001*D.square().mean()/t['T'].square().mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        assert torch.isfinite(gn);optimizer.step();step+=1;maxgrad=max(maxgrad,float(gn));losses.append((float(eC.detach()),float(eV.detach())))
        del a,t,D,dv,vr,eC,eV,loss;gc.collect()
    row=dict(epoch=epoch+1,updates=step,online_covariance_ratio=float(np.mean([x[0] for x in losses])),online_response_ratio=float(np.mean([x[1] for x in losses])),maximum_gradient_norm=maxgrad,lr=lr,elapsed_s=time.monotonic()-start)
    with (HERE/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
with (HERE/'model.npz').open('xb') as f:np.savez(f,**{k:v.detach().numpy() for k,v in model.state_dict().items()})
once(HERE/'model_frozen.json',dict(model=pin(HERE/'model.npz'),normalizers=pin(HERE/'normalizers.npz'),updates=step,epochs=60,check_targets_accessed=False,final_mean_targets_accessed=False))
del optimizer;model.eval();predir=HERE/'predictions';predir.mkdir();predictionrows=[]
with torch.no_grad():
    for mini in ALL:
        a=packet(mini);D=project(model(*inputs(a,norm)),a['prediction'][14]);dv,vr=response(a,D)
        changed_v=a['v']+dv;assert bool(torch.all(changed_v>1e-10))
        original,_=evaluate(a,P,a['v']);changed,_=evaluate(a,P,changed_v);forecast=a['prediction'][-1]+changed-original
        assert torch.isfinite(D).all() and torch.isfinite(forecast).all()
        p=predir/f'mini{mini:04d}.npz'
        with p.open('xb') as f:np.savez(f,D=D.numpy(),variance=dv.numpy(),response=vr.numpy(),forecast=forecast.numpy())
        m=a['prediction'][14];violation=float(torch.max(torch.abs(D@(m/torch.linalg.vector_norm(m)))))
        assert violation<1e-11 and bool(torch.all(torch.diagonal(D)==0))
        predictionrows.append(dict(mini=mini,packet=pin(p),annihilation_max_abs=violation,relative_variance_max=float(torch.max(torch.abs(dv/a['v']))),minimum_variance=float(changed_v.min())))
        del a,D,dv,vr;gc.collect()
once(HERE/'predictions_frozen.json',dict(rows=predictionrows,model=pin(HERE/'model.npz'),check_targets_accessed=False,final_mean_targets_accessed=False))
for mini in CHECK:
    a=packet(mini);T,v,vr,info=truth_covariance(mini,a);p=targetdir/f'mini{mini:04d}.npz'
    with p.open('xb') as f:np.savez(f,T=T.numpy(),variance=v.numpy(),response=vr.numpy())
    info['target']=pin(p);targetrows.append(info);del a,T,v,vr;gc.collect()
rows=[]
for mini in ALL:
    with np.load(predir/f'mini{mini:04d}.npz') as p,np.load(targetdir/f'mini{mini:04d}.npz') as t:
        ratios={key:float(np.sum((p[pkey]-t[tkey])**2)/np.sum(t[tkey]**2)) for key,pkey,tkey in (('covariance','D','T'),('response','response','response'))}
    rows.append(dict(mini=mini,split='TRAIN' if mini in TRAIN else 'CHECK',ratios=ratios))
held=[x for x in rows if x['split']=='CHECK'];gates={str(x['mini']):x['ratios']['covariance']<=.90 and x['ratios']['response']<=.85 for x in held}
once(HERE/'result.json',dict(passed=all(gates.values()),gates=gates,rows=rows,reference_checks=targetrows,model=pin(HERE/'model.npz'),normalizers=pin(HERE/'normalizers.npz'),seal_sha256=sha(HERE/'seal.json'),fit_seal_sha256=sha(HERE/'fit_seal.json'),elapsed_s=time.monotonic()-start,final_mean_targets_accessed=False,native_fitted_candidate=False,single_reference_noise_removed=False))
print(json.dumps(dict(event='complete',passed=all(gates.values()),gates=gates,check=held)),flush=True)
