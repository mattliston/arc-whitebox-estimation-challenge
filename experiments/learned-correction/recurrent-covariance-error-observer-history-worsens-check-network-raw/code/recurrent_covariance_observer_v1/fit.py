import gc,time,math
import numpy as np
import torch
from common import *
from model import Pair,rollout
from data import packet,truth,saved_targets,pack
verify();verify('unit_seal.json');verify('fit_seal.json');torch.set_num_threads(1)
for name in ('unit','capture'):
    g=read(HERE/f'{name}_guard.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
assert read(HERE/'unit_result.json')['passed']
capture=read(HERE/'capture/complete.json');assert len(capture['rows'])==8
for row in capture['rows']:
    assert row['all16_bytes_exact'] and row['bill_exact'] and pin(row['packet']['path'])==row['packet']
start=time.monotonic();features=[];rhos=[];thirds=[]
for mini in TRAIN:
    cells,terminal=packet(mini)
    for a in cells:
        s=a['sigma'];R=a['C']/torch.outer(s,s);S=a['K']/torch.outer(s*s,s);mask=~torch.eye(len(s),dtype=torch.bool)
        features.append(a['F'].double().square().mean(0));rhos.append(R[mask].square().mean());thirds.append(S[mask].square().mean())
    del cells,terminal,a,R,S;gc.collect()
norm=dict(feature_scale=torch.stack(features).mean(0).sqrt().clamp_min(1e-5).float(),rho_scale=torch.stack(rhos).mean().sqrt().clamp_min(1e-5).float(),third_scale=torch.stack(thirds).mean().sqrt().clamp_min(1e-5).float())
with (HERE/'normalizers.npz').open('xb') as f:np.savez(f,**{k:v.numpy() for k,v in norm.items()})
once(HERE/'before_targets.json',dict(normalizers=pin(HERE/'normalizers.npz'),train=list(TRAIN),check=list(CHECK),targets_accessed=False))
targetdir=HERE/'targets';targetdir.mkdir();targetrows=[]
for mini in TRAIN:
    cells,terminal=packet(mini);targets,final,info=truth(mini,cells)
    info['parent_mse']=float((terminal['parent']-final).square().mean());assert info['parent_mse']>0
    p=targetdir/f'mini{mini:04d}.npz'
    with p.open('xb') as f:np.savez(f,final=final.numpy(),**{f'T_{l}_upper':pack(t.numpy()) for l,t in zip(LAYERS,targets)})
    info['target']=pin(p);targetrows.append(info)
    print(json.dumps(dict(event='target',mini=mini,parent_mse=info['parent_mse'])),flush=True)
    del cells,terminal,targets,final;gc.collect()
once(HERE/'train_targets.json',dict(rows=targetrows,check_targets_accessed=False))
models={};initial=None
for mode in ('history','reset'):
    torch.manual_seed(2026092744);model=Pair();assert sum(p.numel() for p in model.parameters())==6209
    state={k:v.detach().numpy().copy() for k,v in model.state_dict().items()}
    if initial is None:initial=state
    else:assert all(np.array_equal(v,initial[k]) for k,v in state.items())
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    rng=np.random.default_rng(2026092744);step=0;updates=180
    for epoch in range(30):
        losses=[];maxgrad=0.;memory_gradient=0.
        for mini in rng.permutation(TRAIN):
            mini=int(mini);cells,terminal=packet(mini);targets,final=saved_targets(mini)
            lr=.00001+(.001-.00001)*.5*(1+math.cos(math.pi*step/(updates-1)))
            for group in optimizer.param_groups:group['lr']=lr
            Ds,e,forecast=rollout(model,cells,terminal,norm,mode)
            cov=torch.stack([(D-t).square().mean()/t.square().mean() for D,t in zip(Ds,targets)]).mean()
            mean=(forecast-final).square().mean()/(terminal['parent']-final).square().mean()
            regularizer=torch.stack([D.square().mean()/t.square().mean() for D,t in zip(Ds,targets)]).mean()
            loss=.5*cov+.5*mean+.001*regularizer;assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            mg=float(torch.linalg.vector_norm(model.edge.weight.grad[:,3]));memory_gradient=max(memory_gradient,mg)
            if mode=='reset':assert mg==0
            gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(gn)
            optimizer.step();step+=1;maxgrad=max(maxgrad,float(gn));losses.append((float(cov.detach()),float(mean.detach())))
            del cells,terminal,targets,final,Ds,e,forecast,cov,mean,regularizer,loss;gc.collect()
        row=dict(mode=mode,epoch=epoch+1,updates=step,online_covariance_ratio=float(np.mean([x[0] for x in losses])),online_final_raw_ratio=float(np.mean([x[1] for x in losses])),maximum_gradient_norm=maxgrad,maximum_memory_gradient=memory_gradient,lr=lr,elapsed_s=time.monotonic()-start)
        with (HERE/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    p=HERE/f'model_{mode}.npz'
    with p.open('xb') as f:np.savez(f,**{k:v.detach().numpy() for k,v in model.state_dict().items()})
    once(HERE/f'model_{mode}_frozen.json',dict(model=pin(p),normalizers=pin(HERE/'normalizers.npz'),updates=step,epochs=30,check_targets_accessed=False))
    models[mode]=model.eval();del optimizer;gc.collect()
predir=HERE/'predictions';predir.mkdir();predictionrows=[]
with torch.no_grad():
    for mini in ALL:
        cells,terminal=packet(mini)
        for mode,model in models.items():
            Ds,e,forecast=rollout(model,cells,terminal,norm,mode)
            assert torch.isfinite(e).all() and torch.isfinite(forecast).all()
            constraint=[]
            for D,a in zip(Ds,cells):
                witness=float(torch.max(torch.abs(D@(a['m']/torch.linalg.vector_norm(a['m'])))))
                assert witness<1e-11 and torch.count_nonzero(torch.diagonal(D))==0 and torch.isfinite(D).all()
                constraint.append(witness)
            p=predir/f'{mode}_mini{mini:04d}.npz'
            with p.open('xb') as f:np.savez(f,e=e.numpy(),forecast=forecast.numpy(),**{f'D_{l}_upper':pack(D.numpy()) for l,D in zip(LAYERS,Ds)})
            predictionrows.append(dict(mini=mini,mode=mode,packet=pin(p),annihilation_max_abs=max(constraint),correction_rms=float(e.square().mean().sqrt())))
            del Ds,e,forecast,D;gc.collect()
        del cells,terminal;gc.collect()
once(HERE/'predictions_frozen.json',dict(rows=predictionrows,models={mode:pin(HERE/f'model_{mode}.npz') for mode in models},check_targets_accessed=False))
for mini in CHECK:
    cells,terminal=packet(mini);targets,final,info=truth(mini,cells)
    info['parent_mse']=float((terminal['parent']-final).square().mean());assert info['parent_mse']>0
    p=targetdir/f'mini{mini:04d}.npz'
    with p.open('xb') as f:np.savez(f,final=final.numpy(),**{f'T_{l}_upper':pack(t.numpy()) for l,t in zip(LAYERS,targets)})
    info['target']=pin(p);targetrows.append(info);del cells,terminal,targets,final;gc.collect()
rows=[]
for mini in ALL:
    for mode in models:
        with np.load(predir/f'{mode}_mini{mini:04d}.npz') as p,np.load(targetdir/f'mini{mini:04d}.npz') as t:
            cov=[float(np.sum((p[f'D_{l}_upper']-t[f'T_{l}_upper'])**2)/np.sum(t[f'T_{l}_upper']**2)) for l in LAYERS]
            mse=float(np.mean((p['forecast']-t['final'])**2))
        parent=next(x['parent_mse'] for x in targetrows if x['mini']==mini)
        rows.append(dict(mini=mini,mode=mode,split='TRAIN' if mini in TRAIN else 'CHECK',covariance_ratio=float(np.mean(cov)),covariance_by_layer=cov,mse=mse,parent_mse=parent,raw_ratio=mse/parent))
h=[x for x in rows if x['split']=='CHECK' and x['mode']=='history'];r=[x for x in rows if x['split']=='CHECK' and x['mode']=='reset']
pooled=sum(x['mse'] for x in h)/sum(x['parent_mse'] for x in h);incremental=sum(x['mse'] for x in h)/sum(x['mse'] for x in r)
gates=dict(pooled_raw=pooled<=.90,each_raw=all(x['raw_ratio']<=.95 for x in h),incremental=incremental<=.975,
           both_vs_reset=all(x['mse']<y['mse'] for x,y in zip(h,r)),covariance=np.mean([x['covariance_ratio'] for x in h])<=.90)
gates={k:bool(v) for k,v in gates.items()}
once(HERE/'result.json',dict(passed=all(gates.values()),gates=gates,rows=rows,reference_checks=targetrows,check_pooled_raw_ratio=pooled,check_history_to_reset_ratio=incremental,
    models={mode:pin(HERE/f'model_{mode}.npz') for mode in models},normalizers=pin(HERE/'normalizers.npz'),fit_seal_sha256=sha(HERE/'fit_seal.json'),elapsed_s=time.monotonic()-start,native_candidate=False))
print(json.dumps(dict(event='complete',passed=all(gates.values()),gates=gates,check_history=h,check_reset=r)),flush=True)
