"""Fixed fresh supplier fit on measured native innovations; no final labels."""
import gc,time
import torch
from common import *
from pair_base import Pair
from projection import project as tproject

verify()
for e in read(HERE/'fit_seal.json')['sources']:assert pin(e['path'])==e
guard_ok('assemble');guard_ok('gradient')
assert read(HERE/'gradient_result.json')['passed']
torch.set_num_threads(1);torch.manual_seed(2026092747)
base,_=tables();labels={x['mini']:x['packet'] for x in read(HERE/'labels/complete.json')['rows']}
started=time.monotonic();cells=[];fs=[];rs=[];ks=[]
terminal=H2/'current_covariance_pair_supervision_v1'
terminal_rows={x['mini']:x['packet'] for x in read(terminal/'capture/complete.json')['rows']}
for mini in TRAIN:
    b=checked_npz(base[mini]['packet'],[f'{k}_{l}' for l in LAYERS for k in ('C','K','sigma','pdf','F','m')])
    wp=base[mini]['weights'];assert pin(wp['path'])==wp;W=np.load(wp['path'],mmap_mode='r')
    z=checked_npz(labels[mini]);last=checked_npz(terminal_rows[mini],['mu','v'])
    for l in LAYERS:
        a={k:torch.from_numpy(b[f'{k}_{l}']) for k in ('C','K','sigma','F','m')}
        a.update(mini=mini,layer=l,W=torch.from_numpy(W[l+1].copy()).double(),T=torch.from_numpy(unpack(z[f'J_{l}_upper'])))
        if l<14:factor=b[f'pdf_{l+1}']/(2*b[f'sigma_{l+1}'])
        else:
            s=np.sqrt(last['v']);factor=np.exp(-.5*(last['mu']/s)**2)/(math.sqrt(2*math.pi)*2*s)
        a['factor']=torch.from_numpy(factor.copy())
        s=a['sigma'];R=a['C']/torch.outer(s,s);S=a['K']/torch.outer(s*s,s)
        fs.append(a['F'].double().square().mean(0))
        rs.append((R.square().sum()-torch.diagonal(R).square().sum())/(1024*1023))
        ks.append((S.square().sum()-torch.diagonal(S).square().sum())/(1024*1023))
        cells.append(a)
    del b,z,last,W;gc.collect()
norm=dict(feature_scale=torch.stack(fs).mean(0).sqrt().clamp_min(1e-5).float(),
    rho_scale=torch.stack(rs).mean().sqrt().clamp_min(1e-5).float(),
    third_scale=torch.stack(ks).mean().sqrt().clamp_min(1e-5).float())
with (HERE/'normalizers.npz').open('xb') as f:np.savez(f,**{k:v.numpy() for k,v in norm.items()})
def response(a,D):return a['factor']*(a['W']*(D@a['W'])).sum(0)
for a in cells:
    s=a['sigma'];a['F']=(a['F']/norm['feature_scale']).clamp(-8,8)
    a['R']=(a.pop('C')/torch.outer(s,s)).float()/norm['rho_scale']
    a['S']=(a.pop('K')/torch.outer(s*s,s)).float()/norm['third_scale']
    a['response']=response(a,a['T']);a['energy']=a['T'].square().mean();a['response_energy']=a['response'].square().mean()
    assert float(a['energy'])>0 and float(a['response_energy'])>0
del fs,rs,ks;gc.collect()
model=Pair();assert sum(x.numel() for x in model.parameters())==6177
optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
rng=np.random.default_rng(2026092747);updates=40*len(cells);step=0
def prediction(a):return tproject(model(a['F'],a['R'],a['S'],a['sigma']),a['m'])
for epoch in range(40):
    losses=[];maxgrad=0.
    for index in rng.permutation(len(cells)):
        a=cells[int(index)];lr=1e-5+(.001-1e-5)*.5*(1+math.cos(math.pi*step/(updates-1)))
        for group in optimizer.param_groups:group['lr']=lr
        D=prediction(a);v=response(a,D)
        eC=(D-a['T']).square().mean()/a['energy']
        eV=(v-a['response']).square().mean()/a['response_energy']
        loss=.5*eC+.5*eV+.001*D.square().mean()/a['energy'];assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        assert torch.isfinite(gn);optimizer.step();step+=1;maxgrad=max(maxgrad,float(gn))
        losses.append((float(eC.detach()),float(eV.detach())))
        del D,v,eC,eV,loss
    row=dict(epoch=epoch+1,updates=step,online_covariance_ratio=float(np.mean([v[0] for v in losses])),
        online_response_ratio=float(np.mean([v[1] for v in losses])),maximum_gradient_norm=maxgrad,
        lr=lr,elapsed_s=time.monotonic()-started)
    with (HERE/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
assert step==1680
with (HERE/'model.npz').open('xb') as f:np.savez(f,**{k:v.detach().numpy() for k,v in model.state_dict().items()})
once(HERE/'model_frozen.json',dict(model=pin(HERE/'model.npz'),normalizers=pin(HERE/'normalizers.npz'),
    updates=step,epochs=40,check_labels_used=False,final_mean_targets_used=False,seed=2026092747,
    labels=pin(HERE/'labels/complete.json'),fresh_initialization=True))
del optimizer;model.eval();out=HERE/'predictions';out.mkdir();rows=[]
with torch.no_grad():
    for mini in TRAIN:
        packet={};metrics=[]
        for a in cells:
            if a['mini']!=mini:continue
            D=prediction(a);v=response(a,D);l=a['layer']
            packet[f'D_{l}_upper']=D.numpy()[UPPER].copy()
            metrics.append(dict(layer=l,covariance_ratio=float((D-a['T']).square().mean()/a['energy']),
                response_ratio=float((v-a['response']).square().mean()/a['response_energy']),
                mean_annihilation_max=float(torch.max(abs(D@(a['m']/torch.linalg.vector_norm(a['m'])))))))
        p=out/f'mini{mini:04d}.npz'
        with p.open('xb') as f:np.savez(f,**packet)
        rows.append(dict(mini=mini,packet=pin(p),cells=metrics))
once(HERE/'fit_result.json',dict(rows=rows,model=pin(HERE/'model.npz'),normalizers=pin(HERE/'normalizers.npz'),
    elapsed_s=time.monotonic()-started,final_mean_targets_used=False,check_labels_used=False,
    native_evaluation_required=True,selected_by_training_loss=False))
print(json.dumps(dict(event='fit_complete',updates=step,mean_covariance_ratio=float(np.mean([c['covariance_ratio'] for r in rows for c in r['cells']])),
    mean_response_ratio=float(np.mean([c['response_ratio'] for r in rows for c in r['cells']])))),flush=True)
