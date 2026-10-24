"""Forty fixed epochs; same-capacity control for four invariant observables."""
import json
from pathlib import Path
import time
import numpy as np
import torch

here=Path(__file__).resolve().parent
out=here/'conditional_skew_energy_probe_verdict_v1.json'
payload=here/'conditional_skew_energy_probe_models_v1.pt'
assert not out.exists() and not payload.exists()
assert json.loads((here/'conditional_skew_energy_probe_rows_v1.json').read_text())['n']==24
VALID=(37,39,42,48);TRAIN=tuple(r for r in range(35,59) if r not in VALID)
torch.manual_seed(5290723);torch.set_num_threads(2)
dev='mps' if torch.backends.mps.is_available() else 'cpu'
records={}
for rank in TRAIN+VALID:
    with np.load(here/'rows_learned_joint_graph_probe_v1'/f'row_{rank:04d}'/'data.npz') as z:
        records[rank]={k:torch.tensor(z[key],device=dev,dtype=torch.float32) for k,key in
                       (('h','h'),('sigma','sigma'),('target','residual'),('rms','layer_rms'))}
    records[rank]['feature']=torch.tensor(np.load(here/'rows_conditional_skew_energy_probe_v1'/f'rank_{rank:04d}.npy'),device=dev)
scale=torch.stack([(records[r]['feature']**2).mean((0,1)) for r in TRAIN]).mean(0).sqrt().clamp_min(1e-8)
for d in records.values():d['feature']=d['feature']/scale
normalizer=torch.stack([(records[r]['target']**2).mean() for r in TRAIN]).mean().item()

def model():
    net=torch.nn.Sequential(torch.nn.Linear(130,64),torch.nn.SiLU(),torch.nn.Linear(64,64),torch.nn.SiLU(),torch.nn.Linear(64,1)).to(dev)
    torch.nn.init.zeros_(net[-1].weight);torch.nn.init.zeros_(net[-1].bias)
    return net

models={'control':model(),'joint':model()}
models['joint'].load_state_dict(models['control'].state_dict())
opts={k:torch.optim.AdamW(v.parameters(),lr=1e-3,weight_decay=2e-5) for k,v in models.items()}
sched={k:torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=40*len(TRAIN),eta_min=1e-4) for k,o in opts.items()}

def predict(kind,d):
    f=d['feature'] if kind=='joint' else torch.zeros_like(d['feature'])
    return d['sigma']*d['rms'][:,None]*models[kind](torch.cat((d['h'],f),dim=-1))[...,0]

def evaluate(ranks):
    per={}
    with torch.no_grad():
        for r in ranks:
            d=records[r];row={'base':float((d['target']**2).mean().cpu())}
            for k in models:row[k]=float(((predict(k,d)-d['target'])**2).mean().cpu())
            per[r]=row
    return dict(per_net=per,mean={k:float(np.mean([z[k] for z in per.values()])) for k in ('base','control','joint')})

start=time.time();history=[]
print(json.dumps(dict(device=dev,train=TRAIN,valid=VALID,epochs=40,scale=scale.cpu().tolist(),initial=evaluate(VALID))),flush=True)
for epoch in range(40):
    for j in torch.randperm(len(TRAIN)).tolist():
        d=records[TRAIN[j]]
        for k in models:
            loss=((predict(k,d)-d['target'])**2).mean()/normalizer
            assert torch.isfinite(loss).item(),(epoch,k)
            opts[k].zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(models[k].parameters(),5.)
            opts[k].step();sched[k].step()
    if epoch%5==4:
        r=dict(epoch=epoch+1,train=evaluate(TRAIN),valid=evaluate(VALID),wall_s=time.time()-start)
        history.append(r);print(json.dumps(dict(epoch=epoch+1,valid=r['valid']['mean'],wall_s=r['wall_s'])),flush=True)
last=history[-1]['valid'];ratio=last['mean']['joint']/last['mean']['control'];parent=last['mean']['joint']/last['mean']['base']
improved=sum(z['joint']<z['base'] for z in last['per_net'].values())
passed=ratio<=.92 and parent<=.95 and improved>=3
torch.save(dict(models={k:{n:v.detach().cpu() for n,v in m.state_dict().items()} for k,m in models.items()},feature_scale=scale.cpu()),payload)
receipt=dict(history=history,train=TRAIN,valid=VALID,joint_vs_control=ratio,joint_vs_parent=parent,nets_improved=improved,
             information_gate_pass=passed,wall_s=time.time()-start,
             qualification='Fixed final epoch, held-out honest teacher residuals; not closed-loop or adjusted performance.')
out.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='history'}),flush=True)
