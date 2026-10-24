"""Fixed40-epoch nonlinear graph information probe, same-capacity control."""
import json
import math
from pathlib import Path
import time
import numpy as np
import torch

here=Path(__file__).resolve().parent
out=here/'learned_joint_graph_probe_verdict_v1.json'
weights=here/'learned_joint_graph_probe_models_v1.pt'
assert not out.exists() and not weights.exists()
assert json.loads((here/'learned_joint_graph_probe_rows_v1.json').read_text())['n']==24
VALID={37,39,42,48}
torch.manual_seed(5290721)
torch.set_num_threads(2)
dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
records={}
started=time.time()
for rank in range(35,59):
    path=here/'rows_learned_joint_graph_probe_v1'/f'row_{rank:04d}'
    assert json.loads((path/'result.json').read_text())['parent_byte_exact']
    with np.load(path/'data.npz') as z:
        r=z['r']; t=z['t']
        operators=[]
        for a in (r,r*r,t,t.transpose(0,2,1)):
            operators.append(a/np.maximum(np.sum(np.abs(a),axis=2,keepdims=True),1e-20))
        data=dict(h=z['h'].copy(), sigma=z['sigma'].astype(np.float32),
                  target=z['residual'].astype(np.float32), rms=z['layer_rms'].astype(np.float32),
                  graph=np.stack(operators,axis=1).astype(np.float32))
        assert all(np.isfinite(a).all() for a in data.values())
    records[rank]={key:torch.from_numpy(value) for key,value in data.items()}
    del data,r,t,operators,a

class Probe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encode=torch.nn.Linear(126,32)
        self.value=torch.nn.Linear(32,16)
        self.hidden=torch.nn.Linear(126+32+64,64)
        self.output=torch.nn.Linear(64,1)
        torch.nn.init.zeros_(self.output.weight);torch.nn.init.zeros_(self.output.bias)
    def forward(self,h,graph,joint):
        z=torch.nn.functional.silu(self.encode(h))
        value=self.value(z)
        if joint:
            messages=torch.matmul(graph,value[:,None,:,:]).permute(0,2,1,3).flatten(2)
        else:
            messages=torch.cat([value]*4,dim=-1)
        hidden=torch.nn.functional.silu(self.hidden(torch.cat([h,z,messages],dim=-1)))
        return self.output(hidden)[...,0]

control=Probe().to(dev);graph=Probe().to(dev);graph.load_state_dict(control.state_dict())
models={'control':control,'graph':graph}
train=[r for r in records if r not in VALID]
normalizer=float(np.mean([float((records[r]['target']**2).mean()) for r in train]))
epochs=40
opts={k:torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=2e-5) for k,m in models.items()}
schedules={k:torch.optim.lr_scheduler.CosineAnnealingLR(o,T_max=epochs*len(train),eta_min=1e-4) for k,o in opts.items()}

def prediction(model,data,joint):
    return data['sigma']*data['rms'][:,None]*model(data['h'],data['graph'],joint)

def evaluate(ranks):
    per_net={}
    for rank in ranks:
        d={key:value.to(dev) for key,value in records[rank].items()}
        with torch.no_grad():
            result={'base':float(torch.mean(d['target']**2).cpu())}
            for kind,model in models.items():
                err=prediction(model,d,kind=='graph')-d['target']
                result[kind]=float(torch.mean(err*err).cpu())
        per_net[rank]=result
        del d
    mean={kind:float(np.mean([row[kind] for row in per_net.values()])) for kind in ('base','control','graph')}
    return dict(mean=mean,per_net=per_net)

print(json.dumps(dict(device=str(dev),train_nets=train,valid_nets=sorted(VALID),epochs=epochs,
    normalizer=normalizer,initial=evaluate(sorted(VALID)),load_wall_s=time.time()-started)),flush=True)
history=[]
for epoch in range(epochs):
    order=torch.randperm(len(train)).tolist()
    for idx in order:
        d={key:value.to(dev) for key,value in records[train[idx]].items()}
        for kind,model in models.items():
            pred=prediction(model,d,kind=='graph')
            loss=torch.mean(((pred-d['target'])/math.sqrt(normalizer))**2)
            assert torch.isfinite(loss).item(),(epoch,kind)
            opts[kind].zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            opts[kind].step();schedules[kind].step()
        del d,pred,loss
    if epoch%5==4:
        rec=dict(epoch=epoch+1,train=evaluate(train),valid=evaluate(sorted(VALID)),wall_s=time.time()-started)
        history.append(rec)
        print(json.dumps({k:v for k,v in rec.items() if k not in ('train','valid')} | dict(train=rec['train']['mean'],valid=rec['valid']['mean'])),flush=True)
last=history[-1]['valid']
ratio=last['mean']['graph']/last['mean']['control']
parent_ratio=last['mean']['graph']/last['mean']['base']
improved=sum(row['graph']<row['base'] for row in last['per_net'].values())
passed=ratio<=.92 and parent_ratio<=.95 and improved>=3
torch.save({k:{name:value.detach().cpu() for name,value in m.state_dict().items()} for k,m in models.items()},weights)
receipt=dict(train_nets=train,valid_nets=sorted(VALID),epochs=epochs,history=history,
    graph_vs_control_ratio=ratio,graph_vs_parent_ratio=parent_ratio,heldout_nets_improved=improved,
    information_gate_pass=passed,wall_s=time.time()-started,
    qualification='Fixed final epoch; parent teacher-forced states and mean residuals. This is not closed-loop performance or an official/panel score.')
out.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='history'},indent=2),flush=True)
