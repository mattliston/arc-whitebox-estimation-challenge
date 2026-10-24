"""Fixed final-epoch scalar-head fit on a branch's on-policy pilot rows."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--branch',required=True,choices=('control','positive'));a=ap.parse_args()
OUT=HERE/f'positive_path_pilot_final_training_{a.branch}_v1.json'
PAYLOAD=HERE/f'payload_positive_path_pilot_final_{a.branch}_v1.npz'
assert not OUT.exists() and not PAYLOAD.exists()
assert (HERE/f'positive_path_pilot_final_rows_{a.branch}_v1.json').exists()
assert 'threadpoolctl' not in __import__('sys').modules
torch.manual_seed(1753362478);torch.set_num_threads(2);assert torch.backends.mps.is_available();dev=torch.device('mps')
VALID={37,39,42,48};Q=dict(np.load(HERE/'payload_fullpair_final_head_v1.npz'));scale=float(Q['rms'][0])
rows={r:dict(np.load(HERE/f'rows_positive_path_pilot_final_{a.branch}_v1'/f'row_{r:04d}/data.npz')) for r in range(35,59)}
def batch(valid):
    d=[v for r,v in rows.items() if (r in VALID)==valid]
    arrays=[np.concatenate([v[k] for v in d]) for k in ('h','sigma','target','old_delta')]
    return tuple(torch.tensor(x,dtype=torch.float32,device=dev) for x in arrays)
tr=batch(False);va=batch(True)
model=torch.nn.Sequential(torch.nn.Linear(126,128),torch.nn.SiLU(),torch.nn.Linear(128,128),torch.nn.SiLU(),torch.nn.Linear(128,1)).to(dev)
with torch.no_grad():
    for i in (0,2,4):
        model[i].weight.copy_(torch.tensor(Q[f'w{i}'],device=dev));model[i].bias.copy_(torch.tensor(Q[f'b{i}'],device=dev))
opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=2e-5)
epochs=40;size=8192;steps=math.ceil(len(tr[0])/size)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs*steps,eta_min=5e-5)
normalizer=float(tr[2].square().mean().cpu());start=time.time();history=[]
def evaluate(data):
    x,s,t,old=data
    with torch.no_grad():
        p=s*scale*model(x)[:,0]
        return dict(mse=float(((p-t)**2).mean().cpu()),old_mse=float(((old-t)**2).mean().cpu()),host_mse=float(t.square().mean().cpu()))
initial=evaluate(va)
for epoch in range(epochs):
    order=torch.randperm(len(tr[0]),device=dev)
    for j in range(0,len(order),size):
        idx=order[j:j+size];x,s,t,_=(v[idx] for v in tr)
        loss=((s*scale*model(x)[:,0]-t)**2).mean()/normalizer
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();sched.step()
    if epoch%10==9:
        r=dict(epoch=epoch+1,train=evaluate(tr),valid=evaluate(va),wall_s=time.time()-start)
        history.append(r);print(json.dumps(dict(branch=a.branch,**r)),flush=True)
result={f'{kind}{i}':getattr(model[i],'weight' if kind=='w' else 'bias').detach().cpu().numpy() for i in (0,2,4) for kind in ('w','b')}
result['rms']=Q['rms'].copy();np.savez(PAYLOAD,**result)
record=dict(branch=a.branch,initial=initial,history=history,train=20,valid=4,epochs=epochs,wall_s=time.time()-start,
            payload_sha256=hashlib.sha256(PAYLOAD.read_bytes()).hexdigest(),
            qualification='Fixed40-epoch matched final head, original Q initialization; final epoch only. Actual NumPy closed-loop comparison still required.')
OUT.write_text(json.dumps(record,indent=2)+'\n')
