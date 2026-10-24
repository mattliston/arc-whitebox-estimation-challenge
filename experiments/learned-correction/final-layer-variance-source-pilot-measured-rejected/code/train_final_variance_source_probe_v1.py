"""Fixed matched final-only40-epoch probe; actual NumPy holdout judgement."""
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
from compact_final_head_v1 import final_delta

HERE=Path(__file__).resolve().parent
OUT=HERE/'final_variance_source_probe_verdict_v1.json'
PROGRESS=HERE/'final_variance_source_training_progress_v1'
assert not OUT.exists() and not PROGRESS.exists()
assert json.loads((HERE/'final_variance_source_rows_pilot_v1.json').read_text())['passed']
assert 'threadpoolctl' not in sys.modules
assert torch.backends.mps.is_available()
torch.set_num_threads(2);dev=torch.device('mps')
VALID=(37,39,42,48);TRAIN=tuple(r for r in range(35,59) if r not in VALID)
rows={r:dict(np.load(HERE/'rows_final_variance_source_probe_v1'/f'row_{r:04d}/data.npz')) for r in range(35,59)}
for r in rows:
    rec=json.loads((HERE/'rows_final_variance_source_probe_v1'/f'row_{r:04d}/result.json').read_text())
    assert rec['passed'] and all(rec['byte_parity'].values())
Q=dict(np.load(HERE/'payload_fullpair_final_head_v1.npz'))
train_sources=np.concatenate([rows[r]['sources'] for r in TRAIN])
center=train_sources.mean(axis=0);std=np.maximum(train_sources.std(axis=0),1e-6)
assert center.shape==std.shape==(14,)
for r,d in rows.items():
    d['extra']=np.clip((d['sources']-center)/std,-8,8).astype(np.float32)

# All input-only rows were frozen and qualified before this label access.
truth={r:np.load(HERE.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{r:04d}.npz')['mc_post'][-1].astype(np.float64) for r in rows}
PROGRESS.mkdir();start=time.time();scale=float(Q['rms'][0]);histories={};payloads={};cpu_predictions={};mps_predictions={}


def make_batch(ranks,enabled):
    inputs=[];sigmas=[];targets=[]
    for r in ranks:
        d=rows[r]
        extra=d['extra'] if enabled else np.zeros_like(d['extra'])
        inputs.append(np.concatenate([d['h'],extra],axis=1))
        sigmas.append(d['sigma']);targets.append(truth[r]-d['host'])
    arrays=(np.concatenate(inputs),np.concatenate(sigmas),np.concatenate(targets))
    return tuple(torch.tensor(x,dtype=torch.float32,device=dev) for x in arrays)


for name,enabled in (('control',False),('sources',True)):
    path=HERE/f'payload_final_variance_source_{name}_pilot_v1.npz'
    assert not path.exists()
    torch.manual_seed(930062)
    tr=make_batch(TRAIN,enabled);va=make_batch(VALID,enabled)
    model=torch.nn.Sequential(torch.nn.Linear(140,128),torch.nn.SiLU(),
          torch.nn.Linear(128,128),torch.nn.SiLU(),torch.nn.Linear(128,1)).to(dev)
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].weight[:,:126].copy_(torch.tensor(Q['w0'],device=dev))
        model[0].bias.copy_(torch.tensor(Q['b0'],device=dev))
        for i in (2,4):
            model[i].weight.copy_(torch.tensor(Q[f'w{i}'],device=dev))
            model[i].bias.copy_(torch.tensor(Q[f'b{i}'],device=dev))
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=2e-5)
    epochs=40;batch_size=8192;steps=math.ceil(len(tr[0])/batch_size)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs*steps,eta_min=5e-5)
    normalizer=float(tr[2].square().mean().cpu());history=[]
    def evaluate(data):
        x,s,t=data
        with torch.no_grad():
            pred=s*scale*model(x)[:,0]
            return float((pred-t).square().mean().cpu())
    initial=evaluate(va)
    for epoch in range(epochs):
        order=torch.randperm(len(tr[0]),device=dev)
        for j in range(0,len(order),batch_size):
            idx=order[j:j+batch_size];x,s,t=(v[idx] for v in tr)
            loss=(s*scale*model(x)[:,0]-t).square().mean()/normalizer
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            opt.step();sched.step()
        if epoch%10==9:
            rec=dict(arm=name,epoch=epoch+1,train_mse=evaluate(tr),valid_mse=evaluate(va),wall_s=time.time()-start)
            history.append(rec);print(json.dumps(rec),flush=True)
    export={f'{kind}{i}':getattr(model[i],'weight' if kind=='w' else 'bias').detach().cpu().numpy().copy() for i in (0,2,4) for kind in ('w','b')}
    export.update(rms=Q['rms'].copy(),source_center=center,source_std=std,source_enabled=np.array([enabled]))
    np.savez(path,**export)
    payloads[name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    histories[name]=dict(initial_valid_mse=initial,epochs=history)
    (PROGRESS/(name+'.json')).write_text(json.dumps(dict(**histories[name],payload=payloads[name]),indent=2)+'\n')
    # Separate ordinary NumPy head evaluation against the actual frozen
    # prefix. No selection state changes when only finalQ is replaced.
    for r in VALID:
        d=rows[r];extra=d['extra'] if enabled else np.zeros_like(d['extra'])
        x=np.concatenate([d['h'],extra],axis=1)
        delta=final_delta(x,d['sigma'],export)
        with torch.no_grad():
            td=torch.tensor(d['sigma'],dtype=torch.float32,device=dev)*scale*model(torch.tensor(x,device=dev))[:,0]
            td=td.cpu().numpy().copy()
        discrepancy=float(np.max(np.abs(td-delta)))
        assert discrepancy<=2e-9,(name,r,discrepancy)
        cpu_predictions[name,r]=np.maximum(d['host']+delta,0)
        mps_predictions[name,r]=discrepancy
    del model,opt,sched,tr,va
    torch.mps.empty_cache()

records=[];saves={}
for r in VALID:
    parent=rows[r]['prediction'][-1]
    values={name:float(np.mean((cpu_predictions[name,r]-truth[r])**2)) for name in ('control','sources')}
    baseline=float(np.mean((parent-truth[r])**2))
    rec=dict(rank=r,parent_raw=baseline,control_raw=values['control'],sources_raw=values['sources'],
             sources_to_parent=values['sources']/baseline,sources_to_control=values['sources']/values['control'],
             control_numpy_mps_max=mps_predictions['control',r],sources_numpy_mps_max=mps_predictions['sources',r])
    records.append(rec)
    saves[f'parent_{r}']=parent
    for name in ('control','sources'):saves[f'{name}_{r}']=cpu_predictions[name,r]
parent_ratio=sum(r['sources_raw'] for r in records)/sum(r['parent_raw'] for r in records)
control_ratio=sum(r['sources_raw'] for r in records)/sum(r['control_raw'] for r in records)
wins=sum(r['sources_to_parent']<1 for r in records)
result=dict(train=TRAIN,valid=VALID,records=records,histories=histories,payloads=payloads,
            pooled_sources_to_parent=parent_ratio,pooled_sources_to_control=control_ratio,parent_wins=wins,
            expansion_gate=parent_ratio<=.90 and control_ratio<=.92 and wins>=3,
            wall_s=time.time()-start,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            qualification='One fixed40-epoch matched final-only Q pilot on20/4 current on-policy nets, original1e9 truths. Inputs and P prefix byte-exact before targets,14 variance-source channels versus same-size zero-input control. Independent NumPy final head with original dynamic prefix, clipped outputs. No fit expansion,14-net adjusted score, bill, candidate or official result.')
np.savez(OUT.with_suffix('.npz'),**saves)
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('histories','payloads')},indent=2),flush=True)
