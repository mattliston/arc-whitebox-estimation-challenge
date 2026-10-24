"""One fixed fresh post14 supplier fit; no checkpoint or amplitude selection."""
import gc,math,time
import numpy as np
import torch
from common import *
from consumer import tensors,anchored,push_delta
from model import Pair,inputs,project

verify();verify('fit_seal.json')
assert read(HERE/'qualification.json')['passed']
assert read(HERE/'unit_result.json')['passed']
for phase in ('unit','qualification'):
    guard=read(HERE/f'{phase}_guard.json')
    assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
capture=read(HERE/'capture/complete.json')
assert capture['completed']
assert tuple(sorted(row['rank'] for row in capture['rows']))==tuple(sorted(TRAIN))
for row in capture['rows']:
    item=row['packet'];assert pin(item['path'])==item
assert tuple(TRAIN)==(38,40,41,43,44,46,47,49,50,51,53,54)
torch.set_num_threads(1);torch.manual_seed(2026092766)
try:torch.set_num_interop_threads(1)
except RuntimeError:pass

def load(rank):
    with np.load(packet(rank),allow_pickle=False) as z:
        return tensors({key:z[key] for key in z.files})

with np.load(HERE/'parent_parameters.npz',allow_pickle=False) as z:
    P=tensors({key:z[key] for key in z.files})
assert all(not value.requires_grad for value in P.values())
start=time.monotonic();features=[];rho=[];third=[]
for rank in TRAIN:
    a=load(rank);s=a['s14'];R=a['C']/torch.outer(s,s);S=a['K']/torch.outer(s*s,s)
    mask=~torch.eye(len(s),dtype=torch.bool)
    features.append(a['F'].double().square().mean(0))
    rho.append(R[mask].square().mean());third.append(S[mask].square().mean())
    del a,R,S;gc.collect()
norm=dict(feature_scale=torch.stack(features).mean(0).sqrt().clamp_min(1e-5).float(),
    rho_scale=torch.stack(rho).mean().sqrt().clamp_min(1e-5).float(),
    third_scale=torch.stack(third).mean().sqrt().clamp_min(1e-5).float())
with (HERE/'normalizers.npz').open('xb') as f:
    np.savez(f,**{key:value.numpy() for key,value in norm.items()})
once(HERE/'before_train_targets.json',dict(normalizers=pin(HERE/'normalizers.npz'),
    packets=[pin(packet(rank)) for rank in TRAIN],train=list(TRAIN),
    actual_screen_targets_accessed=False))
truths={};baseline={};targets=[]
for rank in TRAIN:
    path=P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz'
    with np.load(path,allow_pickle=False) as z:
        truths[rank]=torch.from_numpy(z['mc_post'][15].astype(np.float64).copy())
    a=load(rank);baseline[rank]=float((a['prediction'][-1]-truths[rank]).square().mean())
    assert baseline[rank]>0
    targets.append(pin(path));del a
once(HERE/'train_targets.json',dict(train=list(TRAIN),targets=targets,baseline_mse=baseline))

model=Pair();assert sum(value.numel() for value in model.parameters())==6177
optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
rng=np.random.default_rng(2026092766);step=0;epochs=40;updates=480
for epoch in range(epochs):
    losses=[];max_grad=0.
    for rank in rng.permutation(TRAIN):
        rank=int(rank);a=load(rank)
        lr=.00001+(.001-.00001)*.5*(1+math.cos(math.pi*step/(updates-1)))
        for group in optimizer.param_groups:group['lr']=lr
        D=project(model(*inputs(a,norm)),a['m14'])
        pred=anchored(a,P,D);v=a['v']+push_delta(a,D)
        assert torch.isfinite(pred).all() and bool(torch.all(v>1e-10))
        mse=(pred-truths[rank]).square().mean()/baseline[rank]
        loss=mse+.001*(pred-a['prediction'][-1]).square().mean()/baseline[rank]
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        assert torch.isfinite(grad)
        max_grad=max(max_grad,float(grad));optimizer.step();step+=1
        losses.append(float(mse.detach()))
        del a,D,pred,v,mse,loss;gc.collect()
    row=dict(epoch=epoch+1,updates=step,mean_online_raw_ratio=float(np.mean(losses)),
        maximum_gradient_norm=max_grad,lr=lr,elapsed_s=time.monotonic()-start)
    with (HERE/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

assert step==480
with (HERE/'model.npz').open('xb') as f:
    np.savez(f,**{key:value.detach().numpy() for key,value in model.state_dict().items()})
once(HERE/'model_frozen.json',dict(model=pin(HERE/'model.npz'),normalizers=pin(HERE/'normalizers.npz'),
    epochs=40,updates=480,seed=2026092766,train=list(TRAIN),final_weights_only=True,
    original_parameters_frozen=True,actual_screen_targets_accessed=False))
del optimizer;model.eval();predictions={};rows=[]
with torch.no_grad():
    for rank in TRAIN:
        a=load(rank);D=project(model(*inputs(a,norm)),a['m14'])
        pred=anchored(a,P,D);v=a['v']+push_delta(a,D)
        assert torch.isfinite(pred).all() and bool(torch.all(v>1e-10))
        predictions[str(rank)]=pred.numpy().copy()
        if rank==TRAIN[0]:
            with (HERE/'supplier_reference.npz').open('xb') as f:
                np.savez(f,D=D.numpy())
        mse=float((pred-truths[rank]).square().mean())
        rows.append(dict(rank=rank,parent_mse=baseline[rank],mse=mse,
            raw_ratio=mse/baseline[rank],minimum_variance=float(v.min()),
            correction_rms=float((pred-a['prediction'][-1]).square().mean().sqrt())))
        del a,D,pred,v;gc.collect()
with (HERE/'predictions.npz').open('xb') as f:np.savez(f,**predictions)
once(HERE/'fit_result.json',dict(completed=True,train=list(TRAIN),seed=2026092766,
    epochs=40,updates=480,rows=rows,train_pooled_raw_ratio=sum(r['mse'] for r in rows)/sum(r['parent_mse'] for r in rows),
    model=pin(HERE/'model.npz'),normalizers=pin(HERE/'normalizers.npz'),
    supplier_reference=dict(rank=TRAIN[0],packet=pin(packet(TRAIN[0])),reference=pin(HERE/'supplier_reference.npz')),
    predictions=pin(HERE/'predictions.npz'),targets=targets,
    elapsed_s=time.monotonic()-start,actual_screen_targets_accessed=False,
    scope='Fresh fixed post14 physical fit. TRAIN diagnostics do not gate advance; official-bank native TRAIN2 is next.'))
print(json.dumps(dict(event='fit_complete',updates=step,train_only=True)),flush=True)
