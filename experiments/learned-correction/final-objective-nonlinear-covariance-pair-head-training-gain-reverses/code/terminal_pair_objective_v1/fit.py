import gc,math,time
import numpy as np
import torch
from common import *
from consumer_v4 import tensors,evaluate,anchored,push_delta
from model import Pair,inputs
verify();torch.set_num_threads(1);torch.manual_seed(2026092740)
for phase in ('qualification_v4','capture_exact_01'):
    g=read(HERE/f'{phase}_guard.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
assert read(HERE/'qualification_v4.json')['passed']
for x in read(HERE/'fit_seal.json')['sources']:assert pin(x['path'])==x
def packet(rank):return HERE/('capture_exact_00' if rank in TRAIN[:2] else 'capture_exact_01')/f'row_{rank:04d}.npz'
def load(rank):
    with np.load(packet(rank),allow_pickle=False) as z:return tensors({k:z[k] for k in z.files})
with np.load(HERE/'parent_parameters.npz') as z:P=tensors({k:z[k] for k in z.files})
start=time.monotonic();norms=[];rho=[];third=[]
for rank in TRAIN:
    a=load(rank);s=a['s14'];R=a['C']/torch.outer(s,s);S=a['K']/torch.outer(s*s,s)
    n=len(s);mask=~torch.eye(n,dtype=torch.bool)
    norms.append(a['F'].double().square().mean(dim=0));rho.append(R[mask].square().mean());third.append(S[mask].square().mean())
    del a,R,S;gc.collect()
norm=dict(feature_scale=torch.stack(norms).mean(dim=0).sqrt().clamp_min(1e-5).float(),rho_scale=torch.stack(rho).mean().sqrt().clamp_min(1e-5).float(),third_scale=torch.stack(third).mean().sqrt().clamp_min(1e-5).float())
with (HERE/'normalizers.npz').open('xb') as f:np.savez(f,**{k:v.numpy() for k,v in norm.items()})
once(HERE/'before_train_targets.json',dict(normalizers=pin(HERE/'normalizers.npz'),packets=[pin(packet(r)) for r in TRAIN+CHECK],check_targets_accessed=False))
truths={};baseline={};target_receipts=[]
for rank in TRAIN:
    path=P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz'
    with np.load(path) as z:truths[rank]=torch.from_numpy(z['mc_post'][15].astype(np.float64).copy())
    a=load(rank);baseline[rank]=float((a['prediction'][-1]-truths[rank]).square().mean());del a
    target_receipts.append(pin(path))
once(HERE/'train_targets.json',dict(ranks=TRAIN,targets=target_receipts,baseline_mse=baseline))
model=Pair();assert sum(p.numel() for p in model.parameters())==6177
optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
rng=np.random.default_rng(2026092740);step=0;epochs=40;updates=epochs*len(TRAIN)
for epoch in range(epochs):
    losses=[];max_grad=0.
    for rank in rng.permutation(TRAIN):
        rank=int(rank);a=load(rank)
        lr=.00001+(.001-.00001)*.5*(1+math.cos(math.pi*step/(updates-1)))
        for group in optimizer.param_groups:group['lr']=lr
        D=model(*inputs(a,norm));pred=anchored(a,P,D)
        assert torch.isfinite(pred).all() and bool(torch.all(a['v']+push_delta(a,D)>1e-10))
        mse=(pred-truths[rank]).square().mean()/baseline[rank]
        loss=mse+.001*(pred-a['prediction'][-1]).square().mean()/baseline[rank]
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.);assert torch.isfinite(grad)
        max_grad=max(max_grad,float(grad));optimizer.step();step+=1;losses.append(float(mse.detach()))
        del a,D,pred,mse,loss;gc.collect()
    row=dict(epoch=epoch+1,updates=step,mean_online_raw_ratio=float(np.mean(losses)),maximum_gradient_norm=max_grad,lr=lr,elapsed_s=time.monotonic()-start)
    with (HERE/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
with (HERE/'model.npz').open('xb') as f:np.savez(f,**{k:v.detach().numpy() for k,v in model.state_dict().items()})
once(HERE/'model_frozen.json',dict(model=pin(HERE/'model.npz'),normalizers=pin(HERE/'normalizers.npz'),updates=step,epochs=epochs,terminal_only=True,check_targets_accessed=False))
del optimizer;model.eval();predictions={};dr=[]
with torch.no_grad():
    for rank in TRAIN+CHECK:
        a=load(rank);D=model(*inputs(a,norm));pred=anchored(a,P,D);v=a['v']+push_delta(a,D)
        predictions[str(rank)]=pred.numpy().copy()
        dr.append(dict(rank=rank,correction_rms=float((pred-a['prediction'][-1]).square().mean().sqrt()),relative_variance_rms=float(((v-a['v'])/a['v']).square().mean().sqrt()),minimum_variance=float(v.min())))
        del a,D,pred,v;gc.collect()
with (HERE/'predictions.npz').open('xb') as f:np.savez(f,**predictions)
once(HERE/'predictions_frozen.json',dict(predictions=pin(HERE/'predictions.npz'),model=pin(HERE/'model.npz'),diagnostics=dr,check_targets_accessed=False))
rows=[]
for rank in TRAIN+CHECK:
    if rank not in truths:
        path=P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz'
        with np.load(path) as z:truths[rank]=torch.from_numpy(z['mc_post'][15].astype(np.float64).copy())
        target_receipts.append(pin(path))
    a=load(rank);truth=truths[rank].numpy();base=float(np.mean((a['prediction'][-1].numpy()-truth)**2));mse=float(np.mean((predictions[str(rank)]-truth)**2))
    rows.append(dict(rank=rank,split='TRAIN' if rank in TRAIN else 'CHECK',parent_mse=base,mse=mse,raw_ratio=mse/base));del a
held=[r for r in rows if r['split']=='CHECK'];pooled=sum(r['mse'] for r in held)/sum(r['parent_mse'] for r in held);wins=sum(r['raw_ratio']<1 for r in held);worst=max(r['raw_ratio'] for r in held)
gates=dict(pooled=pooled<=.975,wins=wins>=6,worst=worst<=1.05)
once(HERE/'fit_result.json',dict(passed=all(gates.values()),gates=gates,rows=rows,check_pooled_raw_ratio=pooled,check_wins=wins,check_worst_raw_ratio=worst,targets=target_receipts,model=pin(HERE/'model.npz'),normalizers=pin(HERE/'normalizers.npz'),elapsed_s=time.monotonic()-start,scope='Fixed current-S31 final-objective fit; offline approximate rounded kernel, native transfer still required.'))
print(json.dumps(dict(event='fit_complete',passed=all(gates.values()),check_pooled_raw_ratio=pooled,check_wins=wins,check_worst_raw_ratio=worst)),flush=True)
