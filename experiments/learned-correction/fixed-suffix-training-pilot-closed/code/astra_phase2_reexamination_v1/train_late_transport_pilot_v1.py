"""One five-epoch suffix-only end-to-end pilot; fixed final epoch only."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from current_transport_shadow_v3 import Shadow

here=Path(__file__).resolve().parent
out=here/'late_transport_training_pilot_v1.json'
payload=here/'payload_late_transport_pilot_v1.npz'
progress=here/'late_transport_pilot_progress_v1'
assert not out.exists() and not payload.exists() and not progress.exists()
assert json.loads((here/'current_transport_late_corner_fidelity_v1.json').read_text())['fidelity_pass']
assert json.loads((here/'late_transport_pilot_rows_v1.json').read_text())['n']==24
progress.mkdir()
VALID=(37,39,42,48);TRAIN=tuple(r for r in range(35,59) if r not in VALID)
torch.manual_seed(710298);torch.set_num_threads(2)
dev='mps' if torch.backends.mps.is_available() else 'cpu'
Pnp=dict(np.load(here/'payload_compact_feed_fullpair_v1.npz'))
frozen={'centers','scales','rms','first'}
parameters=torch.nn.ParameterDict({k:torch.nn.Parameter(torch.tensor(v,device=dev,dtype=torch.float32)) for k,v in Pnp.items() if k not in frozen})
labels={r:torch.tensor(np.load(here.parent/'public_g22_operator_features_v1/truthpost'/f'truthpost_rank{r:04d}.npz')['mc_post'],device=dev,dtype=torch.float32) for r in TRAIN+VALID}
base_cpu={r:np.load(here/'rows_current_transport_shadow_v2'/f'row_{r:04d}'/'baseline.npy') for r in TRAIN+VALID}
normalizer=float(np.mean([np.mean((np.maximum(base_cpu[r][-1],0)-labels[r][-1].cpu().numpy())**2) for r in TRAIN]))
opt=torch.optim.Adam(parameters.parameters(),lr=5e-5)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=5*len(TRAIN),eta_min=5e-6)
weight=torch.tensor([1.,1.,1.,1.,1.,1.,4.],device=dev)

def run(rank,training):
    s=Shadow(rank,dev)
    P=dict(s.P0);P.update(dict(parameters.items()))
    dm=torch.zeros(1024,device=dev);dc=torch.zeros((1024,1024),device=dev);dh=torch.zeros((1024,64),device=dev)
    zero=torch.zeros(1024,device=dev);parts=[]
    for l in range(9,16):
        def block(dm,dc,dh,l=l):
            return s.step(l,dm,dc,dh,zero,P)
        if training:dm,dc,dh=checkpoint(block,dm,dc,dh,use_reentrant=False)
        else:dm,dc,dh=block(dm,dc,dh)
        parts.append(dm)
    pred=(s.data['baseline'][9:]+torch.stack(parts)).clamp_min(0)
    mse=((pred-labels[rank][9:])**2).mean(1)
    loss=(mse*weight).sum()/weight.sum()/normalizer
    # s is captured by non-reentrant checkpoint closures until backward.
    return loss,mse,s

def evaluate():
    rows=[]
    with torch.no_grad():
        for rank in VALID:
            loss,mse,s=run(rank,False)
            old=float(np.mean((np.maximum(base_cpu[rank][-1],0)-labels[rank][-1].cpu().numpy())**2))
            value=float(mse[-1].cpu())
            rows.append(dict(rank=rank,shadow_mse=value,parent_mse=old,ratio=value/old))
            del loss,mse,s
    return rows

start=time.time();history=[]
initial=evaluate()
print(json.dumps(dict(device=dev,train=TRAIN,valid=VALID,epochs=5,normalizer=normalizer,initial=initial)),flush=True)
for epoch in range(5):
    losses=[]
    for rank in np.random.default_rng(710298+epoch).permutation(TRAIN):
        assert time.time()-start<900,'15-minute pilot limit'
        loss,mse,s=run(int(rank),True)
        assert torch.isfinite(loss).item(),(epoch,int(rank))
        opt.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(parameters.parameters(),5.)
        assert torch.isfinite(norm).item(),(epoch,int(rank),'gradient')
        opt.step();sched.step()
        losses.append(float(loss.detach().cpu()))
        del loss,mse,s
    rec=dict(epoch=epoch+1,train_normalized_loss=float(np.mean(losses)),valid=evaluate(),wall_s=time.time()-start)
    history.append(rec)
    (progress/f'epoch_{epoch+1:02d}.json').write_text(json.dumps(rec,indent=2)+'\n')
    print(json.dumps(rec),flush=True)
Pout={k:v.copy() for k,v in Pnp.items()}
Pout.update({k:v.detach().cpu().numpy() for k,v in parameters.items()})
np.savez(payload,**Pout)
receipt=dict(train=TRAIN,valid=VALID,epochs=5,initial=initial,history=history,wall_s=time.time()-start,
             qualification='Suffix shadow training only; no real-carrier validation or deployable gain established.')
out.write_text(json.dumps(receipt,indent=2)+'\n')
print('LATE_TRANSPORT_PILOT_TRAINED',flush=True)
