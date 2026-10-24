"""One train-only encoder direction and scalar; no optimizer/step sweep."""
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from torch.autograd import forward_ad
from full_carrier_bank_autodiff_v1 import BankFullCard

HERE=Path(__file__).resolve().parent
OUT=HERE/'encoder_end_to_end_direction_v1.json'
ART=OUT.with_suffix('.npz')
PAYLOAD=HERE/'payload_encoder_end_to_end_direction_v1.npz'
PROGRESS=HERE/'encoder_end_to_end_direction_progress_v1'
assert not any(p.exists() for p in (OUT,ART,PAYLOAD,PROGRESS))
assert json.loads((HERE/'full_carrier_bank_baseline_v1.json').read_text())['passed']
PROGRESS.mkdir()
TRAIN=(35,36,38,40)
VALID=(37,39,42,48)
KEYS=('msg_w','msg_b','enc0_w','enc0_b','enc2_w','enc2_b','enc4_w','enc4_b')
torch.set_num_threads(3)
Pnp=dict(np.load(HERE/'payload_compact_feed_fullpair_v1.npz'))
gradient={k:torch.zeros_like(torch.tensor(Pnp[k])) for k in KEYS}
gradient_rows=[]
start=time.time()


def truth(rank):
    return torch.tensor(np.load(HERE.parent/'public_g22_operator_features_v1/truthpost'/
                               f'truthpost_rank{rank:04d}.npz')['mc_post'],dtype=torch.float64)


def progress(name,record):
    record=dict(**record,wall_s=time.time()-start,
                peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    path=PROGRESS/(name+'.json')
    assert not path.exists()
    path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


for rank in TRAIN:
    assert time.time()-start<900
    card=BankFullCard(rank,'cpu')
    P=dict(card.P0)
    for k in KEYS:P[k]=P[k].clone().requires_grad_(True)
    prediction,_=card.run(P=P,training=True)
    prediction=prediction.clamp_min(0)
    target=truth(rank)
    loss=torch.mean((prediction[-1]-target[-1])**2)/2e-8
    loss.backward()
    row=dict(rank=rank,raw_final_mse=float(loss.detach())*2e-8,gradient_norms={})
    for k in KEYS:
        assert P[k].grad is not None and torch.isfinite(P[k].grad).all(),(rank,k)
        gradient[k]+=P[k].grad.detach()/len(TRAIN)
        row['gradient_norms'][k]=float(torch.linalg.vector_norm(P[k].grad))
    gradient_rows.append(row)
    progress(f'gradient_{rank:04d}',row)
    del prediction,target,loss,card,P

direction={k:-gradient[k]*max(float(np.mean(Pnp[k].astype(np.float64)**2)),1e-8) for k in KEYS}
norms={k:float(torch.linalg.vector_norm(direction[k]))/max(float(np.linalg.norm(Pnp[k])),1e-12) for k in KEYS}
maximum=max(norms.values())
assert maximum>0
direction={k:v/maximum for k,v in direction.items()}
directional_reverse=sum(float(torch.sum(gradient[k]*direction[k])) for k in KEYS)
np.savez(PROGRESS/'direction.npz',**{k:v.numpy() for k,v in direction.items()},
         **{'gradient_'+k:v.numpy() for k,v in gradient.items()})
records=[];saves={};dots=energy=0.
for rank in TRAIN+VALID:
    assert time.time()-start<900
    card=BankFullCard(rank,'cpu')
    with torch.no_grad(),forward_ad.dual_level():
        P=dict(card.P0)
        for k in KEYS:P[k]=forward_ad.make_dual(P[k],direction[k])
        output,_=card.run(P=P)
        output=output.clamp_min(0)
        baseline,tangent=forward_ad.unpack_dual(output)
        baseline=baseline.detach().numpy().copy()
        tangent=tangent.detach().numpy().copy()
    label=truth(rank).numpy()
    error=baseline-label
    assert np.isfinite(tangent).all()
    dot=float(np.mean(error[-1]*tangent[-1]))
    e=float(np.mean(tangent[-1]**2))
    if rank in TRAIN:dots+=dot/len(TRAIN);energy+=e/len(TRAIN)
    row=dict(rank=rank,training=rank in TRAIN,baseline_raw=float(np.mean(error[-1]**2)),
             error_tangent_inner_product=dot,tangent_energy=e)
    records.append(row)
    saves[f'baseline_{rank}']=baseline
    saves[f'tangent_{rank}']=tangent
    saves[f'error_{rank}']=error
    progress(f'tangent_{rank:04d}',row)
    del card,P,output,baseline,tangent,label,error

forward_derivative=2*dots/2e-8
relative=abs(forward_derivative-directional_reverse)/max(abs(forward_derivative),abs(directional_reverse),1e-30)
assert relative<.01,(forward_derivative,directional_reverse,relative)
optimal=-dots/max(energy,1e-60)
step=float(np.clip(optimal,0,.10))
new={k:v.copy() for k,v in Pnp.items()}
for k in KEYS:new[k]=(torch.tensor(new[k])+step*direction[k]).numpy()
for row in records:
    rank=row['rank'];r=saves[f'error_{rank}'][-1]+step*saves[f'tangent_{rank}'][-1]
    row['tangent_raw']=float(np.mean(r*r));row['tangent_ratio']=row['tangent_raw']/row['baseline_raw']
np.savez(PAYLOAD,**new)
np.savez(ART,**saves)
record=dict(train=TRAIN,valid=VALID,keys=KEYS,parameters=sum(Pnp[k].size for k in KEYS),
            group_relative_direction_norm={k:norms[k]/maximum for k in KEYS},
            train_least_squares_step=optimal,applied_step=step,
            reverse_directional_derivative=directional_reverse,
            forward_directional_derivative=forward_derivative,derivative_relative_difference=relative,
            gradients=gradient_rows,records=records,wall_s=time.time()-start,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            payload_sha256=hashlib.sha256(PAYLOAD.read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            qualification='One train-only gradient direction in the existing encoder/message parameters, fixed10% group-relative cap and train-only scalar fit. Heads/physics/finalQ frozen. Exact local CPU branch tangent; dynamic NumPy actual finite-step validation required. Not a raw-score improvement or candidate.')
OUT.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='gradients'},indent=2),flush=True)
