"""One train-only carrier gain direction; actual held-out NumPy judgement."""
from pathlib import Path
import hashlib
import json
import resource
import time
import numpy as np
import torch
from torch.autograd import forward_ad
from inherited_distinct_gain_autodiff_v1 import InheritedDistinctCard
from inherited_distinct_gain_reference_v1 import run as numpy_run

HERE=Path(__file__).resolve().parent
OUT=HERE/'inherited_distinct_gain_pilot_v1.json'
ART=OUT.with_suffix('.npz')
PAYLOAD=HERE/'inherited_distinct_gain_control_v1.npz'
PROGRESS=HERE/'inherited_distinct_gain_progress_v1'
assert not any(p.exists() for p in (OUT,ART,PAYLOAD,PROGRESS))
assert json.loads((HERE/'inherited_distinct_gain_fidelity_v1.json').read_text())['passed']
PROGRESS.mkdir()
torch.set_num_threads(3)
TRAIN=(35,36,38,40)
VALID=(37,39,42,48)
start=time.time()
gradient=torch.zeros(15,dtype=torch.float64)
grad_rows=[]


def truth(rank):
    return np.load(HERE.parent/'public_g22_operator_features_v1/truthpost'/
                   f'truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)


def report(name,record):
    record=dict(**record,wall_s=time.time()-start,
                peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    dest=PROGRESS/(name+'.json')
    assert not dest.exists()
    dest.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


for rank in TRAIN:
    assert time.time()-start<900
    theta=torch.zeros(15,dtype=torch.float64,requires_grad=True)
    card=InheritedDistinctCard(rank,theta)
    prediction,_=card.run(training=True)
    prediction=prediction.clamp_min(0)
    target=torch.tensor(truth(rank),dtype=torch.float64)
    loss=torch.mean((prediction[-1]-target[-1])**2)/2e-8
    loss.backward()
    assert theta.grad is not None and torch.isfinite(theta.grad).all()
    assert abs(float(theta.grad[0]))<1e-12
    gradient+=theta.grad.detach()/len(TRAIN)
    rec=dict(rank=rank,baseline_raw=float(loss.detach())*2e-8,
             gradient=theta.grad.tolist())
    grad_rows.append(rec)
    report(f'gradient_{rank:04d}',rec)
    del theta,card,prediction,target,loss

direction=-gradient
direction[0]=0
maximum=float(direction.abs().max())
assert maximum>0
direction=direction/maximum
reverse=float(gradient@direction)
dots=energy=0.
tangent_rows=[]
for rank in TRAIN:
    assert time.time()-start<900
    with torch.no_grad(),forward_ad.dual_level():
        theta=forward_ad.make_dual(torch.zeros(15,dtype=torch.float64),direction)
        card=InheritedDistinctCard(rank,theta)
        prediction,_=card.run()
        base,tangent=forward_ad.unpack_dual(prediction.clamp_min(0))
        base=base.numpy().copy();tangent=tangent.numpy().copy()
    error=base-truth(rank)
    dot=float(np.mean(error[-1]*tangent[-1]))
    en=float(np.mean(tangent[-1]**2))
    assert np.isfinite(tangent).all()
    dots+=dot/len(TRAIN);energy+=en/len(TRAIN)
    rec=dict(rank=rank,error_tangent_inner_product=dot,tangent_energy=en)
    tangent_rows.append(rec)
    report(f'tangent_{rank:04d}',rec)
    del theta,card,prediction,base,tangent,error

forward=2*dots/2e-8
relative=abs(forward-reverse)/max(abs(forward),abs(reverse),1e-30)
assert relative<.01,(forward,reverse,relative)
unclipped=-dots/max(energy,1e-60)
step=float(np.clip(unclipped,0,.02))
chosen=step*direction.numpy()
assert chosen[0]==0 and np.max(np.abs(chosen))<=.02
np.savez(PAYLOAD,theta=chosen,direction=direction.numpy(),gradient=gradient.numpy())
report('frozen_train_only_control',dict(theta=chosen.tolist(),step=step,
       unconstrained_step=unclipped,reverse_derivative=reverse,
       forward_derivative=forward,derivative_relative_error=relative,
       control_sha256=hashlib.sha256(PAYLOAD.read_bytes()).hexdigest()))

# No pilot holdout label or prediction has been opened before control freeze.
records=[];saves={}
for rank in VALID:
    assert time.time()-start<900
    base=numpy_run(rank,original=True)
    new=numpy_run(rank,chosen)
    card=InheritedDistinctCard(rank,torch.tensor(chosen,dtype=torch.float64))
    with torch.no_grad():
        actual,_=card.run()
        cpu=actual.clamp_min(0).numpy().copy()
    max_error=float(np.max(np.abs(cpu-new)))
    assert max_error<2e-6 and np.isfinite(new).all(),(rank,max_error)
    target=truth(rank)
    old_mse=np.mean((base-target)**2,axis=1)
    new_mse=np.mean((new-target)**2,axis=1)
    rec=dict(rank=rank,parent_raw=float(old_mse[-1]),raw=float(new_mse[-1]),
             ratio=float(new_mse[-1]/old_mse[-1]),
             parent_layer_mse=old_mse.tolist(),layer_mse=new_mse.tolist(),
             torch_numpy_actual_max_error=max_error)
    records.append(rec)
    saves.update({f'parent_{rank}':base,f'new_{rank}':new,f'cpu_{rank}':cpu})
    report(f'heldout_{rank:04d}',rec)
    del card,actual,cpu,base,new,target

ratio=sum(r['raw'] for r in records)/sum(r['parent_raw'] for r in records)
wins=sum(r['ratio']<1 for r in records)
result=dict(train=TRAIN,valid=VALID,gradient_records=grad_rows,
            train_tangent_records=tangent_rows,records=records,
            applied_step=step,unconstrained_step=unclipped,theta=chosen.tolist(),
            reverse_derivative=reverse,forward_derivative=forward,
            derivative_relative_difference=relative,
            pooled_raw_ratio=ratio,holdout_wins=wins,
            expansion_gate=ratio<=.90 and wins>=3,
            wall_s=time.time()-start,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            control_sha256=hashlib.sha256(PAYLOAD.read_bytes()).hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            qualification='One train-only14-coordinate inherited-distinct CP gain direction and scalar, frozen before any holdout labels/predictions. Actual dynamic NumPy final MSE against existing1e9 truths, frozen P/Q and unchanged bank size. CPU action parity checked. Pilot holdouts are not the independent14-net panel; no metered bill or official score.')
np.savez(ART,**saves)
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('gradient_records','train_tangent_records','records')},indent=2),flush=True)
