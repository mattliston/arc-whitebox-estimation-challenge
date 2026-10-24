"""Fixed linear pair-kernel source fit with covariance and next-mean gates."""
import json
import math
from pathlib import Path
import time
import numpy as np
import torch

here=Path(__file__).resolve().parent
out=here/'covariance_innovation_probe_verdict_v1.json'
payload=here/'covariance_innovation_probe_model_v1.npz'
assert not out.exists() and not payload.exists()
assert json.loads((here/'covariance_innovation_probe_rows_v1.json').read_text())['n']==8
TRAIN=(35,100,150,200,250,300);VALID=(400,500)
torch.manual_seed(5290722);torch.set_num_threads(2)
dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
records={}
for rank in TRAIN+VALID:
    with np.load(here/'rows_covariance_innovation_probe_v1'/f'row_{rank:04d}'/'data.npz') as z:
        records[rank]={k:torch.from_numpy(z[k].copy()) for k in z.files}
feature_scale=torch.sqrt(torch.stack([(records[r]['basis']**2).mean((0,1)) for r in TRAIN]).mean(0)).clamp_min(1e-8)
sigma_scale=torch.sqrt(torch.stack([(records[r]['sigma']**2).mean() for r in TRAIN]).mean()).item()
cov_norm=torch.stack([(records[r]['target']**2).mean() for r in TRAIN]).mean().item()
cov_scale=math.sqrt(cov_norm)
for rank,d in records.items():
    d['basis']=d['basis']/feature_scale[None,None,:]*(d['sigma']/sigma_scale)[...,None]
    with torch.no_grad():
        # Offline target contraction, not part of estimator inputs.
        e=d['target'].to(dev);w=d['weight'].to(dev);response=d['response'].to(dev)
        d['value_target']=((w*(e@w)).sum(dim=1)*response).cpu()
    del e,w,response
value_norm=torch.stack([(records[r]['value_target']**2).mean() for r in TRAIN]).mean().item()
assert cov_norm>0 and value_norm>0
theta=torch.nn.Parameter(torch.zeros((3,14,14),device=dev))
opt=torch.optim.Adam([theta],lr=.02)
epochs=100
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs*len(TRAIN),eta_min=.002)

def predict(d):
    f=d['basis'];r=d['rho'];coef=(theta+theta.transpose(1,2))/2
    result=f@coef[0]@f.transpose(1,2)
    result=result+(f@coef[1]@f.transpose(1,2))*r
    result=result+(f@coef[2]@f.transpose(1,2))*(r*r)
    result=result*cov_scale
    value=(d['weight']*(result@d['weight'])).sum(dim=1)*d['response']
    return result,value

def evaluate(ranks):
    results={}
    for rank in ranks:
        d={k:v.to(dev) for k,v in records[rank].items()}
        with torch.no_grad():
            pred,value=predict(d)
            results[rank]=dict(cov_mse=float(((pred-d['target'])**2).mean().cpu()),
                cov_base=float((d['target']**2).mean().cpu()),
                value_mse=float(((value-d['value_target'])**2).mean().cpu()),
                value_base=float((d['value_target']**2).mean().cpu()))
        results[rank]['cov_ratio']=results[rank]['cov_mse']/results[rank]['cov_base']
        results[rank]['value_ratio']=results[rank]['value_mse']/results[rank]['value_base']
        del d,pred,value
    return results

start=time.time();history=[]
print(json.dumps(dict(device=str(dev),train=TRAIN,valid=VALID,cov_norm=cov_norm,value_norm=value_norm,
    basis_rms=feature_scale.tolist(),sigma_scale=sigma_scale,initial_valid=evaluate(VALID))),flush=True)
for epoch in range(epochs):
    order=torch.randperm(len(TRAIN)).tolist()
    for idx in order:
        d={k:v.to(dev) for k,v in records[TRAIN[idx]].items()}
        pred,value=predict(d)
        loss=((pred-d['target'])**2).mean()/cov_norm+((value-d['value_target'])**2).mean()/value_norm
        loss=loss+1e-3*(theta*theta).mean()
        assert torch.isfinite(loss).item(),epoch
        opt.zero_grad(set_to_none=True);loss.backward();opt.step();sched.step()
        del d,pred,value,loss
    if epoch%10==9:
        rec=dict(epoch=epoch+1,train=evaluate(TRAIN),valid=evaluate(VALID),wall_s=time.time()-start)
        history.append(rec)
        print(json.dumps(dict(epoch=epoch+1,valid=rec['valid'],wall_s=rec['wall_s'])),flush=True)
last=history[-1]['valid']
passed=all(row['cov_ratio']<=.75 and row['value_ratio']<=.75 for row in last.values())
coef=((theta+theta.transpose(1,2))/2).detach().cpu().numpy()
np.savez(payload,theta=coef,feature_scale=feature_scale.numpy(),sigma_scale=np.asarray(sigma_scale),cov_scale=np.asarray(cov_scale))
receipt=dict(train_nets=TRAIN,valid_nets=VALID,epochs=epochs,history=history,information_gate_pass=passed,
    final_valid=last,wall_s=time.time()-start,
    qualification='Oracle-state supervised covariance source pilot; two held-out public networks, stored MC covariance labels. Not a deployable or panel score; no independent covariance-replica noise correction.')
out.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='history'},indent=2),flush=True)
