"""Fixed FIT128 training. No CHECK or VALID labels may be loaded here."""
from pathlib import Path
import hashlib
import json
import math
import resource
import sys
import time
import numpy as np
import torch
from model import Head

torch.set_num_threads(1);torch.set_num_interop_threads(1)
HERE=Path(__file__).resolve().parent;P2=HERE.parent.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((HERE/'protocol.json').read_text());cfg=protocol['training']
capture=json.loads((HERE/'capture_complete.json').read_text())
assert capture['all_groups_retired'] and capture['swap_growth_bytes']==0
assert capture['protocol_sha256']==sha(HERE/'protocol.json')
assert capture['graph_sha256']==sha(HERE/'graph.py') and capture['capture_sha256']==sha(HERE/'capture.py')
mode=sys.argv[1];assert mode in ('graph','control')
out=HERE/f'fit_{mode}';out.mkdir()
records={r['rank']:r for r in capture['rows']}

def load(ranks,labels=False):
    assert not labels or list(ranks)==protocol['fit_ranks']
    xs=[];ss=[];ys=[];tt=[]
    for rank in ranks:
        r=records[rank];p=HERE/r['path'];assert sha(p)==r['sha256']
        with np.load(p) as z:
            xs.append(np.concatenate((z['control'],z['graph']),axis=1))
            ss.append(z['sigma'].astype(np.float32));ys.append(z['prediction'][15])
        if labels:
            assert rank not in protocol['check_ranks'] and rank not in protocol['valid_ranks_not_accessed']
            with np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz') as z:tt.append(z['mc_post'][15].astype(np.float64))
    result=(np.concatenate(xs),np.concatenate(ss),np.concatenate(ys))
    return result+(np.concatenate(tt),) if labels else result

x,sd,base,truth=load(protocol['fit_ranks'],labels=True)
mean=x.mean(0,dtype=np.float64).astype(np.float32)
scale=np.maximum(x.std(0,dtype=np.float64),cfg['feature_scale_floor']).astype(np.float32)
x=np.clip((x-mean)/scale,-cfg['feature_clip'],cfg['feature_clip'])
if mode=='control':x[:,94:]=0
X=torch.from_numpy(x);S=torch.from_numpy(sd)
E=torch.from_numpy((base-truth).astype(np.float32))
torch.manual_seed(cfg['seed']);model=Head()
with torch.no_grad():assert torch.equal(model(X[:32]),torch.zeros(32))
optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr_initial'],weight_decay=cfg['weight_decay'],foreach=False)
rng=np.random.default_rng(cfg['seed']);batch=cfg['batch'];steps=cfg['epochs']*math.ceil(len(x)/batch);step=0
started=time.monotonic();baseline=float(np.mean((base-truth)**2));history=[]
with (out/'training.jsonl').open('x') as log:
    for epoch in range(cfg['epochs']):
        order=rng.permutation(len(x));total=0.
        for b in range(0,len(x),batch):
            ind=torch.from_numpy(order[b:b+batch]);phase=step/max(steps-1,1)
            lr=cfg['lr_final']+(cfg['lr_initial']-cfg['lr_final'])*.5*(1+math.cos(math.pi*phase))
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True)
            prediction=model(X[ind])*S[ind]*protocol['model']['output_scale']
            loss=((E[ind]+prediction)**2).mean()*cfg['loss_scale']
            assert torch.isfinite(loss)
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'])
            assert torch.isfinite(grad)
            optimizer.step();total+=float(loss.detach())*len(ind)/cfg['loss_scale'];step+=1
        row=dict(epoch=epoch+1,training_trajectory_mse=total/len(x),baseline_mse=baseline,
                 lr=lr,elapsed_s=time.monotonic()-started)
        history.append(row);log.write(json.dumps(row)+'\n');log.flush()
        print(json.dumps(dict(mode=mode,**row)),flush=True)
assert step==steps
payload={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}
with (out/'weights.npz').open('xb') as f:np.savez(f,mean=mean,scale=scale,**payload)
weights_sha=sha(out/'weights.npz')
# Model is frozen before any CHECK prediction, and CHECK labels stay unopened.
model.eval();checkX,checkS,checkBase=load(protocol['check_ranks'],labels=False)
checkX=np.clip((checkX-mean)/scale,-cfg['feature_clip'],cfg['feature_clip'])
if mode=='control':checkX[:,94:]=0
def predict(a,s):
    ds=[]
    with torch.no_grad():
        for b in range(0,len(a),batch):
            xx=torch.from_numpy(a[b:b+batch]);sig=torch.from_numpy(s[b:b+batch])
            ds.append((model(xx)*sig*protocol['model']['output_scale']).numpy())
    return np.concatenate(ds).astype(np.float64)
delta=predict(checkX,checkS)
pred=(checkBase+delta).astype(np.float32).astype(np.float64).reshape(len(protocol['check_ranks']),1024)
assert np.isfinite(pred).all()
with (out/'check_predictions.npy').open('xb') as f:np.save(f,pred)
fitdelta=predict(x,sd)
fitpred=(base+fitdelta).astype(np.float32).astype(np.float64)
frozen_fit_mse=float(np.mean((fitpred-truth)**2))
result=dict(mode=mode,fit_ranks=protocol['fit_ranks'],check_ranks=protocol['check_ranks'],
            weights_sha256=weights_sha,predictions_sha256=sha(out/'check_predictions.npy'),
            capture_sha256=sha(HERE/'capture_complete.json'),protocol_sha256=sha(HERE/'protocol.json'),
            fit_script_sha256=sha(HERE/'fit.py'),model_sha256=sha(HERE/'model.py'),
            terminal_epoch=cfg['epochs'],steps=step,baseline_fit_mse=baseline,frozen_fit_mse=frozen_fit_mse,
            check_labels_accessed=False,valid_labels_accessed=False,elapsed_s=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,torch=torch.__version__,threads=torch.get_num_threads())
with (out/'result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(dict(event='fit_complete',**result)),flush=True)
