"""One fixed five-fold comparison; no VALID40 labels or test-selected epochs."""
from pathlib import Path
import hashlib
import json
import math
import resource
import time
import numpy as np
import torch

torch.set_num_threads(1);torch.set_num_interop_threads(1)
HERE=Path(__file__).resolve().parent;P2=HERE.parent.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
seal=json.loads((HERE/'seal.json').read_text())
for name,digest in seal['sources'].items():assert sha(HERE/name)==digest
protocol=json.loads((HERE/'protocol.json').read_text());ranks=protocol['ranks']
capture=json.loads((HERE/'capture_complete.json').read_text())
assert capture['all_groups_retired'] and capture['added_swap_bytes']==0
assert capture['seal_sha256']==sha(HERE/'seal.json')
rows={r['rank']:r for r in capture['rows']};xs=[];ss=[];ys=[];truth=[];truthpins={}
for rank in ranks:
    path=HERE/rows[rank]['path'];assert sha(path)==rows[rank]['sha256']
    with np.load(path) as z:
        xs.append(np.concatenate((z['control'],z['messages']),axis=1));ss.append(z['sigma']);ys.append(z['prediction'][15])
    tp=P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz'
    truthpins[str(rank)]=sha(tp)
    with np.load(tp) as z:truth.append(z['mc_post'][15].astype(np.float64))
xs,ss,base,truth=map(np.asarray,(xs,ss,ys,truth))
assert xs.shape==(20,1024,222) and np.isfinite(xs).all()
cfg=protocol['training'];started=time.monotonic();allpred={};fithistory=[]
out=HERE/'fit';out.mkdir()

class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=torch.nn.Sequential(torch.nn.Linear(222,128),torch.nn.SiLU(),torch.nn.Linear(128,64),torch.nn.SiLU(),torch.nn.Linear(64,1))
        torch.nn.init.zeros_(self.layers[4].weight);torch.nn.init.zeros_(self.layers[4].bias)
    def forward(self,x):return self.layers(x).squeeze(-1)

for mode in ('control','messages'):
    prediction=np.empty_like(base)
    for fold in range(5):
        it=np.array([i for i in range(20) if i%5!=fold]);ic=np.array([i for i in range(20) if i%5==fold])
        x=xs[it].reshape(-1,222);sd=ss[it].reshape(-1).astype(np.float32)
        error=(base[it]-truth[it]).reshape(-1).astype(np.float32)
        mean=x.mean(axis=0,dtype=np.float64).astype(np.float32);scale=np.maximum(x.std(axis=0,dtype=np.float64),1e-6).astype(np.float32)
        x=np.clip((x-mean)/scale,-8,8)
        check=np.clip((xs[ic].reshape(-1,222)-mean)/scale,-8,8)
        if mode=='control':x[:,94:]=0.;check[:,94:]=0.
        X=torch.from_numpy(x);S=torch.from_numpy(sd);E=torch.from_numpy(error)
        torch.manual_seed(cfg['seed']);model=Head()
        assert torch.equal(model(X[:10]),torch.zeros(10))
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr_initial'],weight_decay=cfg['weight_decay'],foreach=False)
        rng=np.random.default_rng(cfg['seed']);steps=cfg['epochs']*math.ceil(len(x)/cfg['batch']);step=0;history=[]
        for epoch in range(cfg['epochs']):
            order=rng.permutation(len(x));total=0.
            for b in range(0,len(x),cfg['batch']):
                ind=torch.from_numpy(order[b:b+cfg['batch']]);phase=step/max(steps-1,1)
                lr=cfg['lr_final']+(cfg['lr_initial']-cfg['lr_final'])*.5*(1+math.cos(math.pi*phase))
                for group in optimizer.param_groups:group['lr']=lr
                optimizer.zero_grad(set_to_none=True)
                loss=((E[ind]+model(X[ind])*S[ind]*.001)**2).mean()*1e8
                assert torch.isfinite(loss);loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(grad)
                optimizer.step();total+=float(loss.detach())*len(ind)/1e8;step+=1
            history.append(total/len(x))
        assert step==steps
        payload={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}
        modelpath=out/f'{mode}_fold{fold}.npz'
        with modelpath.open('xb') as f:np.savez(f,mean=mean,scale=scale,**payload)
        # Only the terminal model is evaluated; no held-network scores enter fit.
        with torch.no_grad():
            delta=model(torch.from_numpy(check)).numpy().astype(np.float64)*ss[ic].reshape(-1)*.001
        prediction[ic]=np.maximum(base[ic]+delta.reshape(4,1024),0).astype(np.float32).astype(np.float64)
        rec=dict(mode=mode,fold=fold,fit_ranks=[ranks[i] for i in it],check_ranks=[ranks[i] for i in ic],
                 weights_sha256=sha(modelpath),training_mse_by_epoch=history)
        fithistory.append(rec)
        print(json.dumps(dict(event='fit_done',mode=mode,fold=fold,elapsed_s=time.monotonic()-started)),flush=True)
    allpred[mode]=prediction
with (out/'predictions.npz').open('xb') as f:np.savez(f,base=base,truth=truth,ranks=np.array(ranks),**allpred)
base_mse=np.mean((base-truth)**2,axis=1);scores={}
for mode,pred in allpred.items():
    mse=np.mean((pred-truth)**2,axis=1);ratio=mse/base_mse
    foldratios=[float(np.mean(mse[np.arange(20)%5==f])/np.mean(base_mse[np.arange(20)%5==f])) for f in range(5)]
    scores[mode]=dict(pooled_raw_ratio=float(mse.mean()/base_mse.mean()),network_mse=mse.tolist(),network_ratios=ratio.tolist(),
                     winning_networks=int(np.sum(ratio<1)),fold_ratios=foldratios,winning_folds=sum(x<1 for x in foldratios),
                     worst_network_raw_ratio=float(ratio.max()))
m,c=scores['messages'],scores['control'];ratio_control=m['pooled_raw_ratio']/c['pooled_raw_ratio']
gates=dict(pooled_raw=m['pooled_raw_ratio']<=.98,matched_control=ratio_control<=.985,
           folds=m['winning_folds']>=4,networks=m['winning_networks']>=12,worst=m['worst_network_raw_ratio']<=1.2)
result=dict(passed=all(gates.values()),gates=gates,scores=scores,ratio_to_control=ratio_control,ranks=ranks,
            base_mse=base_mse.tolist(),baseline_pooled_mse=float(base_mse.mean()),truth_sha256=truthpins,
            fold_models=fithistory,seal_sha256=sha(HERE/'seal.json'),predictions_sha256=sha(out/'predictions.npz'),
            capture_sha256=sha(HERE/'capture_complete.json'),valid_labels_accessed=False,elapsed_s=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,torch=torch.__version__)
with (out/'result.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(dict(event='complete',passed=result['passed'],gates=gates,scores=scores,ratio_to_control=ratio_control)),flush=True)
