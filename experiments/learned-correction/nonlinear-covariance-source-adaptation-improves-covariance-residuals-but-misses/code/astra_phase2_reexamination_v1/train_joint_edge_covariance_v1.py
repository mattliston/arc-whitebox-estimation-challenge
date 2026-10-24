"""Fixed control versus joint-K21 nonlinear covariance-source information fit."""
import hashlib
import json
import math
from pathlib import Path
import resource
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
OUT=HERE/'joint_edge_covariance_verdict_v1.json'
assert not OUT.exists()
assert 'threadpoolctl' not in __import__('sys').modules
TRAIN=(35,100,150,200,250,300);VALID=(400,500);EPOCHS=40;SEED=5290826
torch.set_num_threads(2);assert torch.backends.mps.is_available();DEV=torch.device('mps')
data={};provenance={}
for rank in TRAIN+VALID:
    folder=HERE/'rows_joint_edge_covariance_v1'/f'row_{rank:04d}'
    meta=json.loads((folder/'result.json').read_text());assert meta['all_old_arrays_byte_exact']
    oldpath=HERE/'rows_covariance_innovation_probe_v2'/f'row_{rank:04d}'/'data.npz'
    assert hashlib.sha256(oldpath.read_bytes()).hexdigest()==meta['old_source_sha256']
    assert hashlib.sha256((folder/'data.npz').read_bytes()).hexdigest()==meta['output_sha256']
    with np.load(oldpath) as z:d={k:torch.from_numpy(z[k].copy()) for k in z.files}
    d['S']=torch.from_numpy(np.load(folder/'data.npz')['S'].copy());data[rank]=d;provenance[rank]=meta
scale=torch.sqrt(torch.stack([(data[r]['basis']**2).mean((0,1)) for r in TRAIN]).mean(0)).clamp_min(1e-8)
rho_scale=math.sqrt(float(torch.stack([(data[r]['rho']**2).mean() for r in TRAIN]).mean()))
third_scale=math.sqrt(float(torch.stack([(data[r]['S']**2).mean() for r in TRAIN]).mean()))
sigma_scale2=float(torch.stack([(data[r]['sigma']**2).mean() for r in TRAIN]).mean())
cov_norm=float(torch.stack([(data[r]['target']**2).mean() for r in TRAIN]).mean())
cov_scale=math.sqrt(cov_norm)
envelope_energy=[]
for rank,d in data.items():
    d['gate']=d['basis'][:,:,0].clone();d['pdf']=d['basis'][:,:,1].clone()
    if rank in TRAIN:
        a=d['pdf'][:,:,None]*d['gate'][:,None,:]
        envelope_energy.append((.5*(a+a.transpose(1,2))) .square().mean())
    d['basis']=d['basis']/scale[None,None,:]
    d['rho']=d['rho']/rho_scale;d['S']=d['S']/third_scale
    with torch.no_grad():
        e=d['target'].to(DEV);w=d['weight'].to(DEV);r=d['response'].to(DEV)
        d['value_target']=((w*(e@w)).sum(1)*r).cpu()
    del e,w,r
envelope_scale=math.sqrt(float(torch.stack(envelope_energy).mean()))
value_norm=float(torch.stack([(data[r]['value_target']**2).mean() for r in TRAIN]).mean())
assert min(rho_scale,third_scale,sigma_scale2,cov_norm,envelope_scale,value_norm)>0


class Pair(torch.nn.Module):
    def __init__(self,use_joint):
        super().__init__();self.use_joint=use_joint
        self.left=torch.nn.Parameter(torch.randn(14,16)*.12)
        self.right=torch.nn.Parameter(torch.randn(14,16)*.12)
        self.edge=torch.nn.Parameter(torch.randn(3,16)*.12)
        self.bias=torch.nn.Parameter(torch.zeros(16))
        self.output=torch.nn.Parameter(torch.zeros(16))

    def forward(self,d):
        f=d['basis'];R=d['rho'];S=d['S'] if self.use_joint else torch.zeros_like(d['S'])
        h=(f@self.left)[:,None,:]+(f@self.right)[None,:,:]+self.bias
        h=h+R[:,:,None]*self.edge[0]+S[:,:,None]*self.edge[1]+S.T[:,:,None]*self.edge[2]
        raw=torch.tanh(h)@self.output
        raw=raw*d['pdf'][:,None]*d['gate'][None,:]
        pred=.5*(raw+raw.T)*(d['sigma'][:,None]*d['sigma'][None,:])*(cov_scale/(sigma_scale2*envelope_scale))
        return pred


def cell(rank,j):return {k:v[j].to(DEV) for k,v in data[rank].items()}


def evaluate(model,ranks):
    result={}
    with torch.no_grad():
        for rank in ranks:
            sums=np.zeros(4)
            for j in range(7):
                d=cell(rank,j);p=model(d);v=(d['weight']*(p@d['weight'])).sum(0)*d['response']
                sums+=np.array([float(((p-d['target'])**2).mean().cpu()),float(d['target'].square().mean().cpu()),
                                float(((v-d['value_target'])**2).mean().cpu()),float(d['value_target'].square().mean().cpu())])
                del d,p,v
            sums/=7
            result[rank]=dict(cov_mse=sums[0],cov_base=sums[1],cov_ratio=sums[0]/sums[1],
                              value_mse=sums[2],value_base=sums[3],value_ratio=sums[2]/sums[3])
    return result


def numerical_certificate():
    torch.manual_seed(SEED);model=Pair(True).to(DEV)
    d={k:v[0,:16].to(DEV) for k,v in data[TRAIN[0]].items() if k in ('basis','sigma','gate','pdf')}
    d.update(rho=data[TRAIN[0]]['rho'][0,:16,:16].to(DEV),S=data[TRAIN[0]]['S'][0,:16,:16].to(DEV))
    with torch.no_grad():
        zero=model(d).cpu().numpy();assert np.array_equal(zero,np.zeros_like(zero))
        model.output.copy_(torch.linspace(-.2,.2,16,device=DEV))
        actual=model(d).cpu().numpy()
    p={k:v.detach().cpu().numpy().astype(np.float64) for k,v in model.named_parameters()}
    x={k:v.cpu().numpy().astype(np.float64) for k,v in d.items()}
    def numpy_predict(x):
        h=(x['basis']@p['left'])[:,None,:]+(x['basis']@p['right'])[None,:,:]+p['bias']
        h+=x['rho'][:,:,None]*p['edge'][0]+x['S'][:,:,None]*p['edge'][1]+x['S'].T[:,:,None]*p['edge'][2]
        raw=(np.tanh(h)@p['output'])*x['pdf'][:,None]*x['gate'][None,:]
        return .5*(raw+raw.T)*np.outer(x['sigma'],x['sigma'])*(cov_scale/(sigma_scale2*envelope_scale))
    ref=numpy_predict(x);error=float(np.linalg.norm(actual-ref)/np.linalg.norm(ref))
    perm=np.random.default_rng(SEED).permutation(16)
    xp={k:(v[np.ix_(perm,perm)] if k in ('rho','S') else v[perm]) for k,v in x.items()}
    permutation=float(np.max(np.abs(numpy_predict(xp)-ref[np.ix_(perm,perm)])))
    symmetry=float(np.max(np.abs(actual-actual.T)))
    assert error<2e-6 and permutation<1e-14 and symmetry==0,(error,permutation,symmetry)
    del model,d
    return dict(zero_byte_exact=True,numpy_mps_relative_error=error,permutation_error=permutation,symmetry_error=symmetry)


certificate=numerical_certificate();start=time.time();models={};histories={}
orders=[np.random.default_rng(SEED+epoch).permutation(len(TRAIN)*7).tolist() for epoch in range(EPOCHS)]
print(json.dumps(dict(device=str(DEV),certificate=certificate,train=TRAIN,valid=VALID,
                      epochs=EPOCHS,cov_norm=cov_norm,value_norm=value_norm)),flush=True)
for name,use_joint in (('control',False),('joint',True)):
    path=HERE/f'joint_edge_covariance_{name}_model_v1.pt';assert not path.exists()
    torch.manual_seed(SEED);model=Pair(use_joint).to(DEV)
    optimizer=torch.optim.Adam(model.parameters(),lr=.005)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS*len(TRAIN)*7,eta_min=.0005)
    initial=evaluate(model,VALID);assert all(v['cov_ratio']==1 and v['value_ratio']==1 for v in initial.values())
    history=[]
    for epoch,order in enumerate(orders):
        for idx in order:
            rank=TRAIN[idx//7];j=idx%7;d=cell(rank,j)
            p=model(d);v=(d['weight']*(p@d['weight'])).sum(0)*d['response']
            loss=((p-d['target'])**2).mean()/cov_norm+((v-d['value_target'])**2).mean()/value_norm
            penalty=sum(t.square().sum() for t in model.parameters())/sum(t.numel() for t in model.parameters())
            loss=loss+.001*penalty
            assert torch.isfinite(loss).item(),(name,epoch,rank,j)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();scheduler.step()
            del d,p,v,loss,penalty
        if epoch%5==4:
            rec=dict(epoch=epoch+1,valid=evaluate(model,VALID),wall_s=time.time()-start)
            history.append(rec);print(json.dumps(dict(model=name,**rec)),flush=True)
    final=history[-1]['valid'];histories[name]=history;models[name]=final
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},use_joint=use_joint,
                    feature_scale=scale,rho_scale=rho_scale,third_scale=third_scale,sigma_scale2=sigma_scale2,
                    cov_scale=cov_scale,envelope_scale=envelope_scale),path)
    del model,optimizer,scheduler
passed=all(models['joint'][r]['cov_ratio']<=.75 and models['joint'][r]['value_ratio']<=.75
           and models['joint'][r]['value_mse']<=.9*models['control'][r]['value_mse'] for r in VALID)
record=dict(train_nets=TRAIN,valid_nets=VALID,epochs=EPOCHS,seed=SEED,certificate=certificate,
            final_valid=models,history=histories,information_gate_pass=bool(passed),
            input_provenance=provenance,wall_s=time.time()-start,
            peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            qualification='Fixed nonlinear source pilot with joint S versus matched zero-S control. Oracle previous states; coherent8M covariance labels; held-out400/500. Final epoch only. Not a deployable, noise-corrected, recursive or adjusted score.')
OUT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k not in ('history','input_provenance')},indent=2),flush=True)
