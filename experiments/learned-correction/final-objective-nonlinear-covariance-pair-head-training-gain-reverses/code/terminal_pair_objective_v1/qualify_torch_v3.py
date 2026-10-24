import numpy as np
import torch
from common import *
from consumer_v3 import tensors,evaluate,push_delta
from model import Pair
verify();torch.set_num_threads(1);torch.manual_seed(2026092740)
g=read(HERE/'capture_00_guard.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
with np.load(HERE/'parent_parameters.npz') as z:P=tensors({k:z[k] for k in z.files})
rows=[];output={}
def rowmask(mu,v):
    z=np.abs(mu/np.sqrt(np.maximum(v,1e-10)))
    count=min(len(z),((np.count_nonzero(z<=3.5)+31)//32)*32)
    mask=np.zeros(len(z),dtype=bool);mask[np.argsort(z,kind='stable')[:count]]=True
    return mask
for rank in TRAIN[:2]:
    path=HERE/'capture_00'/f'row_{rank:04d}.npz'
    with np.load(path) as z:a=tensors({k:z[k] for k in z.files})
    v=a['v'].clone().requires_grad_();base,x=evaluate(a,P,v)
    yp=a['prediction'][-1].numpy();gap=np.abs(base.detach().numpy()-yp)
    bound=2*np.abs(np.spacing(yp.astype(np.float32))).astype(np.float64)+1e-9
    ok=bool(np.all(gap<=bound));assert ok,(rank,float(gap.max()),float((gap/bound).max()))
    rng=np.random.default_rng(2026092740+rank);F=a['F'].double().numpy();F/=np.maximum(np.sqrt(np.mean(F*F,axis=0)),1e-5)
    s=a['s14'].numpy();R=a['C'].numpy()/np.outer(s,s);S=a['K'].numpy()/np.outer(s*s,s)
    rr=max(float(np.sqrt(np.mean(R*R))),1e-5);sr=max(float(np.sqrt(np.mean(S*S))),1e-5)
    directions=[]
    for j in range(2):
        q=F@rng.normal(0,1/np.sqrt(94),94)
        raw=np.tanh(q[:,None]+.3*q[None,:]+R/rr+(.2*S/sr if j else 0))
        D=(raw+raw.T)*np.outer(s,s)*.001;np.fill_diagonal(D,0.)
        dv=push_delta(a,torch.from_numpy(D)).detach().numpy()
        # Input-only fixed normalization to a .1% relative-variance RMS.
        D*=.001/np.sqrt(np.mean((dv/a['v'].numpy())**2));dt=torch.from_numpy(D)
        dv=push_delta(a,dt);w=torch.sin(torch.arange(len(v),dtype=torch.float64)*.173)
        scalar=(base*w).sum();grad=torch.autograd.grad(scalar,v,retain_graph=True)[0]
        slope=float((grad*dv).sum());eps=1e-3
        fp=evaluate(a,P,v+eps*dv)[0];fm=evaluate(a,P,v-eps*dv)[0]
        fd=float(((fp-fm)*w).sum()/(2*eps));rel=abs(slope-fd)/max(abs(slope),1e-10)
        # FP32 final GNN makes sub-ULP finite differences approximate.
        fdlarge=float(((evaluate(a,P,v+.5*dv)[0]-evaluate(a,P,v-.5*dv)[0])*w).sum())
        rellarge=abs(slope-fdlarge)/max(abs(slope),1e-10)
        assert rellarge<.02,(rank,j,slope,fdlarge,rellarge)
        for amplitude in (1.,.5):
            plus=evaluate(a,P,v+amplitude*dv)[0].detach().numpy()
            minus=evaluate(a,P,v-amplitude*dv)[0].detach().numpy()
            mask0=rowmask(a['mu'].numpy(),a['v'].numpy())
            maskp=rowmask(a['mu'].numpy(),(v+amplitude*dv).detach().numpy())
            maskm=rowmask(a['mu'].numpy(),(v-amplitude*dv).detach().numpy())
            stable=bool(np.array_equal(mask0,maskp) and np.array_equal(mask0,maskm))
            output[f'{rank}_{j}_{amplitude}_plus']=plus;output[f'{rank}_{j}_{amplitude}_minus']=minus
            directions.append(dict(direction=j,amplitude=amplitude,rowcut_stable=stable))
        output[f'{rank}_{j}_D']=D
        directions.append(dict(direction=j,gradient=slope,fd=fd,small_step_relative=rel,large_step_relative=rellarge))
    rows.append(dict(rank=rank,zero_max_abs=float(gap.max()),zero_max_fraction_of_bound=float((gap/bound).max()),directions=directions))
# Smooth tiny all-parameter directional derivative of the full pair supplier.
torch.manual_seed(2026092740);m=Pair().double()
with torch.no_grad():m.output.weight.normal_(0,.1);m.output.bias.fill_(.05)
n=9;F=torch.randn(n,94,dtype=torch.float64);R=torch.randn(n,n,dtype=torch.float64);S=torch.randn_like(R);s=torch.rand(n,dtype=torch.float64)+.5
params=list(m.parameters());loss=(m(F,R,S,s)*torch.arange(n*n,dtype=torch.float64).reshape(n,n)).sum()
grads=torch.autograd.grad(loss,params);dirs=[torch.randn_like(p) for p in params];d=float(sum((a*b).sum() for a,b in zip(grads,dirs)))
def perturb(step):
    with torch.no_grad():
        for p,q in zip(params,dirs):p.add_(step*q)
perturb(1e-6);p=float((m(F,R,S,s)*torch.arange(n*n,dtype=torch.float64).reshape(n,n)).sum())
perturb(-2e-6);q=float((m(F,R,S,s)*torch.arange(n*n,dtype=torch.float64).reshape(n,n)).sum());perturb(1e-6)
relative=abs(d-(p-q)/2e-6)/max(abs(d),1e-12);assert relative<1e-6
with (HERE/'qualification_directions_v3.npz').open('xb') as f:np.savez(f,**output)
once(HERE/'torch_qualification_v3.json',dict(passed=True,rows=rows,pair_parameter_count=sum(p.numel() for p in params),pair_gradient_relative=relative,directions=pin(HERE/'qualification_directions_v3.npz'),targets_accessed=False))
print(json.dumps(dict(torch_qualified=True,pair_gradient_relative=relative,rows=rows)),flush=True)
