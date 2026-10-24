import numpy as np
import torch
from common import *
from model import Pair,rollout
from numpy_replay import replay
verify();verify('unit_seal.json');torch.set_num_threads(1);torch.manual_seed(2026092744)
n=7;dtype=torch.float64;cells=[]
for _ in range(3):
    B=torch.randn(n,n,dtype=dtype)/3;C=B@B.T+torch.eye(n,dtype=dtype);s=torch.sqrt(torch.diagonal(C))
    cells.append(dict(F=torch.randn(n,94,dtype=dtype)/2,C=C,K=torch.randn(n,n,dtype=dtype)*.2,
        sigma=s,m=torch.linspace(.8,1.2,n,dtype=dtype),gate=torch.sigmoid(torch.randn(n,dtype=dtype)),
        pdf=torch.rand(n,dtype=dtype)*.2+.1,W=torch.randn(n,n,dtype=dtype)/np.sqrt(n)))
terminal=dict(W=torch.randn(n,n,dtype=dtype)/np.sqrt(n),sigma=torch.ones(n,dtype=dtype),
              gate=torch.rand(n,dtype=dtype),pdf=torch.rand(n,dtype=dtype)*.2+.1,parent=torch.ones(n,dtype=dtype))
norm=dict(feature_scale=torch.ones(94,dtype=dtype),rho_scale=torch.tensor(1.,dtype=dtype),third_scale=torch.tensor(1.,dtype=dtype))
model=Pair().double();assert sum(p.numel() for p in model.parameters())==6209
for mode in ('history','reset'):
    Ds,e,y=rollout(model,cells,terminal,norm,mode)
    assert all(torch.count_nonzero(D)==0 for D in Ds) and torch.count_nonzero(e)==0 and torch.equal(y,terminal['parent'])
with torch.no_grad():model.output.weight.normal_(0,.03);model.output.bias.fill_(.007)
def npdict(a):return {k:v.detach().numpy().copy() for k,v in a.items()}
nc=[npdict(a) for a in cells];nt=npdict(terminal);nn=npdict(norm)
rows=[]
directions=[torch.randn_like(p) for p in model.parameters()];length=sum(d.square().sum() for d in directions).sqrt();directions=[d/length for d in directions]
targets=[torch.randn(n,n,dtype=dtype)*.001 for _ in cells];finaltarget=torch.randn(n,dtype=dtype)*.001
def loss(mode):
    Ds,e,y=rollout(model,cells,terminal,norm,mode)
    return sum((D-t).square().mean() for D,t in zip(Ds,targets))+(e-finaltarget).square().mean()
for mode in ('history','reset'):
    Ds,e,y=rollout(model,cells,terminal,norm,mode)
    nd,ne,ny=replay(npdict(model.state_dict()),nc,nt,nn,mode)
    error=max([float(np.max(np.abs(D.detach().numpy()-d))) for D,d in zip(Ds,nd)]+[float(np.max(np.abs(e.detach().numpy()-ne))),float(np.max(np.abs(y.detach().numpy()-ny)))])
    assert error<1e-12,(mode,error)
    model.zero_grad(set_to_none=True);L=loss(mode);L.backward()
    gradient=sum((p.grad*d).sum() for p,d in zip(model.parameters(),directions)).item()
    memory_gradient=float(torch.linalg.vector_norm(model.edge.weight.grad[:,3]))
    if mode=='reset':assert memory_gradient==0
    else:assert memory_gradient>1e-15
    eps=1e-6
    with torch.no_grad():
        for p,d in zip(model.parameters(),directions):p.add_(eps*d)
        plus=float(loss(mode))
        for p,d in zip(model.parameters(),directions):p.add_(-2*eps*d)
        minus=float(loss(mode))
        for p,d in zip(model.parameters(),directions):p.add_(eps*d)
    finite=(plus-minus)/(2*eps);relative=abs(finite-gradient)/max(abs(finite),abs(gradient),1e-20)
    assert relative<1e-5,(mode,relative,finite,gradient)
    rows.append(dict(mode=mode,replay_max_abs=error,directional_gradient_relative=relative,memory_edge_gradient_norm=memory_gradient))
once(HERE/'unit_result.json',dict(passed=True,parameters=6209,rows=rows,targets_accessed=False,unit_seal_sha256=sha(HERE/'unit_seal.json')))
print(json.dumps(dict(passed=True,rows=rows)))
