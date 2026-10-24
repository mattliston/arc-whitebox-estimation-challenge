import numpy as np
import torch
from common import *
from projection import project
from pair_base import Pair
verify();torch.set_num_threads(1);torch.manual_seed(2026092747);rng=np.random.default_rng(2026092747)
records=[]
for n in (5,9,13):
    m=rng.uniform(.5,1.5,n);m/=np.linalg.norm(m)
    A=rng.normal(size=(n,n));B=rng.normal(size=(n,n));A=(A+A.T)/2;B=(B+B.T)/2
    ii,jj=np.triu_indices(n,1);basis=np.zeros((len(ii),n,n))
    for k,(i,j) in enumerate(zip(ii,jj)):basis[k,i,j]=basis[k,j,i]=1/np.sqrt(2)
    constraints=np.einsum('kij,j->ik',basis,m)
    values=np.einsum('kij,ij->k',basis,A)
    expected=values-constraints.T@np.linalg.lstsq(constraints@constraints.T,constraints@values,rcond=None)[0]
    expected=np.einsum('k,kij->ij',expected,basis)
    at,bt,mt=[torch.from_numpy(x) for x in (A,B,m)]
    D=project(at,mt);E=project(bt,mt)
    errors=[float(torch.max(torch.abs(D-torch.from_numpy(expected)))),float(torch.max(torch.abs(D-D.T))),float(torch.max(torch.abs(torch.diagonal(D)))),float(torch.max(torch.abs(D@mt))),float(torch.max(torch.abs(project(D,mt)-D))),abs(float((D*bt).sum()-(at*E).sum()))]
    assert max(errors)<1e-10
    records.append(dict(n=n,maximum_error=max(errors)))
torch.manual_seed(2026092747);model=Pair().double();n=9
with torch.no_grad():model.output.weight.normal_(0,.1);model.output.bias.fill_(.05)
F=torch.randn(n,94,dtype=torch.float64);R=torch.randn(n,n,dtype=torch.float64);S=torch.randn_like(R);s=torch.rand(n,dtype=torch.float64)+.5;m=torch.ones(n,dtype=torch.float64)
G=torch.randn(n,n,dtype=torch.float64)
W=torch.randn(n,7,dtype=torch.float64);T=project(torch.randn(n,n,dtype=torch.float64),m);v=(W*(T@W)).sum(0)
def objective():
    D=project(model(F,R,S,G,s),m);dv=(W*(D@W)).sum(0)
    return .5*((D-T)**2).mean()/T.square().mean()+.5*((dv-v)**2).mean()/v.square().mean()+.001*D.square().mean()/T.square().mean()
pars=list(model.parameters());loss=objective();grad=torch.autograd.grad(loss,pars);directions=[torch.randn_like(p) for p in pars]
derivative=float(sum((g*d).sum() for g,d in zip(grad,directions)))
def perturb(t):
    with torch.no_grad():
        for p,d in zip(pars,directions):p.add_(t*d)
perturb(1e-6);fp=float(objective());perturb(-2e-6);fm=float(objective());perturb(1e-6)
fd=(fp-fm)/2e-6;error=abs(fd-derivative)/max(abs(derivative),1e-12);assert error<1e-5
once(HERE/'gradient_result.json',dict(passed=True,projection=records,gradient_relative=error,parameters=sum(p.numel() for p in pars),targets_accessed=False,seal_sha256=sha(HERE/'seal.json')))
print(json.dumps(dict(passed=True,projection=records,gradient_relative=error)))
