from common import *
import numpy as np
import torch
from model import Supplier,numpy_pair,numpy_diag
torch.set_num_threads(1);torch.set_num_interop_threads(1)
verify();torch.manual_seed(20260927392);model=Supplier().double()
rng=np.random.default_rng(392);a=torch.tensor(rng.normal(size=(9,140)));b=torch.tensor(rng.normal(size=(9,140)));e=torch.tensor(rng.normal(size=(9,4)))
assert torch.equal(model.pair(a,b,e),torch.zeros(9,dtype=torch.float64))
assert torch.equal(model.diag(a),torch.zeros(9,dtype=torch.float64))
with torch.no_grad():
    model.output.weight.copy_(torch.tensor(rng.normal(size=(1,32))*.1))
    model.diagonal[2].weight.copy_(torch.tensor(rng.normal(size=(1,32))*.1))
p={k:v.detach().numpy().copy() for k,v in model.state_dict().items()}
pair=model.pair(a,b,e);diag=model.diag(a)
err=max(float(np.max(np.abs(pair.detach().numpy()-numpy_pair(a.numpy(),b.numpy(),e.numpy(),p)))),
        float(np.max(np.abs(diag.detach().numpy()-numpy_diag(a.numpy(),p)))))
assert err<1e-12
assert torch.allclose(pair,model.pair(b,a,e[:,[1,0,3,2]]),rtol=0,atol=1e-14)
directions=[torch.tensor(rng.normal(size=tuple(x.shape))) for x in model.parameters()]
norm=torch.sqrt(sum(torch.sum(d*d) for d in directions));directions=[d/norm for d in directions]
weight=torch.tensor(rng.normal(size=9))
def loss():return torch.sum(model.pair(a,b,e)*weight)+torch.sum(model.diag(a)*weight.flip(0))
model.zero_grad();value=loss();value.backward()
analytic=sum(float(torch.sum(x.grad*d)) for x,d in zip(model.parameters(),directions))
saved=[x.detach().clone() for x in model.parameters()];eps=1e-5;values=[]
for sign in (-1,1):
    with torch.no_grad():
        for x,original,d in zip(model.parameters(),saved,directions):x.copy_(original+sign*eps*d)
    values.append(float(loss().detach()))
with torch.no_grad():
    for x,original in zip(model.parameters(),saved):x.copy_(original)
numeric=(values[1]-values[0])/(2*eps)
rel=abs(analytic-numeric)/max(abs(analytic),abs(numeric),1e-12);assert rel<3e-6
F=torch.tensor(rng.normal(size=(7,140)));R=torch.tensor(rng.normal(size=(7,7)));S=torch.tensor(rng.normal(size=(7,7)))
with torch.no_grad():D=model.matrix(F,R,S,.1,.3)
assert torch.equal(D,D.T)
i,j=np.triu_indices(7,1);ee=torch.stack((R[i,j],R[j,i],S[i,j],S[j,i]),axis=1)
with torch.no_grad():reference=model.pair(F[i],F[j],ee)*.1
matrix_error=float(torch.max(torch.abs(D[i,j]-reference)));assert matrix_error<1e-12
result=dict(passed=True,numpy_max_abs=err,gradient_relative_error=rel,matrix_pair_max_abs=matrix_error,
            parameter_count=sum(x.numel() for x in model.parameters()),zero_initial_output_exact=True,symmetry_exact=True,
            seal_sha256=sha(HERE/'seal.json'),targets_accessed=False)
once(HERE/'unit_result.json',result);print(json.dumps(result),flush=True)
