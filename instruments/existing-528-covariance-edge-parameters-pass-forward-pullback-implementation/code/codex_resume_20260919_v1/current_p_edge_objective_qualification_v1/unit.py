import math
from common import *
verify()
import numpy as np
import torch
import model_edge as model
torch.set_num_threads(1);torch.set_num_interop_threads(1)
p,a,c,e=model.torch_payload(torch,np,True)
assert sum(e[k].numel() for k in EKEYS)==528
assert all(not v.requires_grad for v in p.values())
assert {k for k,v in e.items() if v.requires_grad}==set(EKEYS)
en={k:v.detach().numpy().copy() for k,v in e.items()}
n=7;q=np.arange(n,dtype=np.float64);alpha=np.linspace(-1.2,1.3,n);s=.8+.04*q
pdf=np.exp(-alpha**2/2)/math.sqrt(2*math.pi)
g=np.array([.5*(1+math.erf(v/math.sqrt(2))) for v in alpha])
sk=.02*np.sin(q+.3);ku=.03*np.cos(q+.2)
m=s*(pdf+alpha*g)+.005*np.sin(q)
R=.9*np.eye(n)+.1*np.ones((n,n));C=R*np.outer(s,s)
S=.012*np.sin(q[:,None]+2*q[None,:]+.2)
K=S*(s[:,None]**2*s[None,:])
aux=dict(alpha=alpha,pdf=pdf,gate=g,sigma=s,pre_k3=sk*s**3,pre_k4=ku*s**4,pre_cov=C)
ten=lambda x:torch.from_numpy(np.asarray(x).copy())
env=model.source_environment(model.TorchNP(torch),None,e,torch)
out=env['_edge_source'](8,ten(m),ten(K),{k:ten(v) for k,v in aux.items()})
cot=np.cos(q[:,None]*.7+q[None,:]*.3+.2)
loss=(out*ten(cot)).sum();loss.backward()

# Independent FP64 assembly and analytic pullback, after the documented
# FP32 feature/scaling boundary. This does not differentiate a full estimator.
ms=m/s
F=np.stack((g,pdf,alpha*pdf,(alpha*alpha-1)*pdf,ms,alpha*g,sk*g,sk*pdf,sk*alpha*pdf,sk*ms,
    ku*g,ku*pdf,ku*alpha*pdf,ku*ms),axis=1).astype(np.float32)/en['feature_scale']
R32=(C/np.outer(s,s)).astype(np.float32)/en['rho_scale']
S32=(K/(s[:,None]**2*s[None,:])).astype(np.float32)/en['third_scale']
f=F.astype(np.float64);r=R32.astype(np.float64);z=S32.astype(np.float64)
weights={k:v.astype(np.float64) for k,v in en.items()}
h=(f@weights['left'])[:,None,:]+(f@weights['right'])[None,:,:]+weights['bias']
h=h+r[:,:,None]*weights['edge'][0]+z[:,:,None]*weights['edge'][1]+z.T[:,:,None]*weights['edge'][2]
t=np.tanh(h);sf=s.astype(np.float32).astype(np.float64);pf=pdf.astype(np.float32).astype(np.float64);gf=g.astype(np.float32).astype(np.float64)
scale=np.outer(sf,sf)*weights['cov_factor'];gate=pf[:,None]*gf[None,:]
raw=(t@weights['output'])*gate
expected=.5*(raw+raw.T)*scale
forward=float(np.linalg.norm(out.detach().numpy()-expected)/np.linalg.norm(expected))
assert forward<=2e-6,forward
pull=.5*(cot+cot.T)*scale*gate
dh=pull[:,:,None]*weights['output']*(1-t*t)
grad=dict(left=f.T@dh.sum(1),right=f.T@dh.sum(0),
    edge=np.stack([(dh*v[:,:,None]).sum((0,1)) for v in (r,z,z.T)]),
    bias=dh.sum((0,1)),output=(pull[:,:,None]*t).sum((0,1)))
errors={k:float(np.linalg.norm(e[k].grad.numpy()-grad[k])/np.linalg.norm(grad[k])) for k in EKEYS}
assert max(errors.values())<=2e-5,errors
write(HERE/'unit_result.json',dict(passed=True,forward_relative_error=forward,
    analytic_pullback_relative_errors=errors,edge_trainable_values=528,parent_p_frozen=True,
    independent_analytic_local_gradient=True,native_derivative_certified=False,targets_decoded=False))
print(read(HERE/'unit_result.json'),flush=True)
