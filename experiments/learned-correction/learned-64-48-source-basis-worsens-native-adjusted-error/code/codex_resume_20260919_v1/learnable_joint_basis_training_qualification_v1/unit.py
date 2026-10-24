"""Independent dense-core reference for translated carrier values/derivatives."""
import gc,itertools,resource,time
from common import *
verify();assert os.getpriority(os.PRIO_PROCESS,0)>=10
assert all(os.environ.get(k)=='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
import numpy as np
import torch
import carrier_translation
from model_basis import TorchNP,tensor_sha
assert np.__version__=='1.26.4' and torch.__version__=='2.7.0'
torch.set_num_threads(1);torch.set_num_interop_threads(1)
env=carrier_translation.environment(torch)
pairs=list(itertools.combinations_with_replacement(range(3),2))
index=lambda a,b:pairs.index(tuple(sorted((a,b))))
k3=np.array([[[env['K3'][index(a,b)][c] for c in range(3)] for b in range(3)] for a in range(3)],np.float64)
k4=np.array([[[[env['K4'][index(a,b)][index(c,d)] for d in range(3)] for c in range(3)] for b in range(3)] for a in range(3)],np.float64)
keys=('wt','hidden','variance','theta','phi','basis','left','g','d','head','head_b')
def fixture(n,nonzero):
    i=np.arange(n)[:,None];j=np.arange(n)[None,:]
    wt=(np.eye(n)/2+((3*i+5*j)%7-3)/64).astype(np.float32)
    left=(np.eye(n)/2+((7*i+3*j)%11-5)/128).astype(np.float32)
    hidden=(((np.arange(n)[:,None]*11+np.arange(64)[None,:]*7)%31-15)/32).astype(np.float32)
    basis=(np.eye(64,48)+(((7*np.arange(64)[:,None]+11*np.arange(48)[None,:])%13)-6)/512).astype(np.float32)
    return dict(wt=wt,hidden=hidden,variance=(1+np.arange(n)/16).astype(np.float32),
        theta=np.array([((-1 if i%2 else 1)/4 if nonzero else 0) for i in range(16)],np.float32),
        phi=np.array([((-1 if i%3 else 1)/8 if nonzero else 0) for i in range(16)],np.float32),basis=basis,left=left,
        g=(.5+np.arange(n)/32).astype(np.float32),d=(.125+np.arange(n)/128).astype(np.float32),
        head=np.array([(((i*7)%9-4)/4096 if nonzero else 0) for i in range(64)],np.float32),head_b=np.array(1/4096 if nonzero else 0,np.float32))
def reference(x):
    n=x['wt'].shape[0]
    def source(hidden,variance):
        f=np.sqrt(variance)[:,None]*np.tanh((x['wt']@hidden)@x['basis'])/4
        a=f[:,:16];b=f[:,16:32]*(.05*np.tanh(x['theta']));c=f[:,32:48]*(.025*np.tanh(x['phi']))
        return np.stack((a,b,c),axis=2),np.zeros((n,16,3,3)),2*b@b.T+6*c@c.T
    z,u,cov=source(x['hidden'],x['variance'])
    z=np.einsum('ij,jsa->isa',x['left'],z);u=np.einsum('ij,jsab->isab',x['left'],u)
    z2,u2,cov2=source(x['left']@x['hidden'],x['variance']+np.diag(cov))
    z=np.concatenate((z,z2),axis=1);u=np.concatenate((u,u2),axis=1)
    def planes(z,u):
        third=np.einsum('abc,isa,isb,jsc->ij',k3,z,z,z,optimize=True)
        third+=np.einsum('abcd,isab,isc,jsd->ij',k4,u,z,z,optimize=True)
        third+=.5*np.einsum('abcd,isa,isb,jscd->ij',k4,z,z,u,optimize=True)
        q31=np.einsum('abcd,isa,isb,isc,jsd->ij',k4,z,z,z,z,optimize=True)
        q22=np.einsum('abcd,isa,isb,jsc,jsd->ij',k4,z,z,z,z,optimize=True)
        return third,q31,q22
    pre=planes(z,u);g,d=x['g'],x['d']
    post21=g[:,None]**2*g[None,:]*pre[0]+(g*d)[:,None]*g[None,:]*pre[1]+.5*g[:,None]**2*d[None,:]*pre[2]
    post22=g[:,None]**2*g[None,:]**2*pre[2]
    u=g[:,None,None,None]*u+d[:,None,None,None]*np.einsum('isa,isb->isab',z,z)
    z=g[:,None,None]*z
    z=np.einsum('ij,jsa->isa',x['left'],z);u=np.einsum('ij,jsab->isab',x['left'],u)
    terminal=planes(z,u);final3=np.diag(terminal[0]);final4=np.diag(terminal[2])
    c0=np.diag(x['variance']);c1=c0+cov;v0=np.diag(c0);v1=np.diag(c1);s1=np.sqrt(v1)
    parent4=.01*v0*v0;q4=np.diag(pre[2]);den=(v1*s1)[:,None]*s1[None,:]
    explicit=(parent4[:,None]*c0/v0[:,None]+pre[1])/den
    automatic=((parent4+q4)/(v1*v1))[:,None]*(c1/(s1[:,None]*s1[None,:]))
    offset=explicit-automatic;np.fill_diagonal(offset,0)
    radial=np.asarray(.001-3*np.sum(post22)/(np.trace(c1)**2+2*np.sum(c1*c1)))
    target,inherit,trace,cp=.1*pre[0],.2*pre[0],.3*pre[0],.4*pre[0]
    innovation=np.diag(target)-np.diag(inherit)-np.diag(trace)-np.diag(post21)
    rest=target-cp-trace-post21
    head=x['hidden']@x['head']+x['head_b']
    return [cov,cov2,*pre,post21,post22,final3,final4,offset,radial,innovation,rest,head]
def translated(x,dtype):
    api=TorchNP(torch)
    if dtype==torch.float64:api.float32=torch.float64
    mm=lambda a,b:a@b;cf=env;n=x['wt'].shape[0]
    state,cov=cf['causal_source'](api,mm,x['wt'],x['hidden'],x['variance'],x['theta'],x['phi'],x['basis'])
    state=cf['affine'](api,mm,state,x['left'])
    new,cov2=cf['causal_source'](api,mm,x['wt'],x['left']@x['hidden'],x['variance']+torch.diagonal(cov),x['theta'],x['phi'],x['basis'])
    state=cf['append'](api,state,new);pre=cf['planes'](api,mm,state)
    post=cf['post_planes'](api,pre,x['g'],x['d'])
    state=cf['gate'](api,state,x['g'],x['d']);state=cf['affine'](api,mm,state,x['left'])
    final3,final4=cf['marginals'](api,state)
    c0=torch.diag(x['variance']);c1=c0+cov
    offset=cf['normalized31_offset'](api,c0,.01*x['variance']**2,c1,pre[1],torch.diagonal(pre[2]))
    radial=cf['radial_residual'](api,.001,post[1],c1)
    innovation,rest=cf['third_residual'](api,.1*pre[0],.2*pre[0],.3*pre[0],.4*pre[0],post[0])
    head=x['hidden']@x['head']+x['head_b']
    return [cov,cov2,*pre,*post,final3,final4,offset,radial,innovation,rest,head]
def np_objective(outputs):
    total=0.
    for i,value in enumerate(outputs):
        probe=np.cos(np.arange(value.size)*.137+(i+1)*.3).reshape(value.shape)
        total+=(i+1)/128*np.mean(value*probe)
    return total
def torch_objective(outputs):
    total=0.
    for i,value in enumerate(outputs):
        probe=torch.cos(torch.arange(value.numel(),dtype=torch.float64)*.137+(i+1)*.3).reshape(value.shape)
        total=total+(i+1)/128*torch.mean(value.to(torch.float64)*probe)
    return total
records=[];started=time.monotonic()
for n in (5,9):
    for nonzero in (False,True):
        inputs=fixture(n,nonzero);assert tuple(inputs)==keys
        reference_inputs={k:v.astype(np.float64) for k,v in inputs.items()};expected=reference(reference_inputs)
        gradients={};forward={}
        for dtype in (torch.float64,torch.float32):
            x={k:torch.tensor(v,dtype=dtype,requires_grad=True) for k,v in inputs.items()}
            before={k:tensor_sha(v) for k,v in x.items()};out=translated(x,dtype)
            errors=[float(np.max(np.abs(a.detach().numpy().astype(np.float64)-b))/max(1e-12,float(np.max(np.abs(b))))) for a,b in zip(out,expected)]
            assert len(out)==len(expected)==14 and max(errors)<=3e-5,(n,nonzero,str(dtype),errors)
            torch_objective(out).backward()
            assert all(v.grad is not None and bool(torch.isfinite(v.grad).all()) for v in x.values())
            assert {k:tensor_sha(v) for k,v in x.items()}==before
            gradients[str(dtype)]={k:v.grad.detach().numpy().astype(np.float64) for k,v in x.items()};forward[str(dtype)]=errors
            if not nonzero:
                for k in ('wt','hidden','variance','basis','left','g','d'):
                    assert float(np.max(np.abs(gradients[str(dtype)][k])))<=(1e-10 if dtype==torch.float64 else 2e-7),(n,k)
                for k in ('theta','phi','head','head_b'):assert float(np.linalg.norm(gradients[str(dtype)][k]))>0,(n,k)
            del x,out;gc.collect()
        checks=[];comparisons=[]
        for key in keys:
            grad=gradients['torch.float64'][key]
            direction=np.cos(np.arange(inputs[key].size)*.37+.4).reshape(inputs[key].shape)
            direction/=max(1.,float(np.linalg.norm(direction)))
            plus={k:v.copy() for k,v in reference_inputs.items()};minus={k:v.copy() for k,v in reference_inputs.items()}
            plus[key]+=1e-5*direction;minus[key]-=1e-5*direction
            central=(np_objective(reference(plus))-np_objective(reference(minus)))/(2e-5)
            analytic=float(np.sum(grad*direction));difference=abs(central-analytic)
            assert difference<=3e-5*abs(analytic)+3e-10,(n,nonzero,key,central,analytic)
            difference32=float(np.linalg.norm(gradients['torch.float32'][key]-grad));norm=float(np.linalg.norm(grad))
            assert difference32<=3e-4*norm+1e-8,(n,nonzero,key,difference32,norm)
            checks.append(dict(key=key,central=central,analytic=analytic,absolute_error=difference))
            comparisons.append(dict(key=key,gradient32_norm_error=difference32,gradient64_norm=norm))
        records.append(dict(width=n,nonzero=nonzero,forward_errors=forward,directional_checks=checks,gradient_precision_checks=comparisons,inputs_preserved=True))
verify();write(HERE/'unit_result.json',dict(passed=True,cases=4,precisions=2,directional_checks=44,records=records,
    independent_dense_core_reference=True,native_derivative_certified=False,full_network_calls=0,targets_decoded=False,
    elapsed_s=time.monotonic()-started,peak_self_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,cases=4,derivative_checks=44),flush=True)
