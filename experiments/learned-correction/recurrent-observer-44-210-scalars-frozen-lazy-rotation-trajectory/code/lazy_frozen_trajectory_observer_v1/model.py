"""Exact copied GNN on fixed features; no physical recurrence or target argument."""
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO']='0.1'
import numpy as np
import torch

TRUNK=('msg_w','msg_b','enc0_w','enc0_b','enc2_w','enc2_b','enc4_w','enc4_b')
HEAD=('head15','head_b15','gain15')
KEYS=TRUNK+HEAD
MASK=(1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,1)

def runtime():
    assert np.__version__=='1.26.4' and torch.__version__.split('+')[0]=='2.7.0'
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert torch.backends.mps.is_available()
    recommended=int(torch.mps.recommended_max_memory())
    fraction=min(1.,8_000_000_000/recommended)
    torch.mps.set_per_process_memory_fraction(fraction)
    return dict(torch_version=torch.__version__,numpy_version=np.__version__,mps_recommended_bytes=recommended,
                mps_allocation_limit_bytes=recommended*fraction,mps_fallback=False,threads=1)

def initialize(P,device,trainable=True,dtype=torch.float32):
    p={k:torch.tensor(P[k],dtype=dtype,device=device,requires_grad=trainable) for k in TRUNK}
    for new,old in [('head15','head'),('head_b15','head_b'),('gain15','gain')]:
        p[new]=torch.tensor(P[old][15].copy(),dtype=dtype,device=device,requires_grad=trainable)
    assert sum(v.numel() for v in p.values())==44210 and tuple(p)==KEYS
    fixed={k:torch.tensor(P[k],dtype=dtype,device=device) for k in ('centers','scales','rms')}
    fixed['mask']=torch.tensor(MASK,dtype=dtype,device=device)[None,:]
    assert bool((fixed['scales']>0).all())
    return p,fixed

def inputs(arrays,weights,device,dtype=torch.float32):
    n=weights.shape[1]
    assert weights.shape==(16,n,n) and arrays['features'].shape==(16,n,30)
    return dict(features=torch.tensor(arrays['features'],dtype=dtype,device=device),
                sigma=torch.tensor(arrays['sigma'],dtype=dtype,device=device),
                host=torch.tensor(arrays['final_host'],dtype=dtype,device=device),
                weights=torch.tensor(weights,dtype=dtype,device=device))

def forward(data,p,fixed,retain_states=False):
    """Final served vector, delta and optional states. No input is written."""
    xall=data['features'];n=xall.shape[1]
    state=torch.zeros((n,64),dtype=xall.dtype,device=xall.device)
    states=[state] if retain_states else None
    for l in range(1,16):
        x=torch.clamp((xall[l]-fixed['centers'][l])/fixed['scales'][l],-8.,8.)
        if l==15:x=x*fixed['mask']
        g=x[:,5:6]
        src=state@p['msg_w'].T+p['msg_b']
        w=data['weights'][l];w2=w*w;w3=w2*w
        s1,s2,s3=src[:,:16],src[:,16:32],src[:,32:]
        m1=w.T@s1;m2=w2.T@s2;m3=w3.T@s3
        e1=w2.T@(s1**2);e2=(w2*w2).T@(s2**2);e3=(w3*w3).T@(s3**2)
        msgs=torch.cat((g*m1,g*m2,g*m3,torch.log1p(torch.clamp(e1,min=0.)),
                        torch.log1p(torch.clamp(e2,min=0.)),torch.log1p(torch.clamp(e3,min=0.))),dim=1)
        h=torch.cat((x,torch.clamp(msgs,-8.,8.)),dim=1)
        h=h@p['enc0_w'].T+p['enc0_b'];h=h/(1+torch.exp(-h))
        h=h@p['enc2_w'].T+p['enc2_b'];h=h/(1+torch.exp(-h))
        state=h@p['enc4_w'].T+p['enc4_b']
        if retain_states:states.append(state)
    delta=(data['sigma'][15]*fixed['rms'][15])*(state@p['head15']+p['head_b15'])*(1+.25*torch.tanh(p['gain15']))
    served=torch.clamp(data['host']+delta,min=0.)
    return served,delta,states

def export(P,p):
    result={k:v.copy() for k,v in P.items()}
    for k in TRUNK:result[k]=p[k].detach().cpu().numpy().copy()
    for new,old in [('head15','head'),('head_b15','head_b'),('gain15','gain')]:
        result[old][15]=p[new].detach().cpu().numpy()
    for k in P:
        if k in TRUNK:continue
        if k in ('head','head_b','gain'):assert result[k][:15].tobytes()==P[k][:15].tobytes()
        else:assert result[k].tobytes()==P[k].tobytes()
    assert all(v.dtype==np.float32 and np.isfinite(v).all() for v in result.values())
    return result

def retire():
    import gc
    gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
