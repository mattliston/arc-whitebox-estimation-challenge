"""Current C64 mean GNN only; no physical closure or legacy auxiliary heads."""
import numpy as np
import torch
from common import TRAINABLE,FROZEN

MASK=[1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,1]

class GNN(torch.nn.Module):
    def __init__(self,payload):
        super().__init__();assert set(payload)==set(TRAINABLE+FROZEN)
        assert payload['centers'].shape==(16,30) and payload['head'].shape==(16,64)
        assert payload['msg_w'].shape==(48,64) and payload['enc0_w'].shape==(128,126)
        assert payload['first'].shape==(1,) and float(payload['first'][0])==1.
        self.p=torch.nn.ParameterDict({k:torch.nn.Parameter(torch.from_numpy(np.array(v,dtype=np.float32,copy=True)),requires_grad=k in TRAINABLE) for k,v in payload.items()})
        self.register_buffer('mask',torch.tensor(MASK,dtype=torch.float32))

    def forward(self,x,sigma,W,return_states=False):
        p=self.p;n=x.shape[1];state=torch.zeros(n,64,dtype=x.dtype,device=x.device);deltas=[];states=[]
        assert x.shape==(16,1024,30) and sigma.shape==(16,1024) and W.shape==(16,1024,1024)
        for l in range(16):
            if l==0:
                deltas.append(torch.zeros_like(sigma[l]));states.append(state);continue
            xl=torch.clamp((x[l]-p['centers'][l])/p['scales'][l],-8,8)
            if l==15:xl=xl*self.mask
            g=xl[:,5:6];src=state@p['msg_w'].T+p['msg_b'];w=W[l];w2=w*w;w3=w2*w
            parts=src.split(16,dim=1)
            messages=[g*(w.T@parts[0]),g*(w2.T@parts[1]),g*(w3.T@parts[2])]
            messages += [torch.log1p(torch.clamp(w2.T@(parts[0]*parts[0]),min=0)),torch.log1p(torch.clamp((w2*w2).T@(parts[1]*parts[1]),min=0)),torch.log1p(torch.clamp((w3*w3).T@(parts[2]*parts[2]),min=0))]
            h=torch.cat([xl,torch.clamp(torch.cat(messages,dim=1),-8,8)],dim=1)
            # Native silu is x/(1+exp(-x)); preserve that expression.
            h=h@p['enc0_w'].T+p['enc0_b'];h=h/(1+torch.exp(-h))
            h=h@p['enc2_w'].T+p['enc2_b'];h=h/(1+torch.exp(-h))
            state=h@p['enc4_w'].T+p['enc4_b']
            delta=((sigma[l]*p['rms'][l])*(state@p['head'][l]+p['head_b'][l]))*(1+.25*torch.tanh(p['gain'][l:l+1]))
            deltas.append(delta);states.append(state)
        result=torch.stack(deltas)
        return (result,torch.stack(states)) if return_states else result

def runtime():
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    assert torch.get_num_threads()==1
    torch.manual_seed(40)

def payload(path):
    with np.load(path,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}

def tensors(arrays,weights,gaussian=False):
    sigma=arrays['sigma_gaussian' if gaussian else 'sigma']
    return (torch.from_numpy(np.array(arrays['x'],dtype=np.float32,copy=True)),torch.from_numpy(np.array(sigma,dtype=np.float32,copy=True)),torch.from_numpy(np.array(weights,dtype=np.float32,copy=True)))

def loss(pred,target,model):
    per=((pred[1:]-target[1:])**2).mean(1)/(model.p['rms'][1:]**2)
    return (per*torch.linspace(.5,1.5,15,device=pred.device)).mean()+per[-1]
