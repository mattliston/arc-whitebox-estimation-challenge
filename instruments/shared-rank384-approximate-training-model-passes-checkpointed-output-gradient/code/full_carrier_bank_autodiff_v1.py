"""Full current prefix with the frozen bank's scalar-only final head.

Offline CPU autograd implementation. Uses the identical mathematical
consumer and learned parameters, not an extra deployment correction.
"""
import torch
import numpy as np
from full_carrier_autodiff_v1 import HERE, FullCard, diagonal, raw_marginals


def scalar_features(l,m,aux,fixed):
    s = aux['sigma']
    v = s*s
    one,zero = torch.ones_like(s),torch.zeros_like(s)
    vals = (one*(2*l/15-1),one*l/16,one,aux['alpha'],s.log()-s.log().mean(),
            aux['gate'],aux['pdf'],aux['gaussian_mean']/s,m/s,(m-aux['gaussian_mean'])/s,
            aux['pre_k3']/(v*s),aux['post_v']/v,aux['pre_k4']/(v*v),zero,zero,zero,
            fixed[:,16],fixed[:,17],fixed[:,18],fixed[:,19],
            aux['post_k3']/(v*s),aux['post_k4']/(v*v),
            zero,zero,zero,zero,zero,zero,zero,one*(m/s).mean())
    return torch.stack(vals,1).float()


def final_read(x,sigma,w,state,P,Q):
    x = ((x-P['centers'][15])/P['scales'][15]).clamp(-8,8)
    mask = torch.tensor([1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,1],
                        device=x.device,dtype=x.dtype)
    x = x*mask
    g = x[:,5:6]
    src = state@P['msg_w'].T+P['msg_b']
    cm = P['msg_w'].shape[0]//3
    w2,w3 = w*w,w*w*w
    m1,m2,m3 = w.T@src[:,:cm],w2.T@src[:,cm:2*cm],w3.T@src[:,2*cm:]
    e1 = w2.T@(src[:,:cm]**2)
    e2 = (w2*w2).T@(src[:,cm:2*cm]**2)
    e3 = (w3*w3).T@(src[:,2*cm:]**2)
    msgs = torch.cat((g*m1,g*m2,g*m3,e1.clamp_min(0).log1p(),e2.clamp_min(0).log1p(),e3.clamp_min(0).log1p()),dim=1)
    h = torch.cat((x,msgs.clamp(-8,8)),dim=1)
    h = torch.nn.functional.silu(h@Q['w0'].T+Q['b0'])
    h = torch.nn.functional.silu(h@Q['w2'].T+Q['b2'])
    return sigma.float()*Q['rms'][0]*(h@Q['w4'][0]+Q['b4'][0])


class BankFullCard(FullCard):
    def __init__(self,rank,device='cpu'):
        super().__init__(rank,device)
        self.Q0 = {k:torch.tensor(v,device=self.device,dtype=torch.float32)
                   for k,v in dict(np.load(HERE/'payload_fullpair_final_head_v1.npz')).items()}

    def step(self,l,state,control,P):
        if l < self.depth-1:
            return super().step(l,state,control,P)
        m,C,p,lam,star,b,alive,radial,hstate = state
        w,ws,dt = self.w[l],self.ws[l],self.dtype
        mu = ws.T@m
        pushed = C.float()@w
        v = (w.to(dt)*pushed.to(dt)).sum(0)
        metric = diagonal(C,torch.diagonal(C)*alive)
        metric_push = metric.float()@w
        bv = (w.to(dt)*metric_push.to(dt)).sum(0)
        bpre = ws.T@b
        pp = w.T@p
        k3 = (pp.to(dt)**3*lam.to(dt)).sum(1)
        wt = w.T
        k3 = k3+((wt@star)*wt*wt).sum(1).to(dt)+3*bv*bpre
        host,aux = raw_marginals(mu,v,k3,radial*v*v)
        x = scalar_features(l,host,aux,self.fixed[l])
        delta = final_read(x,aux['sigma'],w,hstate,P,self.Q0).to(dt)
        return (host+delta+control,C,p,lam,star,b,alive,radial,hstate)
