"""Differentiable mean/covariance perturbation dynamics around a frozen card.

Offline torch only. No estimator or billing claims. K3/K4 and masks frozen.
"""
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

HERE = Path(__file__).resolve().parent


def diagonal(a, d):
    return a - torch.diag(torch.diagonal(a)) + torch.diag(d)


def nonlinear(mu, cov, k21, k4):
    v = torch.diagonal(cov).clamp_min(1e-10)
    s = v.sqrt(); a = mu/s
    p = torch.exp(-a*a/2)/math.sqrt(2*math.pi)
    g = torch.erfc(-a/math.sqrt(2))/2
    mg = s*p+mu*g
    r2g = (mu*mu+v)*g+mu*s*p
    r3g = mu*(mu*mu+3*v)*g+(mu*mu+2*v)*s*p
    r4g = (mu**4+6*mu*mu*v+3*v*v)*g+mu*(mu*mu+5*v)*s*p
    d = p/s; e = -a*p/v; f = (a*a-1)*p/(v*s)
    k3 = torch.diagonal(k21)
    m = mg+k3*e/6+k4*f/24
    r2 = r2g+k3*d/3+k4*e/12
    r3 = r3g+k3*g+k4*d/4
    r4 = r4g+4*k3*mg+k4*g
    var = r2-m*m
    post3 = r3-3*m*r2+2*m**3
    post4 = r4-4*m*r3+6*m*m*r2-3*m**4-3*var*var
    rho_full = cov/torch.outer(s,s)
    zero = torch.zeros_like(s)
    rho = diagonal(rho_full,zero)
    t21 = diagonal(k21/(v[:,None]*s[None,:]),zero)
    # Frozen radial input is inferred from the baseline per-neuron K4.
    # Preserve its actual baseline K22 radial shape when variances change.
    radial = (k4/(v*v)).mean()
    t22 = radial*(1+2*rho_full*rho_full)/3
    t3 = k3/s**3; t4 = k4/v**2
    t31 = t4[:,None]*rho
    A = [mg/s,g,p,-a*p,(a*a-1)*p,(-a**3+3*a)*p,(a**4-6*a*a+3)*p]
    ans = torch.zeros_like(cov); rr = torch.ones_like(cov)
    for r in range(7):
        if r:
            ans = ans + rr*torch.outer(A[r], A[r])
        if r <= 2:
            ec = (t21*torch.outer(A[r+2],A[r+1])+t21.T*torch.outer(A[r+1],A[r+2]))/2
            ec = ec+t22*torch.outer(A[r+2],A[r+2])/4
            ec = ec+(t31*torch.outer(A[r+3],A[r+1])+t31.T*torch.outer(A[r+1],A[r+3]))/6
            if r:
                ec = ec+(torch.outer(t3*A[r+3],A[r])+torch.outer(A[r],t3*A[r+3]))/6
                ec = ec+(torch.outer(t4*A[r+4],A[r])+torch.outer(A[r],t4*A[r+4]))/24
            ans = ans+rr*ec
        rr = rr*rho/(r+1)
    postcov = diagonal(ans*torch.outer(s,s),var)
    aux = dict(sigma=s,alpha=a,pdf=p,gate=g,gaussian_mean=mg,pre_v=v,pre_cov=cov,
               pre_k3=k3,pre_k4=k4,post_v=var,post_k3=post3,post_k4=post4)
    return m, postcov, aux


def features(l, m, cov, k21, aux, fixed):
    n = len(m); s = aux['sigma']; v = s*s
    rho = aux['pre_cov']/torch.outer(s,s)
    t = k21/(v[:,None]*s[None,:])
    ps = aux['post_v'].clamp_min(1e-12).sqrt()
    pr = (cov/torch.outer(ps,ps)).clamp(-1,1)
    one = torch.ones_like(s)
    vals = (one*(2*l/15-1),one*l/16,one,aux['alpha'],s.log()-s.log().mean(),
            aux['gate'],aux['pdf'],aux['gaussian_mean']/s,m/s,(m-aux['gaussian_mean'])/s,
            aux['pre_k3']/(v*s),aux['post_v']/v,aux['pre_k4']/v**2,rho.mean(1),
            (rho*rho).mean(1).clamp_min(1e-30).sqrt(),rho.abs().mean(1),
            fixed[:,16],fixed[:,17],fixed[:,18],fixed[:,19],
            aux['post_k3']/(v*s),aux['post_k4']/v**2,
            (t*t).sum(1).clamp_min(1e-30).sqrt(),(t*t).sum(0).clamp_min(1e-30).sqrt(),
            t.sum(1)/math.sqrt(n),t.sum(0)/math.sqrt(n),rho@aux['alpha']/n,
            (pr*pr).mean(1).clamp_min(1e-30).sqrt(),pr.mean(1),one*(m/s).mean())
    return torch.stack(vals,1)


def gnn(l, x, sigma, w, state, P):
    if l < 1:
        return torch.zeros_like(sigma),state*0
    x = ((x-P['centers'][l])/P['scales'][l]).clamp(-8,8)
    g = x[:,5:6]; src = state@P['msg_w'].T+P['msg_b']
    cm = P['msg_w'].shape[0]//3
    w2 = w*w; w3 = w2*w
    m1 = w.T@src[:,:cm]; m2 = w2.T@src[:,cm:2*cm]; m3 = w3.T@src[:,2*cm:]
    e1 = w2.T@(src[:,:cm]**2); e2 = (w2*w2).T@(src[:,cm:2*cm]**2); e3 = (w3*w3).T@(src[:,2*cm:]**2)
    msg = torch.cat((g*m1,g*m2,g*m3,e1.clamp_min(0).log1p(),e2.clamp_min(0).log1p(),e3.clamp_min(0).log1p()),1)
    h = torch.cat((x,msg.clamp(-8,8)),1)
    h = torch.nn.functional.silu(h@P['enc0_w'].T+P['enc0_b'])
    h = torch.nn.functional.silu(h@P['enc2_w'].T+P['enc2_b'])
    state = h@P['enc4_w'].T+P['enc4_b']
    delta = sigma*P['rms'][l]*(state@P['head'][l]+P['head_b'][l])*(1+.25*torch.tanh(P['gain'][l]))
    return delta,state


class Shadow:
    def __init__(self, rank, device='mps'):
        folder = HERE/'rows_current_transport_shadow_v1'/f'row_{rank:04d}'
        self.dev = torch.device(device)
        self.data = {p.stem: torch.tensor(np.load(p),device=self.dev,dtype=torch.float32)
                     for p in folder.glob('*.npy')}
        self.P0 = {k:torch.tensor(v,device=self.dev,dtype=torch.float32)
                   for k,v in dict(np.load(HERE/'payload_compact_feed_fullpair_v1.npz')).items()}
        self.base = []
        with torch.no_grad():
            for l in range(16):
                d = self.data
                m,c,a = nonlinear(d['mu'][l],d['pre_cov'][l],d['k21'][l],d['k4'][l])
                f = features(l,m,c,d['k21'][l],a,d['x'][l])
                delta,h = gnn(l,d['x'][l],a['sigma'],d['weights'][l],d['state_prev'][l],self.P0)
                self.base.append((m,c,f,delta,h))

    def step(self, l, dm, dc, dh, control, P):
        d = self.data; w = d['weights'][l]
        mu = d['mu'][l]+w.T@dm
        cp = d['pre_cov'][l]+w.T@dc@w
        m,c,a = nonlinear(mu,cp,d['k21'][l],d['k4'][l])
        m0,c0,f0,delta0,h0 = self.base[l]
        em = m-m0
        f = d['x'][l]+(features(l,m,c,d['k21'][l],a,d['x'][l])-f0)
        delta,h = gnn(l,f,a['sigma'],w,d['state_prev'][l]+dh,P)
        ed = delta-delta0+control
        # Stable difference of raw-second-preserving mean feedback.
        base_post = d['host'][l]+d['delta'][l]
        actual_delta = d['delta'][l]+ed
        ec = c-c0-torch.outer(base_post,ed)-torch.outer(ed,base_post)
        ec = ec-torch.outer(em,actual_delta)-torch.outer(actual_delta,em)-torch.outer(ed,ed)
        mask = d['alive'][l]
        ec = diagonal(ec*torch.outer(mask,mask),torch.diagonal(ec))
        return em+ed,ec,h-h0

    def run(self, P=None, controls=None, training=False):
        P = self.P0 if P is None else P
        if controls is None:
            controls = torch.zeros((16,1024),device=self.dev)
        dm = torch.zeros(1024,device=self.dev)
        dc = torch.zeros((1024,1024),device=self.dev)
        dh = torch.zeros((1024,64),device=self.dev)
        outputs = []
        for l in range(16):
            if training:
                def block(dm,dc,dh,control,l=l):
                    return self.step(l,dm,dc,dh,control,P)
                dm,dc,dh = checkpoint(block,dm,dc,dh,controls[l],use_reentrant=False)
            else:
                dm,dc,dh = self.step(l,dm,dc,dh,controls[l],P)
            outputs.append(dm)
        return torch.stack(outputs)
