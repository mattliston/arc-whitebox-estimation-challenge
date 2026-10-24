"""Offline differentiable current full-pair P0 card, including the CP bank.

CPU: float64 moments, float32 carrier products and learned head. MPS: all
float32, requiring separate derivative-fidelity validation. No targets.
"""
import math
from pathlib import Path
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from current_transport_shadow_v3 import diagonal, features as node_features, gnn

HERE = Path(__file__).resolve().parent


def select(score, floor=.01):
    n = len(score)
    count = min(n, ((int((score.detach() >= floor).sum().item())+31)//32)*32)
    ix = torch.argsort(-score.detach(), stable=True)[:count]
    mask = torch.zeros(n, dtype=score.dtype, device=score.device)
    mask = mask.scatter(0, ix, torch.ones(count, dtype=score.dtype, device=score.device))
    return ix, mask


def raw_marginals(mu, v, k3, k4):
    v = v.clamp_min(1e-10)
    s = v.sqrt()
    a = mu/s
    pdf = torch.exp(-a*a/2)/math.sqrt(2*math.pi)
    g = ((1+torch.erf(a/math.sqrt(2)))/2 if mu.device.type == 'mps'
         else torch.erfc(-a/math.sqrt(2))/2)
    mg = s*pdf+mu*g
    r2g = (mu*mu+v)*g+mu*s*pdf
    r3g = mu*(mu*mu+3*v)*g+(mu*mu+2*v)*s*pdf
    r4g = (mu**4+6*mu*mu*v+3*v*v)*g+mu*(mu*mu+5*v)*s*pdf
    d = pdf/s
    e = -a*pdf/v
    f = (a*a-1)*pdf/(v*s)
    m = mg+k3*e/6+k4*f/24
    r2 = r2g+k3*d/3+k4*e/12
    r3 = r3g+k3*g+k4*d/4
    r4 = r4g+4*k3*mg+k4*g
    var = r2-m*m
    p3 = r3-3*m*r2+2*m**3
    p4 = r4-4*m*r3+6*m*m*r2-3*m**4-3*var*var
    aux = dict(sigma=s, alpha=a, pdf=pdf, gate=g, gaussian_mean=mg,
               pre_v=v, pre_k3=k3, pre_k4=k4, post_v=var,
               post_k3=p3, post_k4=p4)
    return m, aux


def nonlinear(mu, cov, k21, radial):
    n = len(mu)
    v = torch.diagonal(cov)
    k3 = torch.diagonal(k21)
    k4 = radial*v*v
    m, aux = raw_marginals(mu, v, k3, k4)
    s, a, pdf, g, mg = (aux[k] for k in ('sigma','alpha','pdf','gate','gaussian_mean'))
    zero = torch.zeros_like(s)
    rho_full = cov/torch.outer(s, s)
    rho = diagonal(rho_full, zero)
    t21 = diagonal(k21/(v[:, None]*s[None, :]), zero)
    k22 = radial*(torch.outer(v, v)+2*cov*cov)/3
    t22 = k22/torch.outer(v, v)
    t3, t4 = k3/s**3, k4/v**2
    t31 = t4[:, None]*rho
    A = [mg/s, g, pdf, -a*pdf, (a*a-1)*pdf,
         (-a**3+3*a)*pdf, (a**4-6*a*a+3)*pdf]
    B = [(a*a+1)*g+a*pdf, 2*mg/s, 2*g]
    B += [2*A[r-1] for r in range(3, 7)]

    def pair(F, G):
        ans = torch.zeros_like(cov)
        rr = torch.ones_like(cov)
        for r in range(7):
            if r:
                ans = ans+rr*torch.outer(F[r], G[r])
            if r <= 2:
                ec = (t21*torch.outer(F[r+2], G[r+1])+t21.T*torch.outer(F[r+1], G[r+2]))/2
                ec = ec+t22*torch.outer(F[r+2], G[r+2])/4
                ec = ec+(t31*torch.outer(F[r+3], G[r+1])+t31.T*torch.outer(F[r+1], G[r+3]))/6
                if r:
                    ec = ec+(torch.outer(t3*F[r+3], G[r])+torch.outer(F[r], t3*G[r+3]))/6
                    ec = ec+(torch.outer(t4*F[r+4], G[r])+torch.outer(F[r], t4*G[r+4]))/24
                ans = ans+rr*ec
            rr = rr*rho/(r+1)
        return ans

    c = pair(A, A)*torch.outer(s, s)
    p21 = pair(B, A)*torch.outer(v, s)
    p22 = pair(B, B)*torch.outer(v, v)
    tpost = p21-2*m[:, None]*c
    qpost = p22-2*m[:, None]*p21.T-2*m[None, :]*p21+4*torch.outer(m, m)*c-2*c*c
    qpost = diagonal(qpost, aux['post_k4'])
    c = diagonal(c, aux['post_v'])
    tpost = diagonal(tpost, aux['post_k3'])
    radial_new = 3*qpost.sum()/(torch.trace(c)**2+2*(c*c).sum())
    aux['pre_cov'] = cov
    return m, c, tpost, radial_new, aux


class FullCard:
    def __init__(self, rank, device='cpu', scalar_dtype=None):
        folder = HERE/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'
        self.device = torch.device(device)
        self.dtype = scalar_dtype or (torch.float32 if device == 'mps' else torch.float64)
        wn = np.load(folder/'weights.npy')
        self.w = torch.tensor(wn, device=self.device, dtype=torch.float32)
        self.ws = self.w.to(self.dtype)
        self.n = wn.shape[1]
        self.depth = len(wn)
        # These four columns depend ONLY on the unchanged weight matrix.
        fixed = np.zeros((self.depth, self.n, 30), np.float32)
        w2 = wn*wn
        w3, w4 = w2*wn, w2*w2
        fixed[:, :, 16] = wn.sum(1)*.7071067811865475
        fixed[:, :, 17] = (w2-.001953125).sum(1)*11.313708498984761
        fixed[:, :, 18] = w3.sum(1)*93.47798314754834
        fixed[:, :, 19] = (w4-1.1444091796875e-5).sum(1)*836.0924988699916
        self.fixed = torch.tensor(fixed, device=self.device, dtype=self.dtype)
        self.P0 = {k:torch.tensor(v, device=self.device, dtype=torch.float32)
                   for k, v in dict(np.load(HERE/'payload_compact_feed_fullpair_v1.npz')).items()}

    def initial(self):
        n, dev, dt = self.n, self.device, self.dtype
        return (torch.zeros(n, device=dev, dtype=dt), torch.eye(n, device=dev, dtype=dt),
                torch.zeros((n, 0), device=dev), torch.zeros(0, device=dev),
                torch.zeros((n, n), device=dev), torch.zeros(n, device=dev, dtype=dt),
                torch.ones(n, device=dev, dtype=dt), torch.zeros((), device=dev, dtype=dt),
                torch.zeros((n, self.P0['head'].shape[1]), device=dev))

    def step(self, l, state, control, P):
        m, C, p, lam, star, b, previous_alive, radial, hstate = state
        n, dt = self.n, self.dtype
        w, ws = self.w[l], self.ws[l]
        mu = ws.T@m
        Cpre = ((w.T@C.float())@w).to(dt)
        metric = diagonal(C, torch.diagonal(C)*previous_alive)
        Bpre = ((w.T@metric.float())@w).to(dt)
        bpre = ws.T@b
        vv = torch.diagonal(Cpre)
        alpha = mu/vv.clamp_min(1e-10).sqrt()
        gg = ((1+torch.erf(alpha/math.sqrt(2)))/2 if self.device.type == 'mps'
              else torch.erfc(-alpha/math.sqrt(2))/2)
        indices, alive = select(gg)
        live = torch.outer(alive, alive)
        pp = w.T@p
        if len(lam):
            sub = pp[indices]
            subplane = ((sub*sub)@(sub*lam).T).to(dt)
            # index_put is differentiable in its values, indices are constant
            # on each selection cell. No gradients across top-k boundaries.
            cp = torch.zeros_like(Cpre).index_put((indices[:,None], indices[None,:]), subplane)
            cpdiag = (pp.to(dt)**3*lam.to(dt)).sum(1)
            cp = diagonal(cp, cpdiag)
        else:
            cp = torch.zeros_like(Cpre)
        wt = w.T
        wa = wt@star
        sp = (2*(wa*wt)@wt.T+(wt*wt)@wa.T).to(dt)/3
        sp = diagonal(sp*live, torch.diagonal(sp))
        k21 = cp+sp+torch.diagonal(Bpre)[:,None]*bpre[None,:]+2*Bpre*bpre[:,None]
        host, C, target, new_radial, aux = nonlinear(mu, Cpre, k21, radial)
        x = node_features(l, host, C, k21, aux, self.fixed[l]).float()
        delta, hs = gnn(l, x, aux['sigma'].float(), w, hstate, P)
        delta = delta.to(dt)+control
        C = C-torch.outer(host, delta)-torch.outer(delta, host)-torch.outer(delta, delta)
        m = host+delta
        if l == self.depth-1:
            return (m, C, p, lam, star, b, alive, new_radial, hs)
        g = aux['gate']
        d = aux['pdf']/aux['sigma']
        birth = d*(1+radial/3)
        b = (g*bpre+radial*d*vv/6)*alive
        selected, births = select(torch.minimum(g, 1-g))
        C = diagonal(C*live, torch.diagonal(C))
        target = diagonal(target*live, torch.diagonal(target))
        inherited = g[:,None]**2*g[None,:]*cp
        bp = diagonal(g[:,None]*Cpre, torch.zeros_like(g))
        Bpost = diagonal(C, torch.diagonal(C)*alive)
        trace_post = torch.diagonal(Bpost)[:,None]*b[None,:]+2*Bpost*b[:,None]
        residual = target-inherited-trace_post
        beta = 2*(aux['gaussian_mean']-host*g)
        innovation = torch.diagonal(residual)
        safe_birth = birth.clamp_min(1e-12)
        u = torch.maximum(torch.minimum(beta/safe_birth, 8*vv), -8*vv)
        disc = 4*innovation/safe_birth-3*u*u
        gap = torch.minimum(torch.maximum(disc, .01*vv*vv).sqrt(), 16*vv)
        t, s = (u+gap)/2, (u-gap)/2
        a = birth/gap
        p1, p2 = diagonal(bp, t), diagonal(bp, s)
        born = bp*bp*(birth*births)[None,:]+(beta*births)[:,None]*bp.T
        born = diagonal(born, birth*births*(3*u*u+gap*gap)/4)
        cp_post = (inherited+born)*live
        p = torch.cat((g.float()[:,None]*pp, p1[:,selected].float(), p2[:,selected].float()), dim=1)
        p = alive.float()[:,None]*p
        lam = torch.cat((lam, a[selected].float(), -a[selected].float()))
        rest = target-cp_post-trace_post
        star = (3*rest.T-2*torch.diag(torch.diagonal(rest))).float()
        return (m, C, p, lam, star, b, alive, new_radial, hs)

    def run(self, P=None, controls=None, training=False, capture=False):
        P = self.P0 if P is None else P
        if controls is None:
            controls = torch.zeros((self.depth,self.n), device=self.device, dtype=self.dtype)
        state = self.initial()
        means, trace = [], []
        for l in range(self.depth):
            if training:
                def block(*args, l=l):
                    return self.step(l, args[:-1], args[-1], P)
                state = checkpoint(block, *state, controls[l], use_reentrant=False)
            else:
                state = self.step(l, state, controls[l], P)
            means.append(state[0])
            if capture:
                trace.append(dict(layer=l, columns=state[2].shape[1],
                                  active=int(state[6].sum().item()),
                                  radial=float(state[7].detach().cpu())))
        return torch.stack(means), trace
