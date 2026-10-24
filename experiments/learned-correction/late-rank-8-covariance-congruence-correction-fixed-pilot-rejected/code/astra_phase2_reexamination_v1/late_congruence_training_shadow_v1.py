"""Suffix covariance-action training around the certified baseline shadow."""
import math
import torch
from current_transport_shadow_v3 import Shadow, diagonal, nonlinear, features, gnn


class TrainCongruenceShadow(Shadow):
    def step_train(self, l, dm, dc, dh, P, Q):
        d = self.data
        w = d['weights'][l]
        mu = d['mu'][l]+w.T@dm
        pre = d['pre_cov'][l]+w.T@dc@w
        m, c, aux = nonlinear(mu, pre, d['k21'][l], d['k4'][l])
        m0, c0, f0, delta0, h0 = self.base[l]
        em = m-m0
        f = d['x'][l]+features(l, m, c, d['k21'][l], aux, d['x'][l])-f0
        delta, h = gnn(l, f, aux['sigma'], w, d['state_prev'][l]+dh, P)
        ed = delta-delta0
        base_post = d['host'][l]+d['delta'][l]
        actual_delta = d['delta'][l]+ed
        ec = c-c0-torch.outer(base_post, ed)-torch.outer(ed, base_post)
        ec = ec-torch.outer(em, actual_delta)-torch.outer(actual_delta, em)-torch.outer(ed, ed)
        mask = d['alive'][l]
        ec = diagonal(ec*torch.outer(mask, mask), torch.diagonal(ec))
        if 10 <= l <= 14:
            actual_state = d['state_prev'][l+1]+h-h0
            hf = ((actual_state-Q['centers'][l])/Q['scales'][l]).clamp(-8, 8)
            hf = torch.cat((hf, torch.ones((len(m), 1), device=self.dev)), dim=1)
            sigma = aux['sigma']
            u = .02*sigma[:, None]*torch.tanh(hf@Q['u'][l-10])*mask[:, None]/math.sqrt(len(m))
            v = (hf@Q['v'])*mask[:, None]/(sigma[:, None]*math.sqrt(len(m)))
            host0, delta_base = d['host'][l], d['delta'][l]
            cbase = c0-torch.outer(host0, delta_base)-torch.outer(delta_base, host0)-torch.outer(delta_base, delta_base)
            cbase = diagonal(cbase*torch.outer(mask, mask), torch.diagonal(cbase))
            current = cbase+ec
            cv = current@v
            ec = ec+u@cv.T+cv@u.T+u@(v.T@cv)@u.T
        return em+ed, ec, h-h0
