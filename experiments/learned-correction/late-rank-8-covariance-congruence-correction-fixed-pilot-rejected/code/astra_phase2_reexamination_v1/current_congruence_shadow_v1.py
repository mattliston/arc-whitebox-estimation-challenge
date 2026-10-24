"""Versioned extension: fixed rank-eight covariance actions on the shadow."""
import torch
from current_transport_shadow_v3 import Shadow, diagonal


class CongruenceShadow(Shadow):
    def step_action(self, l, dm, dc, dh, u, v):
        zero = torch.zeros_like(dm)
        dm, dc, dh = self.step(l, dm, dc, dh, zero, self.P0)
        if not 10 <= l <= 14:
            return dm, dc, dh
        data = self.data
        _, c0, _, _, _ = self.base[l]
        host, delta = data['host'][l], data['delta'][l]
        c0 = c0-torch.outer(host, delta)-torch.outer(delta, host)-torch.outer(delta, delta)
        mask = data['alive'][l]
        c0 = diagonal(c0*torch.outer(mask, mask), torch.diagonal(c0))
        current = c0+dc
        cv = current@v
        dc = dc+u@cv.T+cv@u.T+u@(v.T@cv)@u.T
        return dm, dc, dh

    def run_action(self, u, v):
        dm = torch.zeros(1024, device=self.dev)
        dc = torch.zeros((1024, 1024), device=self.dev)
        dh = torch.zeros((1024, 64), device=self.dev)
        outputs = []
        for l in range(16):
            dm, dc, dh = self.step_action(l, dm, dc, dh, u[l], v[l])
            outputs.append(dm)
        return torch.stack(outputs)
