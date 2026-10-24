"""CPU full-carrier derivative for the coherent central-state action."""
import inspect
import textwrap
import torch
import full_carrier_autodiff_v1 as full
from full_carrier_bank_autodiff_v1 import BankFullCard


def scaling(layer, aux, theta):
    s = aux['sigma']
    if layer == 0:
        return torch.ones_like(s)
    raw = torch.stack((aux['pre_k3']/s**3, aux['gate'], s.log()), 1)
    centered = raw - raw.mean(0)
    phi = torch.cat((torch.ones((len(s), 1), dtype=s.dtype, device=s.device),
                     (centered / ((centered**2).mean(0)+1e-12).sqrt()).clamp(-4, 4)), 1)
    z = 2*layer/15-1
    basis = torch.tensor((1, z, (3*z*z-1)/2), dtype=s.dtype, device=s.device)
    return (.002 * torch.tanh(phi @ (basis @ theta))).exp()


def transform(C, p, star, b, scale):
    return (scale[:, None]*C*scale[None, :],
            (scale[:, None]*p).to(p.dtype),
            (scale[:, None]*star*scale[None, :]**2).to(star.dtype),
            scale*b)


source = textwrap.dedent(inspect.getsource(full.FullCard.step))
old = '    star = (3*rest.T-2*torch.diag(torch.diagonal(rest))).float()'
assert source.count(old) == 1
source = source.replace(old, old + '''
    scale = scaling(l,aux,self.theta)
    C,p,star,b = transform(C,p,star,b,scale)''')
env = dict(vars(full), scaling=scaling, transform=transform)
exec(source, env)
scaled_step = env['step']


class CoherentScaleCard(BankFullCard):
    def __init__(self, rank, theta=None):
        super().__init__(rank, 'cpu')
        self.theta = torch.zeros((3,4), dtype=torch.float64) if theta is None else theta

    def step(self, l, state, control, P):
        if l == self.depth-1:
            return BankFullCard.step(self, l, state, control, P)
        return scaled_step(self, l, state, control, P)
