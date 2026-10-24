"""CPU autograd port of the stored-coefficient conditional birth increment."""
import inspect
import textwrap
import torch
import full_carrier_autodiff_v1 as full
from full_carrier_bank_autodiff_v1 import BankFullCard


def birth_scale(layer,state,birth,births,theta):
    if layer == 0:
        return torch.ones_like(birth,dtype=torch.float32)
    h = state.float()
    weight = (birth*births).float()
    weight = weight/weight.sum().clamp_min(1e-20)
    centered = h-(weight[:,None]*h).sum(0)
    rms = ((weight[:,None]*centered*centered).sum(0)+1e-6).sqrt()
    standardized = (centered/rms).clamp(-8,8)
    z = torch.tensor(2*layer/15-1,dtype=torch.float32)
    basis = torch.stack((torch.ones_like(z),z,(3*z*z-1)/2))
    coefficient = basis@theta.float()
    response = torch.tanh(standardized@coefficient)
    return 1+.2*(response-(weight*response).sum())


def coefficient_plane_shift(p1,p2,a0,a1,births):
    dt = p1.dtype
    first = p1.float().to(dt)
    t = torch.diagonal(first)
    s = torch.diagonal(p2.float()).to(dt)
    first = full.diagonal(first,torch.zeros_like(t))
    da = (a1.to(dt)-a0.to(dt))*births
    scale = da*(t-s)
    shift = first*first*scale[None,:]+(scale*(t+s))[:,None]*first.T
    return full.diagonal(shift,scale*(t*t+t*s+s*s))


source = textwrap.dedent(inspect.getsource(full.FullCard.step))
edits = {
    '    cp_post = (inherited+born)*live':
        '''    a0 = a.float()
    eta = birth_scale(l,hs,birth,births,self.theta)
    a1 = a0*eta
    shift = coefficient_plane_shift(p1,p2,a0,a1,births)
    cp_post = (inherited+born+shift)*live''',
    'lam = torch.cat((lam, a[selected].float(), -a[selected].float()))':
        'lam = torch.cat((lam, a1[selected], -a1[selected]))',
}
for old,new in edits.items():
    assert source.count(old) == 1,old
    source = source.replace(old,new)
env = dict(vars(full),birth_scale=birth_scale,coefficient_plane_shift=coefficient_plane_shift)
exec(source,env)
conditional_step = env['step']


class ConditionalBirthCard(BankFullCard):
    def __init__(self,rank,theta=None):
        super().__init__(rank,'cpu')
        self.theta = torch.zeros((3,64),dtype=torch.float32) if theta is None else theta

    def step(self,l,state,control,P):
        if l == self.depth-1:
            return BankFullCard.step(self,l,state,control,P)
        return conditional_step(self,l,state,control,P)
