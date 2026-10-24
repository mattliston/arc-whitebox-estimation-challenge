"""Exact current CPU differentiation, new CP inherited-distinct action."""
import inspect
import textwrap
import torch
import full_carrier_autodiff_v1 as full
from full_carrier_bank_autodiff_v1 import BankFullCard

source=textwrap.dedent(inspect.getsource(full.FullCard.step))
edits=(
    ('    cp_post = (inherited+born)*live',
     '    gain = 1+self.theta[l]\n    cp_post = (inherited*gain**3+born)*live'),
    ('g.float()[:,None]*pp,', '(g*gain).float()[:,None]*pp,'),
)
for before,after in edits:
    assert source.count(before)==1,before
    source=source.replace(before,after)
env=dict(vars(full))
exec(source,env)
changed_step=env['step']


class InheritedDistinctCard(BankFullCard):
    def __init__(self,rank,theta=None):
        super().__init__(rank,'cpu')
        self.theta=torch.zeros(15,dtype=torch.float64) if theta is None else theta

    def step(self,l,state,control,P):
        if l==self.depth-1:
            return BankFullCard.step(self,l,state,control,P)
        return changed_step(self,l,state,control,P)
