"""New offline bank class: identical carrier and heads, centered arithmetic.

Compile explicit copies of the frozen step bodies in a separate namespace;
do not monkeypatch the old modules or change any frozen payload or selector.
"""
import inspect
import math
import textwrap
import numpy as np
import torch
from full_carrier_autodiff_v1 import HERE, FullCard, diagonal, select, node_features, gnn
from full_carrier_bank_autodiff_v1 import BankFullCard, scalar_features, final_read
from centered_torch_numerics_v1 import marginals, nonlinear, normal_cdf


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


prefix_source = textwrap.dedent(inspect.getsource(FullCard.step))
prefix_source = replace_once(prefix_source,
    "    gg = ((1+torch.erf(alpha/math.sqrt(2)))/2 if self.device.type == 'mps'\n"
    "          else torch.erfc(-alpha/math.sqrt(2))/2)",
    "    gg = normal_cdf(alpha)")
prefix_source = replace_once(prefix_source, "    beta = 2*(aux['gaussian_mean']-host*g)",
                             "    beta = aux['birth_beta']")
assert prefix_source.count('= select(') == 2
prefix_source = prefix_source.replace('= select(', '= self._select(')
final_source = textwrap.dedent(inspect.getsource(BankFullCard.step))
final_source = replace_once(final_source,
    '    if l < self.depth-1:\n        return super().step(l,state,control,P)\n', '')
final_source = replace_once(final_source, '= raw_marginals(', '= marginals(')
_prefix_env = dict(torch=torch, math=math, diagonal=diagonal, node_features=node_features,
                   gnn=gnn, nonlinear=nonlinear, normal_cdf=normal_cdf)
_final_env = dict(torch=torch, diagonal=diagonal, marginals=marginals,
                  scalar_features=scalar_features, final_read=final_read)
exec(compile(prefix_source, __file__+':prefix', 'exec'), _prefix_env)
exec(compile(final_source, __file__+':final', 'exec'), _final_env)


class CenteredBankFullCard(FullCard):
    def __init__(self, rank, device='cpu', scalar_dtype=None, audit_selections=False):
        super().__init__(rank, device, scalar_dtype)
        self.Q0 = {k:torch.tensor(v, device=self.device, dtype=torch.float32)
                   for k,v in dict(np.load(HERE/'payload_fullpair_final_head_v1.npz')).items()}
        self.audit_selections = audit_selections
        self.selections = []

    def _select(self, score):
        ix, mask = select(score)
        if self.audit_selections:
            self.selections.append((ix.detach().cpu().numpy().copy(),
                                    mask.detach().cpu().numpy().copy()))
        return ix, mask

    def step(self, l, state, control, P):
        fn = _prefix_env['step'] if l < self.depth-1 else _final_env['step']
        return fn(self, l, state, control, P)
