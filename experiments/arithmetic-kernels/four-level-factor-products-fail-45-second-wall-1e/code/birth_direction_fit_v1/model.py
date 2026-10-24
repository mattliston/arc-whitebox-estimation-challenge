"""Offline current all-P response card and direct common-base intervention.

No targets, fit, numerical job, or candidate imports at module import.
The deployment reference remains the separately built FNP implementation.
"""
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))

import numpy as np
import torch
import centered_bank_autodiff_v1 as centered_source
import centered_bank_autodiff_v3 as certified
from full_carrier_autodiff_v1 import diagonal
from joint_prefix_final_data_v1 import fixed_features, weight_path, TRAIN, VALID
from tensor_clamp_min_v1 import tensor_clamp_min, rewrite_clamps


def residualized_basis(B, C, K21, aux):
    """Covariance-unit directions; no detach of endogenous state."""
    sigma, alpha, pdf, gate = (aux[k] for k in ('sigma','alpha','pdf','gate'))
    d, q = pdf / sigma, alpha / sigma
    t3 = torch.diagonal(K21) / sigma**3
    raw = (.5*d[:,None]*K21,
           -.5*gate[:,None]*K21.T*q[None,:],
           -.5*d[:,None]*(C*C)*q[None,:],
           C*(t3*(alpha*alpha-1)*pdf/6)[:,None])
    zero = torch.zeros_like(sigma)
    denominator = tensor_clamp_min((B*B).sum(0), 1e-20)
    result = []
    for item in raw:
        item = diagonal(item, zero)
        item = item-B*((B*item).sum(0)/denominator)[None,:]
        result.append(diagonal(item, zero))
    return result


def birth_action(B, C, K21, aux, coefficients, t, s, a, alive, births):
    """New stored base and repeated-plane delta using stored pair scalars."""
    bases = residualized_basis(B, C, K21, aux)
    shift = sum(coefficients[j].to(B.dtype)*bases[j] for j in range(4))
    old = (B*alive[:,None]).float()
    new = ((B+shift)*alive[:,None]).float()
    old_h, new_h = old.to(B.dtype), new.to(B.dtype)
    difference = new_h-old_h
    aa = a.float().to(B.dtype)
    tt = (t*alive).float().to(B.dtype)
    ss = (s*alive).float().to(B.dtype)
    w = aa*(tt-ss)*births
    z = aa*(tt*tt-ss*ss)*births
    change = (2*old_h*difference+difference*difference)*w[None,:]
    change = change+z[:,None]*difference.T
    change = diagonal(change, torch.zeros_like(t))
    return new, change


def omitted_source(pp, lam, cpdiag, omitted, alive, aux, host, dtype):
    p = pp.to(dtype)
    a = (aux['gate']-host*aux['pdf']/aux['sigma'])*omitted
    beta = aux['birth_beta']*alive
    energy = (p*p).T@a
    column = p.T@beta
    total = (lam.to(dtype)*energy*column).sum()-(cpdiag*a*beta).sum()
    return 6*total


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Unexpected frozen source span: '+old)
    return source.replace(old, new)


# Reuse the previously certified centered complete-carrier identity, but
# retain the full P head at EVERY layer and the current response-row law.
source = centered_source.prefix_source
source = replace_once(source, '    pp = w.T@p',
    "    response_score = torch.where(alive.bool(), torch.minimum(gg,1-gg), -torch.ones_like(gg))\n"
    "    response_indices, response_mask = self._select(response_score)\n"
    "    omitted = alive*(1-response_mask)\n"
    "    pp = w.T@p")
source = replace_once(source,
    '        sub = pp[indices]\n        subplane = ((sub*sub)@(sub*lam).T).to(dt)',
    '        left, right = pp[response_indices], pp[indices]\n'
    '        subplane = ((left*left)@(right*lam).T).to(dt)')
source = replace_once(source,
    'cp = torch.zeros_like(Cpre).index_put((indices[:,None], indices[None,:]), subplane)',
    'cp = torch.zeros_like(Cpre).index_put((response_indices[:,None], indices[None,:]), subplane)')
source = replace_once(source,
    '    x = node_features(l, host, C, k21, aux, self.fixed[l]).float()',
    '    if len(lam) and l < self.depth-1:\n'
    '        extra = omitted_source(pp,lam,cpdiag,omitted,alive,aux,host,dt)\n'
    '        new_radial = new_radial+extra/(torch.trace(C)**2+2*(C*C).sum())\n'
    '    x = node_features(l, host, C, k21, aux, self.fixed[l]).float()')
source = replace_once(source,
    '    p1, p2 = diagonal(bp, t), diagonal(bp, s)',
    '    stored_base, plane_increment = birth_action(bp,Cpre,k21,aux,self.layer_coefficients(l),t,s,a,alive,births)\n'
    '    p1, p2 = diagonal(stored_base, (t*alive).float()), diagonal(stored_base, (s*alive).float())')
source = replace_once(source,
    '    born = diagonal(born, birth*births*(3*u*u+gap*gap)/4)',
    '    born = diagonal(born, birth*births*(3*u*u+gap*gap)/4)\n'
    '    born = born+plane_increment')
code, clamp_count = rewrite_clamps(source, str(__file__)+':step')
assert clamp_count == 2
env = dict(certified._prefix_env, birth_action=birth_action,
           omitted_source=omitted_source)
exec(code, env)


class BirthDirectionCard(centered_source.CenteredBankFullCard):
    def __init__(self, rank, theta=None, coefficients=None, device='mps',
                 weights=None, P=None, audit_selections=False):
        assert (theta is None) != (coefficients is None)
        self.device = torch.device(device)
        self.dtype = torch.float32 if device == 'mps' else torch.float64
        wn = np.load(weight_path(rank), mmap_mode='r') if weights is None else weights
        self.n, self.depth = wn.shape[1], len(wn)
        self.w = torch.tensor(wn, device=self.device, dtype=torch.float32)
        self.ws = self.w.to(self.dtype)
        self.fixed = torch.tensor(fixed_features(wn), device=self.device, dtype=self.dtype)
        if P is None:
            with np.load(PARENT/'payload_response_rows_v1.npz') as payload:
                P = {k:payload[k].copy() for k in payload.files}
        self.P0 = {k:torch.as_tensor(v, device=self.device, dtype=torch.float32)
                   for k,v in P.items()}
        self.theta = theta
        self.coefficients = coefficients
        self.audit_selections = audit_selections
        self.selections = []

    def layer_coefficients(self, layer):
        return (self.theta[layer].tanh() if self.theta is not None
                else self.coefficients[layer])

    def step(self, layer, state, control, P):
        return env['step'](self, layer, state, control, P)


def deployment_coefficients(theta):
    """Deterministic offline data definition, also used by the native builder."""
    values = np.asarray(theta, dtype=np.float32)
    return np.asarray([[math.tanh(float(x)) for x in row] for row in values],
                      dtype=np.float32)
