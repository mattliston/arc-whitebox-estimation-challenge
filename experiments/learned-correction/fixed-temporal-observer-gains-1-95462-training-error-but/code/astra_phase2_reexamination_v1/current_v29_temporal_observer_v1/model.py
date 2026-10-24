#!/usr/bin/env python3
"""Inert, fixed terminal-only observer. No fit, labels, physical model or port.

This is an explicit transcription of the pinned selected mean-only arithmetic,
not an import of the old training module or a whole-old-class source inverse.
Only initialize()/step()/forward() request the separately admitted Torch runtime.
Every production/proposal operation is FP32. reference64 is restricted to the
synthetic n=8 CPU qualification and never changes initialization draw precision.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
P2 = NEW.parent
PINS = {
    HERE/'TRAIN_DESIGN.md': '4c9124877e509570e4d6fcfb2d487dc33154267610a0295c775c6f1e1f60332f',
    HERE/'QUALIFICATION_DESIGN.md': 'dd036422f9907e4c0134d2e444f3d969fb68bf15599b7f86a05f5c811ef90b89',
    HERE/'adapter.py': 'cab844cd7981d31c7e44f4797ad60490729bdfadb8ce46655cdd5966ae9f36b7',
    HERE/'PROTOCOL.md': '2a93907e69427a55c9a11eef7bd2143d9f71835d62f5eb4323700c1a8bd545ba',
    NEW/'radial_gnn_numpy.py': '519abaee0601ea71366fe3a74bd07397064c31257b17ab31b26fca5a39802e29',
    P2/'public_g22_operator_features_v1/train_gnn_oracle.py':
        'f755208b22fc07b4e1c77929100eaad4f0345e2fabfcd7603363d491e415a9bd',
}
SHAPES = {
    'M': (64, 48), 'b_M': (48,),
    'E0': (126, 128), 'b0': (128,),
    'E2': (128, 128), 'b2': (128,),
    'E4': (128, 64), 'b4': (64,),
    'q': (64,), 'b': (),
}
PARAMETER_KEYS = tuple(SHAPES)
DRAW_ORDER = (('M', 64), ('E0', 126), ('E2', 128), ('E4', 128))
INITIALIZATION_SEED = 93091229
ALLOCATED_PARAMETERS = 44209
ZERO_INPUT_PARAMETERS = 384
MAX_CONNECTED_PARAMETERS = 43825
FROZEN_LITERALS = 961
DEPLOYED_LITERALS = 45170
_TORCH = None


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def authenticate():
    """Read only direct source owners; no numerical import or data file."""
    require(sum(math.prod(shape) for shape in SHAPES.values()) == ALLOCATED_PARAMETERS,
            'Exact ten-tensor allocated count')
    pins = {}
    for path, digest in PINS.items():
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                h.update(block)
        require(h.hexdigest() == digest, 'Immutable observer source owner: '+str(path))
        pins[str(path)] = digest
    return pins


def runtime():
    """Called only after a separately reviewed numerical worker admission."""
    global _TORCH
    require('threadpoolctl' not in sys.modules and 'flopscope' not in sys.modules,
            'Observer-only Torch runtime; no threadpoolctl or physical FNP runtime')
    if _TORCH is None:
        require(importlib.metadata.version('numpy') == '1.26.4'
                and importlib.metadata.version('torch').split('+')[0] == '2.7.0',
                'Pinned AP numerical packages')
        import torch
        require(torch.__version__.split('+')[0] == '2.7.0', 'Actual Torch version')
        _TORCH = torch
    return _TORCH


def initialize(device='mps', *, trainable=True):
    """Fresh positive-zero biases/head; exact four PCG64DXSM draw blocks."""
    authenticate()
    torch = runtime()
    require(type(trainable) is bool and torch.device(device).type in ('cpu', 'mps'),
            'Explicit CPU/MPS parameter initialization')
    import numpy as np
    require(np.__version__ == '1.26.4', 'Actual initialization NumPy version')
    rng = np.random.Generator(np.random.PCG64DXSM(INITIALIZATION_SEED))
    values = {}
    for name, fan_in in DRAW_ORDER:
        bound = 1/math.sqrt(fan_in)
        values[name] = rng.uniform(-bound, bound, size=SHAPES[name]).astype(np.float32)
    for name in PARAMETER_KEYS:
        if name not in values:
            values[name] = np.zeros(SHAPES[name], dtype=np.float32)
    parameters = {name: torch.tensor(values[name], dtype=torch.float32,
                                    device=device, requires_grad=trainable)
                  for name in PARAMETER_KEYS}
    require(sum(value.numel() for value in parameters.values()) == ALLOCATED_PARAMETERS,
            'Actual allocated parameter count')
    del values, rng
    return parameters


def _tensor(value, shape, dtype, device, name, *, fixed=False):
    torch = runtime()
    require(isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(shape)
            and value.dtype == dtype and value.device == device, 'Exact tensor '+name)
    if fixed:
        require(not value.requires_grad, 'Frozen input '+name)


def _parameters(parameters, dtype, device):
    require(set(parameters) == set(PARAMETER_KEYS), 'Exactly ten parameter blocks')
    require(len({id(parameters[name]) for name in PARAMETER_KEYS}) == len(PARAMETER_KEYS),
            'Distinct supplied parameter tensor identities')
    for name, shape in SHAPES.items():
        _tensor(parameters[name], shape, dtype, device, name)
        require(parameters[name].is_leaf, 'Supplied parameter leaf '+name)


def _finite(values, where):
    """One scalar synchronization for individually scanned consumed buffers."""
    torch = runtime()
    checks = [torch.isfinite(value).all() for value in values]
    require(bool(torch.stack(checks).all().item()), 'All finite: '+where)


def _precision(value, n, reference64):
    torch = runtime()
    dtype = torch.float64 if reference64 else torch.float32
    require(value.dtype == dtype, 'Exact observer precision')
    require(not torch.is_autocast_enabled(value.device.type), 'No observer autocast')
    if reference64:
        require(n == 8 and value.device.type == 'cpu', 'Tiny CPU64 reference only')
    else:
        require(n in (8, 1024) and value.device.type in ('cpu', 'mps'),
                'Fixed tiny or full-width FP32 observer')
    return dtype, value.device


def step(layer, x, state, W, parameters, center, scale, *, reference64=False):
    """Live update; no readout, input mutation, parameter replacement or detach."""
    torch = runtime()
    require(isinstance(layer, int) and 1 <= layer <= 15, 'No layer0 encoder')
    require(x.ndim == 2, 'Feature matrix')
    n = x.shape[0]
    dtype, device = _precision(x, n, reference64)
    _tensor(x, (n, 30), dtype, device, 'feature row', fixed=True)
    _tensor(state, (n, 64), dtype, device, 'live H')
    _tensor(W, (n, n), dtype, device, 'canonical W', fixed=True)
    _tensor(center, (30,), dtype, device, 'center row', fixed=True)
    _tensor(scale, (30,), dtype, device, 'scale row', fixed=True)
    _parameters(parameters, dtype, device)
    require(bool((scale > 0).all().item()), 'Positive frozen feature scales')

    shifted = x-center
    standardized = shifted/scale
    xl = torch.clamp(standardized, -8., 8.)
    g = xl[:, 5:6]
    src = state @ parameters['M'] + parameters['b_M']
    s1, s2, s3 = src[:, :16], src[:, 16:32], src[:, 32:48]
    w2 = W*W
    w3 = w2*W
    w4 = w2*w2
    w6 = w3*w3
    m1 = W.T @ s1
    m2 = w2.T @ s2
    m3 = w3.T @ s3
    sq1, sq2, sq3 = s1*s1, s2*s2, s3*s3
    e1 = w2.T @ sq1
    e2 = w4.T @ sq2
    e3 = w6.T @ sq3
    p1, p2, p3 = torch.clamp(e1, min=0.), torch.clamp(e2, min=0.), torch.clamp(e3, min=0.)
    l1, l2, l3 = torch.log1p(p1), torch.log1p(p2), torch.log1p(p3)
    gm1, gm2, gm3 = g*m1, g*m2, g*m3
    messages = torch.cat((gm1, gm2, gm3, l1, l2, l3), dim=1)
    clipped_messages = torch.clamp(messages, -8., 8.)
    v = torch.cat((xl, clipped_messages), dim=1)

    a0 = v @ parameters['E0'] + parameters['b0']
    neg0 = -a0
    exp0 = torch.exp(neg0)
    den0 = 1+exp0
    u1 = a0/den0
    a2 = u1 @ parameters['E2'] + parameters['b2']
    neg2 = -a2
    exp2 = torch.exp(neg2)
    den2 = 1+exp2
    u2 = a2/den2
    next_state = u2 @ parameters['E4'] + parameters['b4']
    _finite((x, state, center, scale, W, shifted, standardized, xl, src,
             w2, w3, w4, w6, m1, m2, m3, sq1, sq2, sq3, e1, e2, e3,
             p1, p2, p3, l1, l2, l3, gm1, gm2, gm3, messages,
             clipped_messages, v, a0, neg0, exp0, den0, u1,
             a2, neg2, exp2, den2, u2, next_state), 'observer layer '+str(layer))
    return next_state


def forward(features, sigma, host, weights, parameters, centers, scales, r,
            *, reference64=False, retain_tiny_states=False):
    """Cached observer-only rollout. Caller owns graph retirement and all seals.

    Returns served[16,n], delta[n], state[n,64], and states (empty except
    the explicitly requested tiny diagnostic). No source, target or weight
    file is opened here and no parameter/normalizer/input tensor is written.
    """
    torch = runtime()
    require(features.ndim == 3, 'Full feature trajectory')
    n = features.shape[1]
    dtype, device = _precision(features, n, reference64)
    _tensor(features, (16, n, 30), dtype, device, 'features', fixed=True)
    _tensor(sigma, (16, n), dtype, device, 'sigma', fixed=True)
    _tensor(host, (16, n), dtype, device, 'host', fixed=True)
    _tensor(weights, (16, n, n), dtype, device, 'weights', fixed=True)
    _tensor(centers, (16, 30), dtype, device, 'centers', fixed=True)
    _tensor(scales, (16, 30), dtype, device, 'scales', fixed=True)
    _tensor(r, (), dtype, device, 'residual scale', fixed=True)
    _parameters(parameters, dtype, device)
    require(not retain_tiny_states or n == 8, 'No full-width state-tape retention')
    require(bool((sigma > 0).all().item()) and bool((r > 0).item()),
            'Positive captured sigma and frozen residual scale')
    _finite((features, sigma, host, centers, scales, r, *parameters.values()),
            'fixed trajectory and supplied parameter leaves')
    state = torch.zeros((n, 64), dtype=dtype, device=device)
    states = [state] if retain_tiny_states else []
    for layer in range(1, 16):
        state = step(layer, features[layer], state, weights[layer], parameters,
                     centers[layer], scales[layer], reference64=reference64)
        if retain_tiny_states:
            states.append(state)
    head = state @ parameters['q'] + parameters['b']
    physical_scale = sigma[15]*r
    delta = physical_scale*head
    raw_terminal = host[15]+delta
    terminal = torch.clamp(raw_terminal, min=0.)
    served = torch.cat((host[:15], terminal[None, :]), dim=0)
    _finite((head, physical_scale, delta, raw_terminal, terminal, served), 'terminal readout')
    return dict(served=served, delta=delta, state=state, states=tuple(states))
