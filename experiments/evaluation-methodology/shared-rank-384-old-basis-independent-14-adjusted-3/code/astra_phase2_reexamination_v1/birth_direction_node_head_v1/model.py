"""Qualified birth-direction Torch adapter with a new shared 64x4 readout.

No model call, tensor construction, payload decoding or target read occurs
at import.  Import authenticates the frozen source/JSON lineage, imports the
qualified Torch implementation, and compiles a reversible step adapter.
Instantiation loads only the same weights and frozen P used by that model.
This module does not train, choose coefficients, execute a card, or score.
"""

import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load the pinned source adapter')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


provenance = _module(HERE / 'build.py', 'bdnh_v1_inert_source_builder')
SOURCE_BINDINGS = provenance.authenticate()
CSTAR_LITERAL = provenance.frozen_coefficients()
sys.path.insert(0, str(HERE.parent))
base = _module(provenance.BD / 'model.py', 'bdnh_v1_qualified_birth_direction')
torch = base.torch
np = base.np
diagonal = base.diagonal
residualized_basis = base.residualized_basis
birth_action = base.birth_action


def node_coefficients(H, A, cstar):
    """FP32 (4,n) coefficients, retaining gradients through H and A.

    No tensor is detached here.  In particular frozen P PARAMETERS do not
    mean frozen P inputs or a frozen bank trajectory.  The caller must pass
    the current layer's PRE-mean-correction returned P state.
    """
    n = int(H.shape[0])
    if H.shape != (n, 64) or A.shape != (64, 4) or cstar.shape != (4,):
        raise ValueError('Birth node-head shape mismatch')
    if any(x.dtype != torch.float32 for x in (H, A, cstar)):
        raise TypeError('Birth node-head operands must already be FP32')
    centered = H - H.mean(dim=0)
    scale = ((centered * centered).mean(dim=0) + 1e-6).sqrt()
    standardized = (centered / scale).clamp(-8, 8)
    response = (standardized @ A).tanh()
    coefficients = cstar[None, :] + (1 - cstar.abs())[None, :] * response
    return coefficients.T


# Only call placement and the coefficient argument change.  The original
# residualized basis, actual stored-value plane repair and summation order
# remain the qualified base's functions.  In particular (4,n)[k] broadcasts
# over the LAST (birth-column) axis, never over physical row coordinates.
source = base.source
CHANGES = (
    ('    vv = torch.diagonal(Cpre)\n',
     '    vv = torch.diagonal(Cpre)\n'
     '    self._record_variance(l, vv)\n'),
    ("    delta, hs = gnn(l, x, aux['sigma'].float(), w, hstate, P)\n",
     "    delta, hs = gnn(l, x, aux['sigma'].float(), w, hstate, P)\n"
     '    node_c = self.layer_node_coefficients(l, hs) if l < self.depth-1 else None\n'),
    ('birth_action(bp,Cpre,k21,aux,self.layer_coefficients(l),t,s,a,alive,births)',
     'birth_action(bp,Cpre,k21,aux,node_c,t,s,a,alive,births)'),
)
for old, new in CHANGES:
    source = base.replace_once(source, old, new)
inverse = source
for old, new in reversed(CHANGES):
    inverse = base.replace_once(inverse, new, old)
if inverse != base.source:
    raise RuntimeError('Torch adapter is not an exact inverse of the frozen step')
code, clamp_count = base.rewrite_clamps(source, str(__file__) + ':step')
if clamp_count != 2:
    raise RuntimeError('Unexpected frozen physical clamp count')
env = dict(base.env)
exec(code, env)


class NodeHeadCard(base.BirthDirectionCard):
    """Autonomous uncompressed all-P card; only shared A may be trained.

    A must be FP32 on the same device as the card, or None for a new zero
    leaf.  No P/cstar override or exogenous mean-control path is accepted.
    The small host diagnostic records contain no tensors and never feed
    coefficients.  Selection checks compare checkpoint replay memberships,
    not values from a different network or an old teacher trajectory.
    """

    def __init__(self, rank, A=None, device='mps', weights=None,
                 audit_selections=True):
        cstar = torch.tensor(CSTAR_LITERAL, device=device, dtype=torch.float32)
        super().__init__(rank, coefficients=cstar, device=device, weights=weights,
                         P=None, audit_selections=audit_selections)
        if A is None:
            A = torch.zeros((64, 4), device=self.w.device, dtype=torch.float32,
                            requires_grad=True)
        if not isinstance(A, torch.Tensor) or A.shape != (64, 4):
            raise TypeError('A must be the shared 64-by-4 tensor')
        if A.dtype != torch.float32 or A.device != self.w.device:
            raise TypeError('A must already be FP32 on the card device')
        if not bool(torch.isfinite(A).all().detach().cpu()):
            raise FloatingPointError('Nonfinite node-head A')
        self.A = A
        for value in self.P0.values():
            if value.requires_grad:
                raise RuntimeError('Frozen P must not be trainable')
        self.selection_cells = {}
        self.variance_domain = {}
        self.node_head_calls = {}
        self._layer = None
        self._selection_slot = 0

    def layer_node_coefficients(self, layer, state):
        if layer == 0:
            return self.coefficients[0]
        if not 1 <= layer <= 14:
            raise ValueError('No node head outside birth layers1..14')
        if not bool(torch.isfinite(state).all().detach().cpu()):
            raise FloatingPointError('Nonfinite current P state')
        result = node_coefficients(state, self.A, self.coefficients[layer])
        if not bool(torch.isfinite(result).all().detach().cpu()):
            raise FloatingPointError('Nonfinite node-head coefficients')
        self.node_head_calls[layer] = self.node_head_calls.get(layer, 0) + 1
        return result

    def _record_variance(self, layer, variance):
        value = variance.detach().cpu()
        record = dict(minimum=float(value.min()),
                      below_floor_count=int((value < 1e-10).sum()),
                      finite=bool(torch.isfinite(value).all()))
        self.variance_domain[layer] = record
        if not record['finite'] or record['below_floor_count']:
            raise FloatingPointError('Native/centered prevariance domain failed')

    def _select(self, score):
        indices, mask = base.centered_source.select(score)
        key = (self._layer, self._selection_slot)
        self._selection_slot += 1
        if self.audit_selections:
            ids = indices.detach().cpu().tolist()
            if key in self.selection_cells and self.selection_cells[key] != ids:
                raise RuntimeError('Checkpoint replay changed physical selections')
            self.selection_cells[key] = ids
        return indices, mask

    def step(self, layer, state, control, P):
        if P is not self.P0:
            raise RuntimeError('Only the frozen current P is allowed')
        self._layer, self._selection_slot = layer, 0
        return env['step'](self, layer, state, control, P)

    def run(self, P=None, controls=None, training=False, capture=False):
        if P is not None or controls is not None:
            raise ValueError('No external P, teacher forcing, or mean controls')
        self.selection_cells.clear()
        self.variance_domain.clear()
        self.node_head_calls.clear()
        return super().run(P=None, controls=None, training=training, capture=capture)

    def diagnostics(self):
        return dict(selection_cells={f'{l}:{s}': list(ids)
                    for (l, s), ids in self.selection_cells.items()},
                    variance_domain=dict(self.variance_domain),
                    node_head_calls=dict(self.node_head_calls),
                    frozen_P=True, frozen_cstar=True,
                    allocated_trainable_parameters=256)
