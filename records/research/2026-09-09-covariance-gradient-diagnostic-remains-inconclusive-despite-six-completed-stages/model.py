"""Inert d50 covariance-action gradient-gap adapter, not a fitted estimator.

Import and authenticate() use only stdlib; numerical modules load on an
explicit action/model call.  Supply V0 externally: the protocol owns its
seed9309095 Gaussian(65,8)/sqrt(65) construction, never this class.

full/read_cut have identical arithmetic. read_cut stops the incoming k21
and radial locals immediately before nonlinear(), including their later
P/birth uses. It does NOT freeze k4=radial*diag(Cpre)^2, Cpre, mu, H, or
the independently inherited CP path; it is not the historical tape shadow.

At A=0 the action is algebraically zero. Literal additions may change a
stored -0 to +0: null BYTE parity is a qualification obligation, not an
assumption. No zero-A bypass, symmetrization, PSD repair or tensor repair.
"""

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
BD = NEW / 'birth_direction_fit_v1'
PARENT = BD / 'build_terminal_v1/estimator.py'
THETA = BD / 'training_v1/terminal_theta.json'
BUILD = BD / 'build_terminal_v1/build.json'
PINS = {
    PARENT: 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451',
    THETA: 'de0eb2c2b3dbda9cef1e6d31146755f18f400aaa47f793460017fd9040753337',
    BUILD: '4889c6246e8e0dd420cdb8fbca3ccbce721a163befd226339ebe6d0023cc070f',
    BD / 'model.py': 'cea1ea473db821cea6b4b9d34aeabf06f43ded8c5d77ee4037dd3f3d269df613',
    BD / 'terminal_score_v2/result.json': '884089500b2f7f31cc9f1fbe343c6396df3cf976daee63141915715f8064c29f',
    BD / 'panel_v2/score/result.json': '767cf92415a0391f7b7596c9681adede144bd10b3e5d36f354ec2dce7ce52c1b',
    NEW / 'payload_response_rows_v1.npz': '6b55c0d6071b2350e53e243bcd4137651e2c80a2cd55d04467bf194d0ea54a1c',
    NEW / 'centered_bank_autodiff_v1.py': 'c70e7bc5de06e59d1c95a41658f406a4849820c28ecd8e5aa4edd48ea67ca3e2',
    NEW / 'centered_bank_autodiff_v3.py': '45c88f1e73d694f0781c4f0a01053511addbe465a6746ef11ac38378b093983c',
    NEW / 'full_carrier_autodiff_v1.py': 'd6480f443e807688dca380d19d34e7cfe5c1f7e625104aae87201cfba7d199f7',
    NEW / 'full_carrier_bank_autodiff_v1.py': 'fc6fb6f5bbe8451da88ba78bcc3dfaa0087f3c66d1b62c5e6ceed6362a0b5b1d',
    NEW / 'centered_torch_numerics_v1.py': '2b62c3bf08ec59f7d0f749fc4526417c0c2952ed274b8aa5cba67762ff7929cf',
    NEW / 'centered_torch_numerics_v3.py': '18df077006603d1b434216b2a4ffebc1b1d1e631983163fa0a9213eb0a11412e',
    NEW / 'current_transport_shadow_v1.py': '63e9302de34dbf078bcbedc088594b00c88531543a17c3ec032386b5c8fbb2a3',
    NEW / 'current_transport_shadow_v3.py': '2193ccf82b6fcb3dada1eeb3f2a75910d0b66342ccb8da0f7c55d4886d013326',
    NEW / 'joint_prefix_final_data_v1.py': '22fff8465a13e412c68c1e5c204d7bfa0ce3d7c1cb9b99e1e5378098a9fd8b6b',
    NEW / 'tensor_clamp_min_v1.py': 'c68768a7dc17a09c6e9105817a1b9d958b2409d076dab16f5ea427d5439e9c6b',
    NEW / 'gaussian_cdf_quadrature_v1.py': 'f6f96fbc8f47bd4ffab93ba9cafc805b2ea79435f5a2106649d94a3bc0754398',
}
_CARD_CLASS = None


def authenticate():
    """Hash frozen code/data as bytes, never import a numerical dependency."""
    records = {}
    for path, expected in PINS.items():
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected:
            raise RuntimeError('Frozen covariance-gap dependency changed: ' + str(path))
        records[str(path.resolve())] = dict(path=str(path.resolve()), sha256=digest, bytes=len(data))
    own = Path(__file__).resolve()
    data = own.read_bytes()
    records[str(own)] = dict(path=str(own), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    return records


def frozen_coefficients():
    """Read the already rounded deployed cstar; do not recompute tanh."""
    authenticate()
    document, build = json.loads(THETA.read_text()), json.loads(BUILD.read_text())
    if not (build['passed'] and build['inverse_source_byte_exact']
            and build['candidate_sha256'] == PINS[PARENT] and document['terminal_epoch'] == 6):
        raise RuntimeError('Unqualified terminal source or coefficient checkpoint')
    matches = [n.value for n in ast.parse(PARENT.read_text()).body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == '_BD_COEFFICIENT_LITERAL' for t in n.targets)]
    if len(matches) != 1:
        raise RuntimeError('Expected exactly one deployed coefficient literal')

    def encoded(rows):
        if len(rows) != 15 or any(len(row) != 4 for row in rows):
            raise ValueError('Expected 15-by-4 cstar')
        values = [float(x) for row in rows for x in row]
        if not all(math.isfinite(x) and struct.unpack('<f', struct.pack('<f', x))[0] == x for x in values):
            raise ValueError('cstar must contain exact finite FP32 literals')
        return struct.pack('<60f', *values)

    rows = document['coefficients']
    if encoded(rows) != encoded(build['coefficient_literal']) or encoded(rows) != encoded(ast.literal_eval(matches[0])):
        raise RuntimeError('Training/build/deployed cstar bytes disagree')
    return tuple(tuple(float(x) for x in row) for row in rows)


def head_factors(H, sigma, alive, A, V0):
    """Pure supplied-dtype Torch head; all-FP64 inputs define the smooth oracle.

    No hidden cast, detach, random projection or target-dependent operation.
    The class separately supplies FP32 operands, then widens the returned
    factors to its moment dtype. sigma means pre-map aux['sigma'].
    """
    import torch
    n = H.shape[0]
    if H.shape != (n, 64) or A.shape != (65, 8) or V0.shape != (65, 8):
        raise ValueError('Covariance head shape mismatch')
    if sigma.shape != (n,) or alive.shape != (n,):
        raise ValueError('Covariance head vector shape mismatch')
    if H.dtype not in (torch.float32, torch.float64) or any(x.dtype != H.dtype or x.device != H.device
                                                          for x in (sigma, alive, A, V0)):
        raise TypeError('Head operands must share FP32 or FP64 dtype/device')
    centered = H - H.mean(dim=0)
    Z = (centered / ((centered * centered).mean(dim=0) + 1e-6).sqrt()).clamp(-8, 8)
    h = torch.cat((Z, torch.ones((n, 1), device=H.device, dtype=H.dtype)), dim=1)
    u = .02 * sigma[:, None] * (h @ A).tanh() * alive[:, None] / math.sqrt(n)
    v = (h @ V0) * alive[:, None] / (sigma[:, None] * math.sqrt(n))
    return u, v


def congruence(C, u, v):
    """Literal stored-C congruence; no symmetry assumption or precision cast."""
    n = C.shape[0]
    if C.shape != (n, n) or u.ndim != 2 or u.shape != v.shape or u.shape[0] != n:
        raise ValueError('Congruence shape mismatch')
    if any(x.dtype != C.dtype or x.device != C.device for x in (u, v)):
        raise TypeError('Congruence operands must share dtype/device')
    right, left = C @ v, v.T @ C
    return C + u @ left + right @ u.T + (u @ (v.T @ right)) @ u.T


def covariance_action(C, H, sigma, alive, A, V0):
    """Pure same-dtype composition; the class owns deployment-precision casts."""
    u, v = head_factors(H, sigma, alive, A, V0)
    return congruence(C, u, v)


def covariance_action_numpy64(C, H, sigma, alive, A, V0):
    """Independent ideal FP64 algebra reference, not an FP32 primal oracle."""
    import numpy as np
    C, H, sigma, alive, A, V0 = (np.asarray(x, dtype=np.float64) for x in (C, H, sigma, alive, A, V0))
    n = C.shape[0]
    centered = H - H.mean(axis=0)
    h = np.concatenate((np.clip(centered / np.sqrt((centered * centered).mean(axis=0) + 1e-6), -8, 8),
                        np.ones((n, 1))), axis=1)
    u = .02 * sigma[:, None] * np.tanh(h @ A) * alive[:, None] / math.sqrt(n)
    v = (h @ V0) * alive[:, None] / (sigma[:, None] * math.sqrt(n))
    transform = np.eye(n) + u @ v.T
    return transform @ C @ transform.T


def _load_card_class():
    global _CARD_CLASS
    authenticate()
    if _CARD_CLASS is not None:
        return _CARD_CLASS
    sys.path.insert(0, str(NEW))
    spec = importlib.util.spec_from_file_location('covariance_gap_v1_frozen_bd', BD / 'model.py')
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load the pinned BD adapter')
    base = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = base
    spec.loader.exec_module(base)
    torch = base.torch
    changes = (
        ('    vv = torch.diagonal(Cpre)\n', '    vv = torch.diagonal(Cpre)\n    self._record_variance(l, vv)\n'),
        ('    host, C, target, new_radial, aux = nonlinear(mu, Cpre, k21, radial)\n',
         "    if self.mode == 'read_cut':\n        k21, radial = k21.detach(), radial.detach()\n"
         '    host, C, target, new_radial, aux = nonlinear(mu, Cpre, k21, radial)\n'),
        ('    C = diagonal(C*live, torch.diagonal(C))\n',
         '    C = diagonal(C*live, torch.diagonal(C))\n'
         "    C = self._covariance_write(l, C, hs, aux['sigma'], alive)\n"),
    )
    source = base.source
    for old, new in changes:
        source = base.replace_once(source, old, new)
    inverse = source
    for old, new in reversed(changes):
        inverse = base.replace_once(inverse, new, old)
    if inverse != base.source:
        raise RuntimeError('Covariance step adapter failed its literal inverse')
    code, count = base.rewrite_clamps(source, str(__file__) + ':step')
    if count != 2:
        raise RuntimeError('Frozen physical clamp count changed')
    env = dict(base.env)
    exec(code, env)

    class Card(base.BirthDirectionCard):
        def __init__(self, rank, V0, A=None, mode='full', device='mps', weights=None, audit_selections=True):
            if mode not in ('full', 'read_cut'):
                raise ValueError('Only full and read_cut are defined')
            cstar = torch.tensor(frozen_coefficients(), device=device, dtype=torch.float32)
            super().__init__(rank, coefficients=cstar, device=device, weights=weights, P=None,
                             audit_selections=audit_selections)
            if A is None:
                A = torch.zeros((5, 65, 8), device=self.w.device, dtype=torch.float32, requires_grad=True)
            for name, value, shape in (('A', A, (5, 65, 8)), ('V0', V0, (65, 8))):
                if not isinstance(value, torch.Tensor) or value.shape != shape:
                    raise TypeError(name + ' shape/type mismatch')
                if value.dtype != torch.float32 or value.device != self.w.device:
                    raise TypeError(name + ' must already be FP32 on the card device')
                if not bool(torch.isfinite(value).all().detach().cpu()):
                    raise FloatingPointError('Nonfinite ' + name)
            if V0.requires_grad or any(x.requires_grad for x in self.P0.values()):
                raise RuntimeError('V0, P and cstar are frozen')
            self.A, self.V0, self.mode = A, V0.clone(), mode
            self.selection_cells, self.variance_domain, self.action_calls = {}, {}, {}
            self._layer, self._selection_slot = None, 0
            self.source_bindings = authenticate()
            self.adapter_source_sha256 = hashlib.sha256(source.encode()).hexdigest()

        def _covariance_write(self, layer, C, H, sigma, alive):
            if not 10 <= layer <= 14:
                return C
            u, v = head_factors(H.float(), sigma.float(), alive.float(),
                                self.A[layer - 10], self.V0)
            result = congruence(C, u.to(C.dtype), v.to(C.dtype))
            if not bool(torch.isfinite(result).all().detach().cpu()):
                raise FloatingPointError('Nonfinite covariance write')
            signs = ((C == 0) & (result == 0) & (torch.signbit(C) != torch.signbit(result))).sum()
            self.action_calls[layer] = dict(zero_sign_changes=int(signs.detach().cpu()),
                                           maximum_absolute_change=float((result - C).detach().abs().max().cpu()))
            return result

        def _record_variance(self, layer, variance):
            value = variance.detach().cpu()
            record = dict(minimum=float(value.min()), below_floor_count=int((value < 1e-10).sum()),
                          finite=bool(torch.isfinite(value).all()))
            self.variance_domain[layer] = record
            if not record['finite'] or record['below_floor_count']:
                raise FloatingPointError('Prevariance domain failed')

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
            if P is not self.P0 or bool((control.detach() != 0).any().cpu()):
                raise RuntimeError('No P override or exogenous mean control')
            self._layer, self._selection_slot = layer, 0
            return env['step'](self, layer, state, control, P)

        def run(self, P=None, controls=None, training=False, capture=False):
            if P is not None or controls is not None:
                raise ValueError('No P overrides, controls or teacher forcing')
            self.selection_cells.clear()
            self.variance_domain.clear()
            self.action_calls.clear()
            return super().run(P=None, controls=None, training=training, capture=capture)

        def diagnostics(self):
            return dict(mode=self.mode, frozen_P=True, frozen_cstar=True, frozen_V0=True,
                        allocated_trainable_parameters=2600, inverse_source_byte_exact=True,
                        adapter_source_sha256=self.adapter_source_sha256,
                        variance_domain=dict(self.variance_domain), action_calls=dict(self.action_calls),
                        selection_cells={f'{l}:{s}': list(ids) for (l, s), ids in self.selection_cells.items()})

    _CARD_CLASS = Card
    return Card


class CovarianceGapCard:
    """Lazy constructor with the underlying qualified card's run/step APIs."""
    def __new__(cls, rank, V0, A=None, mode='full', device='mps', weights=None, audit_selections=True):
        return _load_card_class()(rank, V0, A=A, mode=mode, device=device, weights=weights,
                                  audit_selections=audit_selections)
