"""Inert, observation-only adapter for autonomous current-d50 P gradients.

Import/authenticate use only stdlib. Numerical imports and payload decoding
occur only on explicit payload/model calls. Frozen cstar is read as its exact
exported FP32 values, never recomputed. No covariance action, teacher state,
new head, parameter update, or target access is implemented here.
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
PAYLOAD = NEW / 'payload_response_rows_v1.npz'
PKEYS = ('msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b',
         'enc4_w', 'enc4_b', 'head', 'head_b', 'gain')
INACTIVE_KEYS = ('head', 'head_b', 'gain')
PARAMETER_COUNT, ACTIVE_COUNT = 45200, 45134
PINS = {
    PARENT: 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451',
    THETA: 'de0eb2c2b3dbda9cef1e6d31146755f18f400aaa47f793460017fd9040753337',
    BUILD: '4889c6246e8e0dd420cdb8fbca3ccbce721a163befd226339ebe6d0023cc070f',
    BD / 'model.py': 'cea1ea473db821cea6b4b9d34aeabf06f43ded8c5d77ee4037dd3f3d269df613',
    BD / 'terminal_score_v2/result.json': '884089500b2f7f31cc9f1fbe343c6396df3cf976daee63141915715f8064c29f',
    BD / 'panel_v2/score/result.json': '767cf92415a0391f7b7596c9681adede144bd10b3e5d36f354ec2dce7ce52c1b',
    PAYLOAD: '6b55c0d6071b2350e53e243bcd4137651e2c80a2cd55d04467bf194d0ea54a1c',
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
_BASE = _CARD_CLASS = _COEFFICIENTS = _PAYLOAD = None


def require(value, message):
    if not value:
        raise RuntimeError(message)


def authenticate():
    """Flat absolute-path:SHA bindings; hashes bytes without numerical imports."""
    result = {}
    for path, expected in PINS.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, 'Frozen P-card dependency changed: ' + str(path))
        result[str(path.resolve())] = actual
    for path in (Path(__file__).resolve(), HERE / 'PROTOCOL_V1.md'):
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def coefficient_bytes(rows):
    require(len(rows) == 15 and all(len(row) == 4 for row in rows), 'cstar shape')
    values = [float(x) for row in rows for x in row]
    require(all(math.isfinite(x) and struct.unpack('<f', struct.pack('<f', x))[0] == x
                for x in values), 'Exact finite FP32 cstar values required')
    return struct.pack('<60f', *values)


def frozen_coefficients():
    """Return exact deployment values; the constant cache is not mutable state."""
    global _COEFFICIENTS
    authenticate()
    if _COEFFICIENTS is None:
        document, build = json.loads(THETA.read_text()), json.loads(BUILD.read_text())
        require(build['passed'] and build['inverse_source_byte_exact']
                and build['candidate_sha256'] == PINS[PARENT]
                and document['terminal_epoch'] == 6, 'Unqualified d50 checkpoint')
        matches = [node.value for node in ast.parse(PARENT.read_text()).body
                   if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
                   and t.id == '_BD_COEFFICIENT_LITERAL' for t in node.targets)]
        require(len(matches) == 1, 'Unique native cstar literal')
        rows = document['coefficients']
        require(coefficient_bytes(rows) == coefficient_bytes(build['coefficient_literal'])
                == coefficient_bytes(ast.literal_eval(matches[0])), 'cstar byte mismatch')
        _COEFFICIENTS = tuple(tuple(float(x) for x in row) for row in rows)
    return _COEFFICIENTS


def payload_numpy():
    """Explicit lazy numerical call; return independent copies of frozen P."""
    global _PAYLOAD
    authenticate()
    import numpy as np
    if _PAYLOAD is None:
        with np.load(PAYLOAD, allow_pickle=False) as source:
            _PAYLOAD = {key: source[key].copy() for key in source.files}
        require(set(PKEYS).issubset(_PAYLOAD), 'Missing P tensors')
        require(sum(_PAYLOAD[key].size for key in PKEYS) == PARAMETER_COUNT, 'P count')
        require(all(_PAYLOAD[key].dtype == np.float32 for key in PKEYS), 'P precision')
        require(all(np.isfinite(value).all() for value in _PAYLOAD.values()), 'Nonfinite P')
    return {key: value.copy() for key, value in _PAYLOAD.items()}


def _base_module():
    global _BASE
    authenticate()
    if _BASE is None:
        sys.path.insert(0, str(NEW))
        spec = importlib.util.spec_from_file_location('p_coadapt_v1_frozen_bd', BD / 'model.py')
        require(spec is not None and spec.loader is not None, 'Frozen BD loader')
        _BASE = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = _BASE
        spec.loader.exec_module(_BASE)
    return _BASE


def parent_card(rank, device='cpu', weights=None):
    """Unmodified, pinned BirthDirectionCard for the same-backend null oracle."""
    base = _base_module()
    cstar = base.torch.tensor(frozen_coefficients(), device=device, dtype=base.torch.float32)
    return base.BirthDirectionCard(rank, coefficients=cstar, device=device,
                                   weights=weights, audit_selections=True)


def served(value):
    """Native nonnegative convention, followed by the actual FP32 return cast."""
    import torch
    return torch.where(value < 0, torch.zeros_like(value), value).float()


def _card_class():
    global _CARD_CLASS
    if _CARD_CLASS is not None:
        return _CARD_CLASS
    base = _base_module()
    torch = base.torch
    old = '    vv = torch.diagonal(Cpre)\n'
    new = old + '    self._record_variance(l, vv)\n'
    source = base.replace_once(base.source, old, new)
    require(base.replace_once(source, new, old) == base.source, 'Observer source inverse')
    code, count = base.rewrite_clamps(source, str(__file__) + ':step')
    require(count == 2, 'Frozen clamp count changed')
    env = dict(base.env)
    exec(code, env)

    class Card(base.BirthDirectionCard):
        def __init__(self, rank, P=None, device='cpu', audit_selections=True, weights=None):
            require(device in ('cpu', 'mps'), 'Only fixed CPU/MPS implementations')
            original = payload_numpy()
            if P is not None:
                require(set(P) == set(original), 'Supply complete current P mapping')
                for key, value in P.items():
                    require(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                            and value.device.type == device
                            and value.device.index in ((None,) if device == 'cpu' else (0,))
                            and tuple(value.shape) == original[key].shape, 'P type/shape/device: ' + key)
                    require(bool(torch.isfinite(value).all().detach().cpu()), 'Nonfinite P: ' + key)
                    if key not in PKEYS:
                        require(not value.requires_grad, 'Only PKEYS may require gradients')
                        require(value.detach().cpu().numpy().tobytes()
                                == original[key].astype('float32').tobytes(), 'Frozen field changed: ' + key)
                    if key in INACTIVE_KEYS:
                        require(value[0].detach().cpu().numpy().tobytes()
                                == original[key][0].tobytes(), 'Inactive P row changed: ' + key)
            cstar = torch.tensor(frozen_coefficients(), device=device, dtype=torch.float32)
            super().__init__(rank, coefficients=cstar, device=device, weights=weights,
                             P=P, audit_selections=audit_selections)
            if P is not None:
                require(all(self.P0[key] is value for key, value in P.items()),
                        'Supplied P tensor identity/gradient path changed')
            self.selection_cells, self.selection_visits = {}, {}
            self.variance_domain, self.variance_visits = {}, {}
            self._layer, self._selection_slot, self._training = None, 0, False
            self.adapter_source_sha256 = hashlib.sha256(source.encode()).hexdigest()
            self.source_bindings = authenticate()

        def _record_variance(self, layer, variance):
            value = variance.detach().cpu().contiguous()
            record = dict(minimum=float(value.min()), below_floor_count=int((value < 1e-10).sum()),
                          finite=bool(torch.isfinite(value).all()), dtype=str(value.dtype),
                          raw_sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())
            require(record['finite'] and not record['below_floor_count'], 'Prevariance domain failed')
            if layer in self.variance_domain:
                require(self.variance_domain[layer] == record, 'Checkpoint replay changed prevariance bytes')
            self.variance_domain[layer] = record
            self.variance_visits[layer] = self.variance_visits.get(layer, 0) + 1

        def _select(self, score):
            indices, mask = base.centered_source.select(score)
            key = f'{self._layer}:{self._selection_slot}'
            self._selection_slot += 1
            if self.audit_selections:
                ids, mask_cpu = indices.detach().cpu().tolist(), mask.detach().cpu()
                members = mask_cpu.nonzero().flatten().tolist()
                require(sorted(ids) == members and len(ids) == len(set(ids)), 'Selector membership mismatch')
                record = dict(ordered_ids=ids, membership=members, count=len(ids),
                              mask_sha256=hashlib.sha256(mask_cpu.numpy().tobytes()).hexdigest())
                if key in self.selection_cells:
                    require(self.selection_cells[key] == record, 'Checkpoint replay changed selectors')
                self.selection_cells[key] = record
                self.selection_visits[key] = self.selection_visits.get(key, 0) + 1
            return indices, mask

        def step(self, layer, state, control, P):
            require(P is self.P0 and not bool((control.detach() != 0).any().cpu()),
                    'No payload override or exogenous control')
            self._layer, self._selection_slot = layer, 0
            return env['step'](self, layer, state, control, P)

        def run(self, P=None, controls=None, training=False, capture=False):
            require(P is None and controls is None, 'Supply P once at construction; no controls')
            self.selection_cells.clear(); self.selection_visits.clear()
            self.variance_domain.clear(); self.variance_visits.clear()
            self._training = bool(training)
            return super().run(training=training, capture=capture)

        def diagnostics(self):
            return dict(frozen_cstar=True, alltruth_accessed=False, inverse_source_byte_exact=True,
                        allocated_parameters=PARAMETER_COUNT, active_parameters=ACTIVE_COUNT,
                        adapter_source_sha256=self.adapter_source_sha256,
                        variance_domain={str(k): dict(v) for k, v in self.variance_domain.items()},
                        variance_visits={str(k): v for k, v in self.variance_visits.items()},
                        selection_cells={k: dict(v) for k, v in self.selection_cells.items()},
                        selection_visits=dict(self.selection_visits), training=self._training)

    _CARD_CLASS = Card
    return Card


class PCard:
    def __new__(cls, rank, P=None, device='cpu', audit_selections=True, weights=None):
        return _card_class()(rank, P=P, device=device, audit_selections=audit_selections, weights=weights)


def check_domain(card):
    records = card.diagnostics()['variance_domain']
    require(set(records) == {str(l) for l in range(16)}, 'All16 prevariance domains required')
    require(all(x['finite'] and x['minimum'] >= 1e-10 and x['below_floor_count'] == 0
                for x in records.values()), 'Qualified variance domain')
    return records


def check_replay(card):
    """Check every observed replay, allowing non-reentrant early-stop semantics.

    Layer0 has no P dependence and need not be recomputed in backward. Never
    manufacture a second visit. At least one actual variance and selector
    replay is required after a checkpointed training/VJP call.
    """
    result = card.diagnostics()
    check_domain(card)
    expected = {f'{l}:{s}' for l in range(16) for s in range(2 if l == 15 else 3)}
    require(card.audit_selections and set(result['selection_cells']) == expected, 'All47 selector cells required')
    if card._training:
        require(max(result['variance_visits'].values()) >= 2
                and max(result['selection_visits'].values()) >= 2, 'No observed checkpoint replay')
    return result
