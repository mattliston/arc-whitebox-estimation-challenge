"""Inert, full-feedback shared384 extension of the existing COMP proposal card.

This is an offline proposal model, not a participant or a native-gradient
certificate. Numeric imports occur only through explicit runtime/card calls.
There are no targets, optimizers, or parameter updates in this module.
"""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
JOINT = NEW / 'current_comp_joint_p_node_v1'
CANDIDATE = NEW / 'current_comp_shared_gain_slice_v1/build_v1/estimator.py'
PINS = {
    JOINT / 'model.py': 'cb49d7d598a76e430f388e6f3dc84b6b79d89eaf8a0e12cd3c19f8caca61cc86',
    CANDIDATE: 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea',
}
_JOINT = _OP = _CLASS = _BINDINGS = _PROOF = None


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def authenticate(shared_sha256):
    """Bind reviewed source and prove the exact current c2 head identities."""
    global _JOINT, _OP, _BINDINGS
    if _BINDINGS is not None:
        require(_BINDINGS[str(HERE / 'shared_operator.py')] == shared_sha256,
                'Same explicitly reviewed shared operator')
        require(all(sha(p) == h for p, h in _BINDINGS.items()), 'Source bytes unchanged')
        return dict(_BINDINGS)
    require(not {'numpy', 'torch', 'flopscope', 'threadpoolctl'} & set(sys.modules),
            'Source admission before numerical imports')
    pins = {str(p): h for p, h in PINS.items()}
    pins[str(HERE / 'shared_operator.py')] = shared_sha256
    require(all(sha(p) == h for p, h in pins.items()), 'Reviewed source pins')
    _JOINT = module(JOINT / 'model.py', 'shared384_original_joint_model')
    pins.update(_JOINT.authenticate())
    wanted = {'_GNN_LITERAL', '_BDNH_A_LITERAL', '_BD_COEFFICIENT_LITERAL'}
    literals = {}
    for node in ast.parse(CANDIDATE.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in wanted:
                require(target.id not in literals, 'Unique c2 parameter literal')
                literals[target.id] = ast.literal_eval(node.value)
    require(set(literals) == wanted and literals == _JOINT._LITERALS,
            'Every current c2 P/NODE/cstar literal equals admitted COMP initialization')
    _OP = module(HERE / 'shared_operator.py', 'shared384_pure_operator')
    for path in (Path(__file__).resolve(), HERE / 'SELECTION.md'):
        pins[str(path)] = sha(path)
    require(not {'numpy', 'torch', 'flopscope', 'threadpoolctl'} & set(sys.modules),
            'Complete admission remains inert')
    _BINDINGS = pins
    return dict(pins)


def runtime():
    require(_BINDINGS is not None, 'Explicit reviewed admission first')
    return _JOINT.runtime()


def frozen(device, *, trainable_P=False):
    """Only the existing P surface may be fitted; NODE A and cstar stay fixed."""
    runtime()
    return _JOINT.frozen(device, trainable_P=trainable_P, trainable_A=False)


def card_class():
    global _CLASS, _PROOF
    if _CLASS is not None:
        return _CLASS
    base = runtime()
    parent = _JOINT.card_class()
    source = base.source
    for before, after in _JOINT.CHANGES:
        source = base.replace_once(source, before, after)
    original_joint = source
    changes = (
        ('    m, C, p, lam, star, b, previous_alive, radial, hstate = state\n',
         '    m, C, p, lam, star, b, previous_alive, radial, hstate = state[:9]\n'
         '    Q, F, core, stops, previous_indices = state[9:]\n'),
        ('    pp = w.T@p\n',
         '    pp, U, F, core, old_columns = shared_push(torch, l, w, p, lam,\n'
         '        previous_indices, Q, F, core, stops, rank=384)\n'
         '    self._record_shared(l, pp, U, F, core, old_columns, stops)\n'),
        ('        left, right = pp[response_indices], pp[indices]\n'
         '        subplane = ((left*left)@(right*lam).T).to(dt)\n',
         '        subplane = shared_read(torch, pp, lam, response_indices, indices,\n'
         '            U, F, old_columns).to(dt)\n'),
        ('        return (m, C, p, lam, star, b, alive, new_radial, hs)\n',
         '        return (m, C, p, lam, star, b, alive, new_radial, hs,\n'
         '                Q, F, core, stops, previous_indices)\n'),
        ('    return (m, C, p, lam, star, b, alive, new_radial, hs)\n',
         '    if old_columns:\n'
         '        Q = alive.float()[:, None] * (g.float()[:, None] * U)\n'
         '    stops = stops + (int(lam.numel()),)\n'
         '    return (m, C, p, lam, star, b, alive, new_radial, hs,\n'
         '            Q, F, core, stops, indices)\n'),
    )
    for before, after in changes:
        source = base.replace_once(source, before, after)
    inverse = source
    for before, after in reversed(changes):
        inverse = base.replace_once(inverse, after, before)
    require(inverse == original_joint, 'Only the five explicit shared-state splices')
    code, count = base.rewrite_clamps(source, str(__file__) + ':shared_step')
    require(count == 2, 'Original physical clamp ownership')
    env = dict(base.env, shared_push=_OP.push, shared_read=_OP.read,
               node_coefficients=_JOINT._NODE_FN)
    exec(code, env)
    _PROOF = dict(shared_rank=384, full_feedback=True, detached_basis=False,
        mutable_carrier_cache=False, frozen_NODE_A=True, frozen_cstar=True,
        joint_source_sha256=hashlib.sha256(original_joint.encode()).hexdigest(),
        shared_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        five_splice_inverse_exact=True, physical_clamp_count=count,
        native_gradient_certified=False, dense_young_real_algebra_only=True)

    class SharedCard(parent):
        def __init__(self, rank, *, P, A, device='cpu', weights=None):
            super().__init__(rank, P=P, A=A, device=device, weights=weights)
            require(self.n == 1024 and self.depth == 16, 'Phase2 production geometry only')
            require(not self.A.requires_grad, 'NODE A is frozen in this question')
            self.shared_cells = {}
            self.shared_visits = {}

        def initial(self):
            state = super().initial()
            torch = base.torch
            return state + (torch.empty((1024, 0), dtype=torch.float32, device=self.device),
                torch.empty((0, 0), dtype=torch.float32, device=self.device),
                torch.empty((0, 0), dtype=torch.float32, device=self.device), (),
                torch.arange(1024, device=self.device))

        def _record_shared(self, layer, pp, U, F, core, old_columns, stops):
            require(all(v.dtype == base.torch.float32 for v in (pp, F, core)) and
                    (U is None or U.dtype == base.torch.float32), 'Production FP32 carrier state')
            require(len(stops) == layer and all(b > a for a, b in zip((0,) + stops, stops)),
                    'Complete chronological signed cohorts')
            expected = 0 if layer < 5 else stops[min(layer - 5, 9)]
            require(old_columns == expected and pp.shape[1] == (stops[-1] if stops else 0),
                    'Exactly ten original shared joins, no terminal join')
            record = dict(columns=int(pp.shape[1]), old_columns=old_columns,
                F_shape=list(F.shape), core_shape=list(core.shape), stops=list(stops),
                U_shape=None if U is None else list(U.shape))
            if layer in self.shared_cells:
                require(record == self.shared_cells[layer], 'Shared checkpoint geometry replay')
            self.shared_cells[layer] = record
            self.shared_visits[layer] = self.shared_visits.get(layer, 0) + 1

        def step(self, layer, state, control, P):
            require(P is self.P0 and not bool((control.detach() != 0).any().cpu()),
                    'No exogenous state or controls')
            self._layer, self._selection_slot = layer, 0
            return env['step'](self, layer, state, control, P)

        def run(self, P=None, controls=None, training=False, capture=False):
            self.shared_cells.clear()
            self.shared_visits.clear()
            return super().run(P=P, controls=controls, training=training, capture=capture)

        def diagnostics(self):
            result = super().diagnostics()
            result.update(shared_cells={str(k): v for k, v in self.shared_cells.items()},
                          shared_visits={str(k): v for k, v in self.shared_visits.items()},
                          shared_source_proof=dict(_PROOF))
            return result

    _CLASS = SharedCard
    return _CLASS


def card(rank, *, P, A, device='cpu', weights=None):
    return card_class()(rank, P=P, A=A, device=device, weights=weights)
