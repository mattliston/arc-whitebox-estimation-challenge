"""Inert exact-COMP adapter: independent C-only diagonal congruence and live P."""
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
JOINT = NEW/'current_comp_joint_p_node_v1'
COMP_SHA = '2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a'
PKEYS = ('msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b',
         'enc4_w', 'enc4_b', 'head', 'head_b', 'gain')
INACTIVE_KEYS = ('head', 'head_b', 'gain')
P_PARAMETER_COUNT, Q_PARAMETER_COUNT = 45200, 910
PARAMETER_COUNT, ACTIVE_COUNT = 46110, 46044
PINS = {
    JOINT/'model.py': 'cb49d7d598a76e430f388e6f3dc84b6b79d89eaf8a0e12cd3c19f8caca61cc86',
    JOINT/'PROTOCOL.md': '330f744e26f5672748ce94fc20f6e0f617358f83883d6a6c0c4b25c185d11830',
    HERE/'DECISION.md': '814ac1f31121db92f3165ecfa0cb3253e8b66617e3e62d17316f2f288e71be98',
}
COVARIANCE_CHANGE = (
    '    C = diagonal(C*live, torch.diagonal(C))\n',
    '    C = diagonal(C*live, torch.diagonal(C))\n'
    '    if l > 0:\n'
    '        C = self.covariance_write(l, C, hs)\n',
)
_JOINT = _CO = _BASE = _CLASS = _BINDINGS = _PROOF = None


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def adapt(source, changes):
    result = source
    for before, after in changes:
        require(result.count(before) == 1, 'Unique covariance source span')
        result = result.replace(before, after)
    inverse = result
    for before, after in reversed(changes):
        require(inverse.count(after) == 1, 'Unique covariance inverse span')
        inverse = inverse.replace(after, before)
    require(inverse == source, 'Original numerical source restored byte-exactly')
    return result


def authenticate():
    global _JOINT, _CO, _BINDINGS
    if _BINDINGS is not None:
        require(all(sha(path) == digest for path, digest in _BINDINGS.items()), 'Covariance source owners unchanged')
        return dict(_BINDINGS)
    require(not any(key in sys.modules for key in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
        'Explicit source admission before numerical imports')
    pins = {str(path.resolve()): digest for path, digest in PINS.items()}
    require(all(sha(path) == digest for path, digest in pins.items()), 'Frozen exact-COMP/decision authority')
    _JOINT = module(JOINT/'model.py', 'covariance_exact_joint_source_owner')
    pins.update(_JOINT.authenticate())
    require(_JOINT.COMP_SHA == COMP_SHA and _JOINT.PKEYS == PKEYS
        and _JOINT.INACTIVE_KEYS == INACTIVE_KEYS, 'Exact current parent and unchanged P surface')
    for path in (Path(__file__).resolve(), HERE/'MODEL_PROTOCOL.md'):
        pins[str(path)] = sha(path)
    _CO, _BINDINGS = _JOINT._CO, pins
    return dict(pins)


def runtime():
    global _BASE
    require(_JOINT is not None and _BINDINGS is not None, 'Authenticate before numerical runtime')
    if _BASE is None:
        _BASE = _JOINT.runtime()
    return _BASE


def frozen(device, *, trainable_P=False, trainable_Q=True):
    base = runtime()
    P, A = _JOINT.frozen(device, trainable_P=trainable_P, trainable_A=False)
    Q = base.torch.zeros((14, 65), dtype=base.torch.float32, device=device,
                         requires_grad=bool(trainable_Q))
    return P, A, Q


def covariance_scale(H, q):
    """One original-H FP32 affine head; no new normalization or GNN pass."""
    torch = runtime().torch
    require(H.ndim == 2 and H.shape[1] == 64 and q.shape == (65,)
        and H.dtype == q.dtype == torch.float32 and H.device == q.device,
        'Same-device original FP32 hidden channels and actual covariance row')
    z = H @ q[:64] + q[64]
    return torch.exp(.005 * torch.tanh(z))


def covariance_action(C, H, q):
    """C-only compensator, not an all-cumulant affine transformation."""
    scale = covariance_scale(H, q).to(C.dtype)
    return (C * scale[:, None]) * scale[None, :]


def card_class():
    global _CLASS, _PROOF
    if _CLASS is not None:
        return _CLASS
    base = runtime(); parent = _JOINT.card_class()
    joint_source = adapt(base.source, _JOINT.CHANGES)
    source = adapt(joint_source, (COVARIANCE_CHANGE,))
    code, count = base.rewrite_clamps(source, str(__file__)+':step')
    require(count == 2 and source.count('    self._record_variance(l, vv)\n') == 1,
        'Original two clamp rewrites and sole original variance observer')
    env = dict(base.env); exec(code, env)
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    _PROOF = dict(step_inverse_byte_exact=True, joint_step_inverse_byte_exact=True,
        original_step_sha256=digest(base.source), joint_step_sha256=digest(joint_source),
        modified_step_sha256=digest(source), variance_hook_count=1, physical_clamp_count=count,
        only_post_mask_C_write_added=True, covariance_parameter_count=910,
        original_NODE_function_unchanged=True, original_birth_action_unchanged=True,
        original_radial_ownership_unchanged=True, no_new_CP_columns=True,
        no_zero_parameter_bypass=True, no_covariance_write_at_layers0_or15=True,
        no_receiver_geometry=True, no_native_gradient_claim=True)

    class CovarianceCard(parent):
        def __init__(self, rank, *, P, A, Q, device='mps', weights=None):
            require(P is not None and A is not None and Q is not None, 'Explicit current/proposed P, fixed A and Q')
            super().__init__(rank, P=P, A=A, device=device, weights=weights)
            torch = base.torch
            require(isinstance(Q, torch.Tensor) and Q.shape == (14, 65) and Q.dtype == torch.float32
                and Q.device == self.w.device and Q.is_leaf
                and bool(torch.isfinite(Q).all().detach().cpu()), 'Actual finite same-device Q leaf')
            require(not A.requires_grad and A.grad is None
                and A.detach().cpu().contiguous().numpy().tobytes()
                    == base.np.asarray(_JOINT._LITERALS['_BDNH_A_LITERAL'], dtype=base.np.float32).tobytes(),
                'Exact retained frozen COMP NODE A')
            require(all(self.P0[key] is value and (key not in PKEYS or value.is_leaf)
                for key, value in P.items()), 'Actual supplied P leaves; no copied checkpoint parameters')
            require(not self.coefficients.requires_grad, 'Original cstar stays frozen')
            self.Q = Q
            self.covariance_head_calls = {}
            self.source_bindings.update(_BINDINGS)
            self.adapter_source_sha256 = _PROOF['modified_step_sha256']

        def covariance_write(self, layer, C, H):
            require(1 <= layer <= 14 and bool(base.torch.isfinite(H).all().detach().cpu()),
                'Only current finite returned H at layers1..14')
            changed = covariance_action(C, H, self.Q[layer-1])
            require(bool(base.torch.isfinite(changed).all().detach().cpu()), 'Finite compensated physical covariance')
            self.covariance_head_calls[layer] = self.covariance_head_calls.get(layer, 0)+1
            return changed

        def step(self, layer, state, control, P):
            require(P is self.P0 and not bool((control.detach() != 0).any().cpu()), 'No replacement P or mean control')
            self._layer, self._selection_slot = layer, 0
            return env['step'](self, layer, state, control, P)

        def run(self, P=None, controls=None, training=False, capture=False):
            self.covariance_head_calls.clear()
            return super().run(P=P, controls=controls, training=training, capture=capture)

        def diagnostics(self):
            result = super().diagnostics()
            result.update(allocated_parameters=PARAMETER_COUNT, active_parameters=ACTIVE_COUNT,
                covariance_parameter_count=Q_PARAMETER_COUNT, frozen_NODE_A=True, frozen_cstar=True,
                live_P_and_A_allowed=False, live_P_and_Q_allowed=True, endogenous_H_not_detached=True,
                covariance_head_calls={str(key): value for key, value in self.covariance_head_calls.items()},
                covariance_source_proof=dict(_PROOF))
            return result

    _CLASS = CovarianceCard
    return _CLASS


def card(rank, *, P, A, Q, device='mps', weights=None):
    return card_class()(rank, P=P, A=A, Q=Q, device=device, weights=weights)


def source_proof():
    require(_PROOF is not None, 'Construct the explicit card before its source proof')
    return dict(_PROOF)
