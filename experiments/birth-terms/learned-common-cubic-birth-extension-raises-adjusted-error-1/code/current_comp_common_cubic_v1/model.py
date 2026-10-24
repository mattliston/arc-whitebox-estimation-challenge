"""Inert exact-COMP adapter: one shared late common-cubic weight field."""
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
JOINT = NEW/'current_comp_joint_p_node_v1'
BD = NEW/'birth_direction_fit_v1'
COMP_SHA = '2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a'
PKEYS = ('msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b',
         'enc4_w', 'enc4_b', 'head', 'head_b', 'gain')
INACTIVE_KEYS = ('head', 'head_b', 'gain')
PARAMETER_COUNT = ACTIVE_COUNT = Q_PARAMETER_COUNT = 65
ACTED_LAYERS = (10, 11, 12, 13, 14)
PINS = {
    JOINT/'model.py': 'cb49d7d598a76e430f388e6f3dc84b6b79d89eaf8a0e12cd3c19f8caca61cc86',
    JOINT/'PROTOCOL.md': '330f744e26f5672748ce94fc20f6e0f617358f83883d6a6c0c4b25c185d11830',
    BD/'model.py': 'cea1ea473db821cea6b4b9d34aeabf06f43ded8c5d77ee4037dd3f3d269df613',
    HERE/'DECISION.md': 'd90cec782ef7631acc227b1bb857f92d71284920521341764c3d182b87368d80',
}
COMMON_CHANGES = (
    ('    born = born+plane_increment\n',
     '    born = born+plane_increment\n'
     '    lambda_plus, lambda_minus = a.float(), -a.float()\n'
     '    if l in (10, 11, 12, 13, 14):\n'
     '        lambda_plus, lambda_minus, common_increment = self.common_birth_write(\n'
     '            l, hs, stored_base, a, t, s, alive, selected, births, dt)\n'
     '        born = born+common_increment\n'),
    ('    lam = torch.cat((lam, a[selected].float(), -a[selected].float()))\n',
     '    lam = torch.cat((lam, lambda_plus[selected], lambda_minus[selected]))\n'),
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
    changed = source
    for before, after in changes:
        require(changed.count(before) == 1, 'Unique common-cubic source span')
        changed = changed.replace(before, after)
    inverse = changed
    for before, after in reversed(changes):
        require(inverse.count(after) == 1, 'Unique common-cubic inverse span')
        inverse = inverse.replace(after, before)
    require(inverse == source, 'Original numerical source restored byte-exactly')
    return changed


def step_sources(original, joint_changes):
    """Only strings: reversible late write and actual lambda child plumbing."""
    joint = adapt(original, joint_changes)
    changed = adapt(joint, COMMON_CHANGES)
    require(changed.count('    born = born+plane_increment\n') == 1
        and changed.count('birth_action(bp,Cpre,k21,aux,node_c,t,s,a,alive,births)') == 1,
        'Original BD action and its original plane increment remain once')
    return dict(original=original, joint=joint, modified=changed)


def authenticate():
    global _JOINT, _CO, _BINDINGS
    if _BINDINGS is not None:
        require(all(sha(path) == digest for path, digest in _BINDINGS.items()),
            'Common-cubic source owners unchanged')
        return dict(_BINDINGS)
    require(not any(key in sys.modules for key in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
        'Explicit source admission before numerical imports')
    pins = {str(path.resolve()): digest for path, digest in PINS.items()}
    require(all(sha(path) == digest for path, digest in pins.items()),
        'Frozen exact-COMP source and prospective decision')
    _JOINT = module(JOINT/'model.py', 'common_cubic_exact_joint_owner')
    pins.update(_JOINT.authenticate())
    require(_JOINT.COMP_SHA == COMP_SHA and _JOINT.PKEYS == PKEYS
        and _JOINT.INACTIVE_KEYS == INACTIVE_KEYS, 'Unchanged current-COMP parameter surface')
    for path in (Path(__file__).resolve(), HERE/'MODEL_PROTOCOL.md'):
        pins[str(path)] = sha(path)
    _CO, _BINDINGS = _JOINT._CO, pins
    return dict(pins)


def runtime():
    global _BASE
    require(_JOINT is not None and _BINDINGS is not None, 'Authenticate before explicit numerical runtime')
    if _BASE is None:
        _BASE = _JOINT.runtime()
    return _BASE


def frozen(device, *, trainable_Q=True):
    base = runtime()
    P, A = _JOINT.frozen(device, trainable_P=False, trainable_A=False)
    Q = base.torch.zeros((65,), dtype=base.torch.float32, device=device,
                         requires_grad=bool(trainable_Q))
    return P, A, Q


def common_response(H, Q):
    """Original NODE FP32 normalization, then the single new affine field."""
    torch = runtime().torch
    require(H.ndim == 2 and H.shape[1] == 64 and Q.shape == (65,)
        and H.dtype == Q.dtype == torch.float32 and H.device == Q.device,
        'Original FP32 hidden channels and same-device common Q')
    centered = H-H.mean(dim=0)
    scale = ((centered*centered).mean(dim=0)+1e-6).sqrt()
    standardized = (centered/scale).clamp(-8, 8)
    z = standardized @ Q[:64] + Q[64]
    return z.tanh()


def common_weights(a32, t32, s32, response):
    """Live stored FP32 A1-preserving formula; actual rounding is retained."""
    torch = runtime().torch
    operands = (a32, t32, s32, response)
    require(a32.ndim == 1 and all(x.shape == a32.shape and x.dtype == torch.float32
        and x.device == a32.device for x in operands), 'Common weight FP32 operands')
    denominator = torch.maximum(torch.maximum(t32.abs(), s32.abs()), torch.full_like(t32, 1e-20))
    q = ((.25*a32)*response)/denominator
    plus = a32+((-q)*s32)
    minus = (-a32)+(q*t32)
    return plus, minus


def common_plane(B32, a32, t32, s32, plus32, minus32, selected, births, physical_dtype):
    """Full actual stored-pair plane increment, including its nonzero diagonal."""
    base = runtime(); torch = base.torch
    n = int(a32.shape[0])
    require(B32.shape == (n, n) and B32.dtype == torch.float32
        and all(x.shape == (n,) and x.dtype == torch.float32 and x.device == B32.device
            for x in (a32, t32, s32, plus32, minus32))
        and births.shape == (n,) and births.device == B32.device
        and selected.ndim == 1 and selected.dtype == torch.int64 and selected.device == B32.device
        and physical_dtype in (torch.float32, torch.float64), 'Typed complete stored-plane operands')
    B = B32.to(physical_dtype)
    a, t, s = (value.to(physical_dtype) for value in (a32, t32, s32))
    plus, minus = plus32.to(physical_dtype), minus32.to(physical_dtype)
    mask = births.to(physical_dtype)
    dp, dm = plus-a, minus+a
    t2, s2 = t*t, s*s
    d0 = (dp+dm)*mask
    d1 = (dp*t+dm*s)*mask
    d2 = (dp*t2+dm*s2)*mask
    d3 = (dp*(t2*t)+dm*(s2*s))*mask
    square = B*B
    Bs, square_s, d0s = B[:, selected], square[:, selected], d0[selected]
    increment = (square_s*d0s[None, :]) @ Bs.T
    increment = increment+square*d1[None, :]
    increment = increment+d2[:, None]*B.T
    diag = ((square_s*Bs)*d0s[None, :]).sum(dim=1)+d3
    return base.diagonal(increment, diag)


def card_class():
    global _CLASS, _PROOF
    if _CLASS is not None:
        return _CLASS
    base = runtime(); parent = _JOINT.card_class()
    require(Path(base.__file__).resolve() == (BD/'model.py').resolve()
        and sha(BD/'model.py') == PINS[BD/'model.py'], 'Original BD numerical function owner')
    parts = step_sources(base.source, _JOINT.CHANGES)
    source = parts['modified']
    code, count = base.rewrite_clamps(source, str(__file__)+':step')
    require(count == 2 and source.count('    self._record_variance(l, vv)\n') == 1,
        'Original two physical clamp rewrites and sole variance observer')
    env = dict(base.env); exec(code, env)
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    _PROOF = dict(step_inverse_byte_exact=True, joint_step_inverse_byte_exact=True,
        original_step_sha256=digest(parts['original']), joint_step_sha256=digest(parts['joint']),
        modified_step_sha256=digest(source), physical_clamp_count=count, variance_hook_count=1,
        original_NODE_function_unchanged=True, original_BD_function_unchanged=True,
        original_BD_plane_increment_unchanged=True, original_born_defect_preserved=True,
        original_radial_and_trace_unchanged=True, original_factor_columns_unchanged=True,
        generic_children_use_actual_plus_minus=True, all_four_stored_increments_used=True,
        common_plane_diagonal_restored=True, no_new_CP_columns=True, acted_layers=list(ACTED_LAYERS),
        common_parameter_count=65, no_zero_parameter_bypass=True, no_native_gradient_claim=True)

    class CommonCubicCard(parent):
        def __init__(self, rank, *, P, A, Q, device='mps', weights=None):
            require(P is not None and A is not None and Q is not None, 'Explicit frozen COMP P/A and common Q')
            super().__init__(rank, P=P, A=A, device=device, weights=weights)
            torch = base.torch
            require(isinstance(Q, torch.Tensor) and Q.shape == (65,) and Q.dtype == torch.float32
                and Q.device == self.w.device and Q.is_leaf
                and bool(torch.isfinite(Q).all().detach().cpu()), 'Actual finite same-device common Q leaf')
            require(not A.requires_grad and A.grad is None
                and all(not value.requires_grad and value.grad is None for value in P.values()),
                'Only the new Q may require gradients')
            for key, value in P.items():
                require(self.P0[key] is value and value.detach().cpu().contiguous().numpy().tobytes()
                    == base.np.asarray(_JOINT._LITERALS['_GNN_LITERAL'][key], dtype=base.np.float32).tobytes(),
                    'Exact frozen current-COMP P identity/bytes: '+key)
            require(self.A is A and A.detach().cpu().contiguous().numpy().tobytes()
                == base.np.asarray(_JOINT._LITERALS['_BDNH_A_LITERAL'], dtype=base.np.float32).tobytes(),
                'Exact frozen current-COMP NODE A identity/bytes')
            require(not self.coefficients.requires_grad and self.coefficients.grad is None
                and self.coefficients.detach().cpu().contiguous().numpy().tobytes()
                    == _CO.coefficient_bytes(_JOINT._LITERALS['_BD_COEFFICIENT_LITERAL']),
                'Exact frozen original cstar bytes')
            self.Q = Q
            self.common_head_calls = {}
            self.source_bindings.update(_BINDINGS)
            self.adapter_source_sha256 = _PROOF['modified_step_sha256']

        def common_birth_write(self, layer, H, B32, a, t, s, alive, selected, births, dt):
            require(layer in ACTED_LAYERS and bool(base.torch.isfinite(H).all().detach().cpu()),
                'Only finite current returned H at fixed birth layers10..14')
            a32, t32, s32 = a.float(), (t*alive).float(), (s*alive).float()
            response = common_response(H, self.Q)
            plus, minus = common_weights(a32, t32, s32, response)
            increment = common_plane(B32, a32, t32, s32, plus, minus, selected, births, dt)
            require(all(bool(base.torch.isfinite(value).all().detach().cpu())
                for value in (response, plus, minus, increment)), 'Finite common-cubic physical action')
            self.common_head_calls[layer] = self.common_head_calls.get(layer, 0)+1
            return plus, minus, increment

        def step(self, layer, state, control, P):
            require(P is self.P0 and not bool((control.detach() != 0).any().cpu()),
                'No P replacement, teacher forcing or exogenous mean control')
            self._layer, self._selection_slot = layer, 0
            return env['step'](self, layer, state, control, P)

        def run(self, P=None, controls=None, training=False, capture=False):
            self.common_head_calls.clear()
            return super().run(P=P, controls=controls, training=training, capture=capture)

        def diagnostics(self):
            result = super().diagnostics()
            result.update(allocated_parameters=65, active_parameters=65, common_parameter_count=65,
                frozen_parent_parameter_count=45456, frozen_P=True, frozen_NODE_A=True, frozen_cstar=True,
                live_P_and_A_allowed=False, endogenous_H_not_detached=True,
                common_head_calls={str(key): value for key, value in self.common_head_calls.items()},
                common_cubic_source_proof=dict(_PROOF), no_native_gradient_claim=True)
            return result

    _CLASS = CommonCubicCard
    return _CLASS


def card(rank, *, P, A, Q, device='mps', weights=None):
    return card_class()(rank, P=P, A=A, Q=Q, device=device, weights=weights)


def source_proof():
    require(_PROOF is not None, 'Construct the explicit runtime card before its compiled-source proof')
    return dict(_PROOF)
