"""Inert current-COMP adapter: physical birth directions modulated by ROWS."""
import ast
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
PARAMETER_COUNT = ACTIVE_COUNT = 256
PINS = {
    JOINT/'model.py': 'cb49d7d598a76e430f388e6f3dc84b6b79d89eaf8a0e12cd3c19f8caca61cc86',
    JOINT/'PROTOCOL.md': '330f744e26f5672748ce94fc20f6e0f617358f83883d6a6c0c4b25c185d11830',
    BD/'model.py': 'cea1ea473db821cea6b4b9d34aeabf06f43ded8c5d77ee4037dd3f3d269df613',
}
BASIS_CHANGES = (
    ('def residualized_basis(B, C, K21, aux):',
     'def residualized_basis(B, C, K21, aux, row_modulation):'),
    ('    for item in raw:\n', '    for index, item in enumerate(raw):\n'),
    ('        item = diagonal(item, zero)\n',
     '        item = diagonal(item, zero)\n'
     '        item = item * row_modulation[index].to(B.dtype)[:, None]\n'),
)
BIRTH_CHANGES = (
    ('def birth_action(B, C, K21, aux, coefficients, t, s, a, alive, births):',
     'def birth_action(B, C, K21, aux, coefficients, t, s, a, alive, births, row_modulation):'),
    ('    bases = residualized_basis(B, C, K21, aux)\n',
     '    bases = residualized_basis(B, C, K21, aux, row_modulation)\n'),
)
_JOINT = _CO = _BASE = _CLASS = _FUNCTIONS = _BINDINGS = _PROOF = None


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
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def extract(source, name):
    found = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(found) == 1, 'Unique original function: '+name)
    node = found[0]
    return ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


def adapt(source, changes):
    changed = source
    for before, after in changes:
        require(changed.count(before) == 1, 'Unique typed source insertion')
        changed = changed.replace(before, after)
    inverse = changed
    for before, after in reversed(changes):
        require(inverse.count(after) == 1, 'Unique typed source inverse')
        inverse = inverse.replace(after, before)
    require(inverse == source, 'Original function restored byte-for-byte')
    return changed


def helper_sources(source):
    """Stdlib source transformation; no numerical imports or array construction."""
    basis, birth = (extract(source, name) for name in ('residualized_basis', 'birth_action'))
    return dict(original_basis=basis, original_birth=birth,
        basis=adapt(basis, BASIS_CHANGES), birth=adapt(birth, BIRTH_CHANGES))


def authenticate():
    global _JOINT, _CO, _BINDINGS
    if _BINDINGS is not None:
        require(all(sha(p) == h for p, h in _BINDINGS.items()), 'Receiver-direction sources unchanged')
        return dict(_BINDINGS)
    require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
        'Explicit source admission before numerical imports')
    pins = {str(p.resolve()): h for p, h in PINS.items()}
    require(all(sha(p) == h for p, h in pins.items()), 'Frozen exact-COMP source lineage')
    _JOINT = module(JOINT/'model.py', 'receiver_directions_exact_joint_owner')
    pins.update(_JOINT.authenticate())
    require(_JOINT.COMP_SHA == COMP_SHA and _JOINT.PKEYS == PKEYS
        and _JOINT.INACTIVE_KEYS == INACTIVE_KEYS, 'Exact unchanged COMP parameter surface')
    for path in (Path(__file__).resolve(), HERE/'MODEL_PROTOCOL.md', HERE/'DECISION.md'):
        pins[str(path)] = sha(path)
    _CO, _BINDINGS = _JOINT._CO, pins
    return dict(pins)


def runtime():
    global _BASE
    require(_JOINT is not None and _BINDINGS is not None, 'Authenticate before numerical runtime')
    if _BASE is None:
        _BASE = _JOINT.runtime()
    return _BASE


def frozen(device, trainable_R=True):
    base = runtime()
    P, A = _JOINT.frozen(device, trainable_P=False, trainable_A=False)
    R = base.torch.zeros((64, 4), dtype=base.torch.float32, device=device,
                         requires_grad=bool(trainable_R))
    return P, A, R


def row_modulation(H, R):
    """Exact original FP32 NODE normalization, now read along receiver rows."""
    torch = runtime().torch
    n = int(H.shape[0])
    require(H.shape == (n, 64) and R.shape == (64, 4)
        and H.dtype == R.dtype == torch.float32 and H.device == R.device,
        'Same-device original FP32 H and shared64x4 receiver leaf')
    centered = H - H.mean(dim=0)
    scale = ((centered * centered).mean(dim=0) + 1e-6).sqrt()
    standardized = (centered / scale).clamp(-8, 8)
    return (1 + (standardized @ R).tanh()).T


def numeric_functions():
    global _FUNCTIONS
    base = runtime()
    if _FUNCTIONS is None:
        require(Path(base.__file__).resolve() == (BD/'model.py').resolve()
            and sha(BD/'model.py') == PINS[BD/'model.py'], 'Exact original four-direction math owner')
        parts = helper_sources((BD/'model.py').read_text())
        env = dict(base.__dict__)
        exec(compile(parts['basis']+'\n'+parts['birth'], str(__file__)+':typed_row_helpers', 'exec'), env)
        _FUNCTIONS = dict(basis=env['residualized_basis'], birth=env['birth_action'], sources=parts)
    return _FUNCTIONS


def card_class():
    global _CLASS, _PROOF
    if _CLASS is not None:
        return _CLASS
    base = runtime()
    parent = _JOINT.card_class()
    functions = numeric_functions()
    changes = _JOINT.CHANGES[:2] + ((
        _JOINT.CHANGES[2][0],
        'self.receiver_birth_action(l,hs,bp,Cpre,k21,aux,node_c,t,s,a,alive,births)'),)
    source = adapt(base.source, changes)
    code, count = base.rewrite_clamps(source, str(__file__)+':step')
    require(count == 2 and source.count('    self._record_variance(l, vv)\n') == 1,
        'Original two physical clamps and one original variance observer')
    env = dict(base.env)
    exec(code, env)
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    _PROOF = dict(step_inverse_byte_exact=True, basis_inverse_byte_exact=True, birth_inverse_byte_exact=True,
        original_step_sha256=digest(base.source), modified_step_sha256=digest(source),
        original_basis_sha256=digest(functions['sources']['original_basis']),
        modified_basis_sha256=digest(functions['sources']['basis']),
        original_birth_sha256=digest(functions['sources']['original_birth']),
        modified_birth_sha256=digest(functions['sources']['birth']),
        physical_clamp_count=count, variance_hook_count=1,
        row_multiply_after_original_zero_before_projection=True,
        single_original_stored_pair_repair=True, original_NODE_function_unchanged=True,
        no_zero_parameter_bypass=True, no_receiver_bilinear_field=True, no_new_CP_columns=True)

    class ReceiverDirectionsCard(parent):
        def __init__(self, rank, *, P, A, R, device='mps', weights=None):
            require(P is not None and A is not None and R is not None, 'Explicit COMP P/A and receiver R')
            super().__init__(rank, P=P, A=A, device=device, weights=weights)
            torch = base.torch
            require(isinstance(R, torch.Tensor) and R.shape == (64, 4) and R.dtype == torch.float32
                and R.device == self.w.device and R.is_leaf
                and bool(torch.isfinite(R).all().detach().cpu()), 'Actual finite same-device receiver leaf')
            require(not A.requires_grad and all(not value.requires_grad for value in P.values()),
                'All parent P and NODE A parameters frozen')
            for key, value in P.items():
                require(value.detach().cpu().contiguous().numpy().tobytes()
                    == base.np.asarray(_JOINT._LITERALS['_GNN_LITERAL'][key], dtype=base.np.float32).tobytes(),
                    'Exact frozen current-COMP P: '+key)
            require(A.detach().cpu().contiguous().numpy().tobytes()
                == base.np.asarray(_JOINT._LITERALS['_BDNH_A_LITERAL'], dtype=base.np.float32).tobytes(),
                'Exact retained COMP NODE A, not a new fitted head')
            self.R = R
            self.receiver_head_calls = {}
            self.source_bindings.update(_BINDINGS)
            self.adapter_source_sha256 = _PROOF['modified_step_sha256']

        def receiver_birth_action(self, layer, H, B, C, K21, aux, coefficients, t, s, a, alive, births):
            if layer == 0:
                return base.birth_action(B, C, K21, aux, coefficients, t, s, a, alive, births)
            require(1 <= layer <= 14, 'Only original nonterminal birth layers1..14')
            rows = row_modulation(H, self.R)
            require(bool(base.torch.isfinite(rows).all().detach().cpu()), 'Finite physical row modulation')
            self.receiver_head_calls[layer] = self.receiver_head_calls.get(layer, 0)+1
            return functions['birth'](B, C, K21, aux, coefficients, t, s, a, alive, births, rows)

        def step(self, layer, state, control, P):
            require(P is self.P0 and not bool((control.detach() != 0).any().cpu()),
                'No P override or exogenous mean control')
            self._layer, self._selection_slot = layer, 0
            return env['step'](self, layer, state, control, P)

        def run(self, P=None, controls=None, training=False, capture=False):
            self.receiver_head_calls.clear()
            return super().run(P=P, controls=controls, training=training, capture=capture)

        def diagnostics(self):
            result = super().diagnostics()
            result.update(allocated_parameters=256, active_parameters=256, frozen_parent_parameter_count=45456,
                frozen_P=True, frozen_NODE_A=True, frozen_cstar=True, live_P_and_A_allowed=False,
                receiver_head_calls={str(k): v for k, v in self.receiver_head_calls.items()},
                receiver_direction_source_proof=dict(_PROOF), single_original_stored_pair_repair=True,
                endogenous_H_not_detached=True)
            return result

    _CLASS = ReceiverDirectionsCard
    return _CLASS


def card(rank, *, P, A, R, device='mps', weights=None):
    return card_class()(rank, P=P, A=A, R=R, device=device, weights=weights)


def source_proof():
    require(_PROOF is not None, 'Construct the explicit card before requesting step proof')
    return dict(_PROOF)
