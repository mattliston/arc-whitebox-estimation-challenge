"""Inert receiver-by-birth adapter. Numerical imports are explicit and lazy."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
CO = NEW/'current_bd_p_coadaptation_v1'
NODE = NEW/'birth_direction_node_head_v1'
COMP = NEW/'current_p1_node_composition_v1'
COMP_SHA = '2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a'
PINS = {
    CO/'model.py': '556f3b97613110e638f48127e40112ab203623fa9f6b076b0eab7f07d24658a0',
    NODE/'model.py': '61833c47e71867f01376a0a3bf3a976b56b67024c485212015f38e7934aadc71',
    COMP/'build.py': 'f3a3c95b3a7e99a40d8951800fb9d6ec0c0252f0bd4193c3cc2941e6712e61bf',
    COMP/'PROTOCOL.md': '5aa1fefd2e7b60ce77814f39efc4dcea3bcd2cc7ed372d4841da4eb8f973696c',
    COMP/'build_v1/result.json': '78f7adbb09ce5749a442c0b4d171110627a0881a478fece4e19f4f5467ec26d6',
    COMP/'build_v1/completion.json': 'd005b2245db255169bc5807349279a52271d451c57b9fa23c1c760e9b1a93306',
    COMP/'build_v1/terminal_estimator.py': COMP_SHA,
}
_CO = _CLASS = _BASE = _TORCH = _NODE_FN = _LITERALS = None


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


def authenticate():
    """Full source admission occurs before numerical imports, exactly once."""
    global _CO, _LITERALS
    require(_CO is None, 'One explicit source admission per process')
    pins = {str(p.resolve()): h for p, h in PINS.items()}
    require(all(sha(p) == h for p, h in pins.items()), 'Frozen source/build hashes')
    builder = module(COMP/'build.py', 'receiver_comp_source_builder')
    sealed = builder.verify_build(PINS[COMP/'build_v1/result.json'], COMP_SHA,
                                  PINS[COMP/'build.py'], PINS[COMP/'PROTOCOL.md'])
    pins.update(sealed['source_bindings']); pins.update(sealed['input_bindings'])
    _CO = module(CO/'model.py', 'receiver_original_co')
    pins.update(_CO.authenticate())
    wanted = {'_GNN_LITERAL', '_BDNH_A_LITERAL', '_BD_COEFFICIENT_LITERAL'}
    literals = {}
    for node in ast.parse((COMP/'build_v1/terminal_estimator.py').read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in wanted:
                require(name not in literals, 'Unique frozen literal')
                literals[name] = ast.literal_eval(node.value)
    require(set(literals) == wanted, 'All three exact current-COMP literals')
    require(_CO.coefficient_bytes(literals['_BD_COEFFICIENT_LITERAL']) ==
            _CO.coefficient_bytes(_CO.frozen_coefficients()), 'Exact cstar owner')
    _LITERALS = literals
    return pins


def runtime():
    global _BASE, _TORCH, _NODE_FN
    require(_CO is not None and _LITERALS is not None, 'Authenticate before numerical work')
    if _BASE is None:
        _BASE = _CO._base_module()
        _TORCH = _BASE.torch
        source = (NODE/'model.py').read_text()
        functions = [n for n in ast.parse(source).body
                     if isinstance(n, ast.FunctionDef) and n.name == 'node_coefficients']
        require(len(functions) == 1, 'One unchanged original NODE coefficient function')
        env = {'torch': _TORCH}
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(NODE/'model.py')+':function', 'exec'), env)
        _NODE_FN = env['node_coefficients']
    return _BASE


def frozen(device):
    """Explicit current COMP parameters; never default to d50 P or zero A."""
    base = runtime()
    P = {k: base.torch.tensor(v, dtype=base.torch.float32, device=device)
         for k, v in _LITERALS['_GNN_LITERAL'].items()}
    A = base.torch.tensor(_LITERALS['_BDNH_A_LITERAL'], dtype=base.torch.float32, device=device)
    require(A.shape == (64, 4), 'Current frozen NODE A shape')
    return P, A


def geometry(B0, H, Theta, dtype):
    """Stored-base displacement only; never inspect a target or detach H."""
    torch = _TORCH
    centered = H - H.mean(dim=0)
    scale = ((centered * centered).mean(dim=0) + 1e-6).sqrt()
    Z = (centered / scale).clamp(-8, 8)
    R = (((Z @ Theta) @ Z.T) / 64).tanh()
    B = B0.to(dtype)
    V = B * R.to(dtype)
    norm2 = (B * B).sum(0)
    denominator = torch.where(norm2 == 0, torch.ones_like(norm2), norm2)
    F = V - B * ((B * V).sum(0) / denominator)[None, :]
    F = _BASE.diagonal(F, torch.zeros_like(norm2))
    rounded = (B + .1 * F).float()
    new = torch.where(B0 == 0, B0, rounded)
    D = new.to(dtype) - B
    return new, F, D


def extra_plane(B0, D, t, s, a, alive, births, dtype):
    B = B0.to(dtype)
    aa = a.float().to(dtype)
    tt, ss = (t * alive).float().to(dtype), (s * alive).float().to(dtype)
    w, z = aa * (tt - ss) * births, aa * (tt * tt - ss * ss) * births
    extra = (2 * B * D + D * D) * w[None, :] + z[:, None] * D.T
    return _BASE.diagonal(extra, _TORCH.zeros_like(t))


def card_class():
    global _CLASS
    if _CLASS is not None:
        return _CLASS
    base = runtime()
    parent = _CO._card_class()  # the actual class, not the PCard __new__ facade
    changes = (
        ('    vv = torch.diagonal(Cpre)\n',
         '    vv = torch.diagonal(Cpre)\n    self._record_variance(l, vv)\n'),
        ("    delta, hs = gnn(l, x, aux['sigma'].float(), w, hstate, P)\n",
         "    delta, hs = gnn(l, x, aux['sigma'].float(), w, hstate, P)\n"
         '    node_c = self.layer_node_coefficients(l, hs) if l < self.depth-1 else None\n'),
        ('birth_action(bp,Cpre,k21,aux,self.layer_coefficients(l),t,s,a,alive,births)',
         'self.receiver_action(l,hs,bp,Cpre,k21,aux,node_c,t,s,a,alive,births)'),
    )
    source = base.source
    for old, new in changes:
        source = base.replace_once(source, old, new)
    inverse = source
    for old, new in reversed(changes):
        inverse = base.replace_once(inverse, new, old)
    require(inverse == base.source, 'Exact original step inverse outside three inserts')
    code, count = base.rewrite_clamps(source, str(__file__)+':step')
    require(count == 2, 'Exactly original two physical clamps')
    env = dict(base.env)
    exec(code, env)

    class ReceiverCard(parent):
        def __init__(self, rank, Theta, *, P, A, device='mps', weights=None, enabled=True):
            require(P is not None and A is not None, 'Explicit frozen current-COMP P and A required')
            super().__init__(rank, P=P, device=device, weights=weights, audit_selections=True)
            require(A.shape == (64, 4) and A.dtype == base.torch.float32 and A.device == self.w.device,
                    'Frozen NODE operand contract')
            require(Theta.shape == (64, 64) and Theta.dtype == base.torch.float32
                    and Theta.device == self.w.device, 'New shared Theta contract')
            require(not A.requires_grad and all(not p.requires_grad for p in self.P0.values()),
                    'P and NODE A remain frozen')
            require(all(p.detach().cpu().contiguous().numpy().tobytes() ==
                        base.np.asarray(_LITERALS['_GNN_LITERAL'][key], dtype=base.np.float32).tobytes()
                        for key, p in self.P0.items())
                    and A.detach().cpu().contiguous().numpy().tobytes() ==
                    base.np.asarray(_LITERALS['_BDNH_A_LITERAL'], dtype=base.np.float32).tobytes(),
                    'Explicit operands equal the exact retained COMP P and NODE A, not d50 defaults')
            require(bool(base.torch.isfinite(A).all().detach().cpu())
                    and bool(base.torch.isfinite(Theta).all().detach().cpu()), 'Finite parameters')
            self.A, self.Theta, self.enabled = A, Theta, bool(enabled)
            self.receiver_source_sha256 = hashlib.sha256(source.encode()).hexdigest()

        def layer_node_coefficients(self, layer, H):
            if layer == 0:
                return self.coefficients[0]
            require(1 <= layer <= 14, 'Only original NODE birth layers')
            return _NODE_FN(H, self.A, self.coefficients[layer])

        def receiver_action(self, layer, H, B, C, K21, aux, coefficients, t, s, a, alive, births):
            B0, old_increment = base.birth_action(B, C, K21, aux, coefficients, t, s, a, alive, births)
            if not self.enabled or layer == 0:
                return B0, old_increment
            require(1 <= layer <= 14, 'Receiver action only birth1..14')
            B1, _, D = geometry(B0, H, self.Theta, B.dtype)
            return B1, old_increment + extra_plane(B0, D, t, s, a, alive, births, B.dtype)

        def step(self, layer, state, control, P):
            require(P is self.P0 and not bool((control.detach() != 0).any().cpu()),
                    'No P override or exogenous controls')
            self._layer, self._selection_slot = layer, 0
            return env['step'](self, layer, state, control, P)

    _CLASS = ReceiverCard
    return _CLASS


def card(rank, Theta, *, P, A, device='mps', weights=None, enabled=True):
    return card_class()(rank, Theta, P=P, A=A, device=device, weights=weights, enabled=enabled)
