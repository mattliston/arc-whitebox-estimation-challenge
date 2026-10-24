"""Inert exact-COMP adapter with jointly live existing P and NODE A."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE, NEW = Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent
CO, NODE, COMP = (NEW/name for name in ('current_bd_p_coadaptation_v1',
                                      'birth_direction_node_head_v1', 'current_p1_node_composition_v1'))
COMP_SHA = '2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a'
PKEYS = ('msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b',
         'enc4_w', 'enc4_b', 'head', 'head_b', 'gain')
INACTIVE_KEYS = ('head', 'head_b', 'gain')
PARAMETER_COUNT, ACTIVE_COUNT = 45456, 45390
PINS = {
    CO/'model.py': '556f3b97613110e638f48127e40112ab203623fa9f6b076b0eab7f07d24658a0',
    NODE/'model.py': '61833c47e71867f01376a0a3bf3a976b56b67024c485212015f38e7934aadc71',
    COMP/'build.py': 'f3a3c95b3a7e99a40d8951800fb9d6ec0c0252f0bd4193c3cc2941e6712e61bf',
    COMP/'PROTOCOL.md': '5aa1fefd2e7b60ce77814f39efc4dcea3bcd2cc7ed372d4841da4eb8f973696c',
    COMP/'build_v1/result.json': '78f7adbb09ce5749a442c0b4d171110627a0881a478fece4e19f4f5467ec26d6',
    COMP/'build_v1/completion.json': 'd005b2245db255169bc5807349279a52271d451c57b9fa23c1c760e9b1a93306',
    COMP/'build_v1/terminal_estimator.py': COMP_SHA,
    COMP/'score_v3/result.json': '17bd92bb65abb5d113cf3cd57df175b865f03295a7e0d96eadfeb133b6658390',
    COMP/'score_v3/completion.json': '23d4e593af714034e7208a3a5947bef47386cf87a47dd5872c568a22c65415b4',
    COMP/'panel_v2/score/result.json': '379f7fa393f81dfe9af0cd30cb9d8ad5a3a08e71f34242c58bc30f540569023c',
    COMP/'panel_v2/score/completion.json': 'c539b369bd5e8ca15ee6213c6acb9e4e208e3fcaeda52fa3d146dbc1d0d3733f',
    NEW/'JOINT_P_NODE_COMPATIBILITY_SOURCE_AUDIT_20260909_V1.md':
        '85f9941adb0c582c6d1d03cba2487682a918f08c85170a53c34c3ce2f12bfe1b',
}
CHANGES = (
    ('    vv = torch.diagonal(Cpre)\n',
     '    vv = torch.diagonal(Cpre)\n    self._record_variance(l, vv)\n'),
    ("    delta, hs = gnn(l, x, aux['sigma'].float(), w, hstate, P)\n",
     "    delta, hs = gnn(l, x, aux['sigma'].float(), w, hstate, P)\n"
     '    node_c = self.layer_node_coefficients(l, hs) if l < self.depth-1 else None\n'),
    ('birth_action(bp,Cpre,k21,aux,self.layer_coefficients(l),t,s,a,alive,births)',
     'birth_action(bp,Cpre,k21,aux,node_c,t,s,a,alive,births)'),
)
_CO = _BASE = _CLASS = _NODE_FN = _LITERALS = _BINDINGS = _PROOF = None
_NODE_AST = _NODE_TEXT = None


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
    """Source/receipt admission only; caller pins this new model/protocol."""
    global _CO, _LITERALS, _BINDINGS, _NODE_AST, _NODE_TEXT
    if _BINDINGS is not None:
        require(all(sha(p) == h for p, h in _BINDINGS.items()), 'Admitted joint-source bytes unchanged')
        return dict(_BINDINGS)
    require(not any(x in sys.modules for x in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
            'Explicit source admission precedes numerical imports')
    pins = {str(p.resolve()): h for p, h in PINS.items()}
    require(all(sha(p) == h for p, h in pins.items()), 'Frozen joint dependency hashes')
    for path in (Path(__file__).resolve(), HERE/'PROTOCOL.md'):
        pins[str(path)] = sha(path)
    builder = module(COMP/'build.py', 'joint_comp_source_builder')
    sealed = builder.verify_build(PINS[COMP/'build_v1/result.json'], COMP_SHA,
        PINS[COMP/'build.py'], PINS[COMP/'PROTOCOL.md'])
    pins.update(sealed['source_bindings']); pins.update(sealed['input_bindings'])
    for folder in (COMP/'score_v3', COMP/'panel_v2/score'):
        record, end = (json.loads((folder/name).read_text()) for name in ('result.json', 'completion.json'))
        require(record['passed'] and record['completed'] and record['integrity_passed']
            and record['hashes_unchanged'] and record['sources_unchanged']
            and record['candidate_sha256'] == COMP_SHA and end['passed'] and end['completed']
            and end['result_sha256'] == PINS[folder/'result.json'], 'Retained COMP endpoint seals')
        if folder == COMP/'score_v3':
            require(record['advances_to_independent14'], 'Original native40 gate passed')
        else:
            require(record['qualification_passed'] and record['retain_local_component']
                and all(record['group_strict_gain_gate'][g] for g in ('public8', 'virgin6')),
                'Both original independent14 groups passed')
    _CO = module(CO/'model.py', 'joint_original_co')
    pins.update(_CO.authenticate())
    require(_CO.PKEYS == PKEYS and _CO.INACTIVE_KEYS == INACTIVE_KEYS
        and _CO.PARAMETER_COUNT+256 == PARAMETER_COUNT and _CO.ACTIVE_COUNT+256 == ACTIVE_COUNT,
        'Exact original P surface plus256 existing A coordinates')
    wanted = {'_GNN_LITERAL', '_BDNH_A_LITERAL', '_BD_COEFFICIENT_LITERAL'}
    literals = {}
    for node in ast.parse((COMP/'build_v1/terminal_estimator.py').read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in wanted:
                require(name not in literals, 'Unique exact COMP literal')
                literals[name] = ast.literal_eval(node.value)
    require(set(literals) == wanted and _CO.coefficient_bytes(literals['_BD_COEFFICIENT_LITERAL'])
        == _CO.coefficient_bytes(_CO.frozen_coefficients()), 'Exact current literals and frozen cstar')
    node_source = (NODE/'model.py').read_text(); tree = ast.parse(node_source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'node_coefficients']
    changes = [n.value for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'CHANGES' for t in n.targets)]
    require(len(functions) == len(changes) == 1 and ast.literal_eval(changes[0]) == CHANGES,
            'Exactly the original NODE three-splice owner, including one CO variance hook')
    _NODE_AST = functions[0]
    _NODE_TEXT = ''.join(node_source.splitlines(keepends=True)[_NODE_AST.lineno-1:_NODE_AST.end_lineno])
    _LITERALS, _BINDINGS = literals, pins
    return dict(pins)


def runtime():
    global _BASE, _NODE_FN
    require(_CO is not None and _BINDINGS is not None, 'Authenticate before explicit numerical runtime')
    if _BASE is None:
        _BASE = _CO._base_module()
        env = {'torch': _BASE.torch}
        exec(compile(ast.Module(body=[_NODE_AST], type_ignores=[]), str(NODE/'model.py')+':exact_function', 'exec'), env)
        _NODE_FN = env['node_coefficients']
    return _BASE


def frozen(device, *, trainable_P=False, trainable_A=False):
    """Exact COMP initial values, with only explicit gradient flags optional."""
    base = runtime()
    P = {key: base.torch.tensor(value, dtype=base.torch.float32, device=device,
             requires_grad=bool(trainable_P and key in PKEYS))
         for key, value in _LITERALS['_GNN_LITERAL'].items()}
    A = base.torch.tensor(_LITERALS['_BDNH_A_LITERAL'], dtype=base.torch.float32,
                         device=device, requires_grad=bool(trainable_A))
    require(A.shape == (64, 4), 'Exact existing NODE A shape')
    return P, A


def card_class():
    global _CLASS, _PROOF
    if _CLASS is not None:
        return _CLASS
    base = runtime(); parent = _CO._card_class()
    source = base.source
    for old, new in CHANGES:
        source = base.replace_once(source, old, new)
    inverse = source
    for old, new in reversed(CHANGES):
        inverse = base.replace_once(inverse, new, old)
    require(inverse == base.source and source.count('    self._record_variance(l, vv)\n') == 1,
            'Exact original birth-step inverse and one variance observer')
    code, count = base.rewrite_clamps(source, str(__file__)+':step')
    require(count == 2, 'Original two physical clamp rewrites')
    env = dict(base.env); exec(code, env)
    _PROOF = dict(base_source_sha256=hashlib.sha256(base.source.encode()).hexdigest(),
        joint_source_sha256=hashlib.sha256(source.encode()).hexdigest(), inverse_source_byte_exact=True,
        original_node_changes_byte_exact=True, variance_hook_count=1, physical_clamp_count=count,
        node_function_sha256=hashlib.sha256(_NODE_TEXT.encode()).hexdigest(),
        original_birth_action_unchanged=True, no_receiver_geometry=True,
        original_co_constructor_checkpoint_diagnostics=True)

    class JointCard(parent):
        def __init__(self, rank, *, P, A, device='mps', weights=None):
            require(P is not None and A is not None, 'Explicit current-or-proposed P and A; no defaults')
            super().__init__(rank, P=P, device=device, weights=weights, audit_selections=True)
            torch = base.torch
            require(isinstance(A, torch.Tensor) and A.shape == (64, 4) and A.dtype == torch.float32
                and A.device == self.w.device and bool(torch.isfinite(A).all().detach().cpu()),
                'Finite same-shaped supplied A on the actual card device')
            require(all(self.P0[key] is value for key, value in P.items()), 'All supplied P identities retained')
            for key, value in self.P0.items():
                if key not in PKEYS:
                    require(not value.requires_grad and value.detach().cpu().contiguous().numpy().tobytes()
                        == base.np.asarray(_LITERALS['_GNN_LITERAL'][key], dtype=base.np.float32).tobytes(),
                        'Exact frozen COMP normalizer/input field: '+key)
                if key in INACTIVE_KEYS:
                    require(value[0].detach().cpu().contiguous().numpy().tobytes()
                        == base.np.asarray(_LITERALS['_GNN_LITERAL'][key], dtype=base.np.float32)[0].tobytes(),
                        'Exact inactive COMP layer0 bytes: '+key)
            require(not self.coefficients.requires_grad, 'cstar remains frozen')
            self.A = A
            self.node_head_calls = {}
            self.adapter_source_sha256 = _PROOF['joint_source_sha256']
            self.source_bindings.update(_BINDINGS)

        def layer_node_coefficients(self, layer, state):
            if layer == 0:
                return self.coefficients[0]
            require(1 <= layer <= 14, 'Only original NODE birth layers1..14')
            require(bool(base.torch.isfinite(state).all().detach().cpu()), 'Finite current P hidden state')
            result = _NODE_FN(state, self.A, self.coefficients[layer])
            require(bool(base.torch.isfinite(result).all().detach().cpu()), 'Finite original NODE coefficients')
            self.node_head_calls[layer] = self.node_head_calls.get(layer, 0)+1
            return result

        def step(self, layer, state, control, P):
            require(P is self.P0 and not bool((control.detach() != 0).any().cpu()),
                    'No P override or exogenous controls')
            self._layer, self._selection_slot = layer, 0
            return env['step'](self, layer, state, control, P)

        def run(self, P=None, controls=None, training=False, capture=False):
            self.node_head_calls.clear()
            return super().run(P=P, controls=controls, training=training, capture=capture)

        def diagnostics(self):
            result = super().diagnostics()
            result.update(allocated_parameters=PARAMETER_COUNT, active_parameters=ACTIVE_COUNT,
                node_head_calls={str(k): v for k, v in self.node_head_calls.items()},
                joint_source_proof=dict(_PROOF), live_P_and_A_allowed=True, frozen_cstar=True)
            return result

    _CLASS = JointCard
    return _CLASS


def card(rank, *, P, A, device='mps', weights=None):
    return card_class()(rank, P=P, A=A, device=device, weights=weights)


def source_proof():
    require(_PROOF is not None, 'Construct the explicit runtime card before requesting its compiled-source proof')
    return dict(_PROOF)
