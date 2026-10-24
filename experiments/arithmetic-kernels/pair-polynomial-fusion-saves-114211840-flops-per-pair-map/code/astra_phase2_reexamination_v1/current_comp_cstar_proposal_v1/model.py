"""Inert current-COMP relative-headroom proposal; no derivative qualification."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE, NEW = Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent
CS = NEW/'current_comp_cstar_information_v1'
P1 = NEW/'current_d50_native_objective_p1_v1'
COMP_SHA = '2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a'
PKEYS = ('msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b',
         'enc4_w', 'enc4_b', 'head', 'head_b', 'gain')
INACTIVE_KEYS = ('head', 'head_b', 'gain')
PARAMETER_COUNT = ACTIVE_COUNT = 56
DELTA_SHAPE, COEFFICIENT_SHAPE = (14, 4), (15, 4)
PINS = {
    CS/'model.py': 'a34f7de73914b76aee295ad8dba488bbd6257e93cfa33aa9d58124c2610ed0f1',
    CS/'PROTOCOL.md': 'c3660d88e1a9e90d3ff040b5f9cf191a323b0cf00589d1167fe4ff60bf851682',
    CS/'qualification.py': '894128f110b098ea531c930cd7498cf0922ad5227f0c3839f8b9cd7a841db420',
    CS/'QUALIFICATION_PROTOCOL.md': 'b77df4b6d68bb98c7ec6112bf33c150b416c4c467717a427b4227669cd6610a9',
    CS/'watch_qualification.py': '33c44567fd5b80ecb4d8b1e9cd8faae8865a02b80a4e2f59231baea2a5e45141',
    CS/'WATCH_SPEC.md': '08120a7cd04d9499debf9218fe278015f26c9fafa5256cbb2ab763fa5ccefad5',
    CS/'qualification_v1/result.json': 'd2c01e7013633c23b0fe97ce0a2444065232527ab8730951ee7eb432df36a77b',
    CS/'qualification_v1/completion.json': '141253a47adab3bd7b87798241d49468b379305c959334f27fcfd12a6815651c',
    CS/'qualification_v1/finalization.json': '566e3ebf96ddd9fe90ec01379fbf98998d1eb38762ab337bd73f6be9261dfb1b',
    CS/'qualification_watch_v1/result.json': 'adb4a133dc06d21164d766a4e9e9a5a93867a7a6bd1456e0b8e833e9591fd555',
    CS/'qualification_watch_v1/completion.json': '888e107fc153b696926a060351854f7c7e2b06975a8ba743b6d0aef5f8472ec9',
    CS/'qualification_watch_v1/finalization.json': 'a7401eb8c2f326b766079069cccc00455e4140a9bcab83292fb16222f9c3dd87',
    CS/'information.py': '29315b42c602509056e97ebb6b0b69e3357a032a214605db8d4ea418afa481f2',
    CS/'INFORMATION_PROTOCOL.md': '350fc037dcbf347ab984b0999028ff9c0e48e8e1352da226807d7f838ed4c97b',
    P1/'PROTOCOL.md': '2141e2d4f025522a7cbea55fbd5ff10a66c313feacb1b4bd49a13f75a4b9af35',
    P1/'NATIVE40_RESULT.md': '34503b07d71d489365d54dc7646af2fc7d5ed741ee8f0dc6ddc92dea644b9660',
    P1/'INDEPENDENT14_RESULT.md': 'ffc1694526740a83dd2f9a57c5d6bed4b2679a8dcd1472b0b58f977fce72dbdc',
}
_CS = _CO = _BASE = _CLASS = _BINDINGS = _PROOF = None


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
    require(spec is not None and spec.loader is not None, 'Pinned source module')
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def authenticate():
    """Only source/JSON admission; old CS FALSE is preserved, never eligibility."""
    global _CS, _CO, _BINDINGS
    if _BINDINGS is not None:
        require(all(sha(p) == h for p, h in _BINDINGS.items()), 'All admitted bytes unchanged')
        return dict(_BINDINGS)
    require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
            'Authenticate before numerical imports')
    pins = {str(p.resolve()): h for p, h in PINS.items()}
    require(all(sha(p) == h for p, h in pins.items()), 'Frozen source/failure lineage')
    for folder in (CS/'qualification_v1', CS/'qualification_watch_v1'):
        r, c, f = (json.loads((folder/name).read_text())
                   for name in ('result.json', 'completion.json', 'finalization.json'))
        require(r['passed'] is False and c['passed'] is False and f['passed'] is False,
                'Every old CS stage/seal remains FAILED')
        require(c['result_sha256'] == pins[str((folder/'result.json').resolve())]
                and f['completion_sha256'] == pins[str((folder/'completion.json').resolve())],
                'Original CS three-seal links, not a new PASS')
        if folder.name == 'qualification_v1':
            require(r['completed'] is False and r['numerical_qualification_passed'] is False
                    and r['retired_call_count'] == 22
                    and f['result_sha256'] == c['result_sha256'], 'Original numerical stop intact')
        else:
            require(r['retired_calls'] == 22 and r['cleanup_passed'],
                    'Only actual old failed-watch accounting; not new eligibility')
    for path in (Path(__file__).resolve(), HERE/'PROTOCOL.md'):
        pins[str(path)] = sha(path)
    _CS = module(CS/'model.py', 'cstar_proposal_frozen_parameter_owner')
    pins.update(_CS.authenticate())
    require(_CS.COMP_SHA == COMP_SHA and _CS.PKEYS == PKEYS
            and _CS.INACTIVE_KEYS == INACTIVE_KEYS, 'Exact retained COMP and parent surfaces')
    _CO = _CS._CO
    _BINDINGS = pins
    return dict(pins)


def runtime():
    global _BASE
    require(_BINDINGS is not None and _CS is not None, 'Source admission precedes runtime')
    if _BASE is None:
        _BASE = _CS.runtime()
    return _BASE


def initial_coefficients(device):
    """Exact original exported FP32 c0; no inverse tanh or retanh."""
    base = runtime()
    value = base.torch.tensor(_CS._JOINT._LITERALS['_BD_COEFFICIENT_LITERAL'],
                              dtype=base.torch.float32, device=device)
    require(value.shape == COEFFICIENT_SHAPE and not value.requires_grad
            and value.detach().cpu().contiguous().numpy().tobytes()
            == _CO.coefficient_bytes(_CO.frozen_coefficients()), 'Exact original coefficient bytes')
    return value


def frozen(device, *, trainable_delta=True):
    base = runtime()
    P, A, c0 = _CS.frozen(device, trainable_cstar=False)
    del c0
    delta = base.torch.zeros(DELTA_SHAPE, dtype=base.torch.float32, device=device,
                            requires_grad=bool(trainable_delta))
    require(delta.is_leaf and delta.numel() == PARAMETER_COUNT, 'One exact zero56-coordinate leaf')
    return P, A, delta


def _coefficient_parts(delta):
    base = runtime(); torch = base.torch
    require(isinstance(delta, torch.Tensor) and delta.shape == DELTA_SHAPE
            and delta.dtype == torch.float32 and delta.is_leaf
            and torch.autograd.forward_ad.unpack_dual(delta).tangent is None
            and bool(torch.isfinite(delta).all().detach().cpu()), 'Finite actual FP32 proposal leaf; no FAD stage')
    c0 = initial_coefficients(delta.device)
    headroom = torch.ones_like(c0[1:]) - c0[1:].abs()
    active = c0[1:] + headroom * delta.tanh()
    actual = torch.cat((c0[:1], active), dim=0)
    require(headroom.dtype == active.dtype == actual.dtype == torch.float32
            and bool((headroom > 0).all().detach().cpu())
            and bool(torch.isfinite(actual).all().detach().cpu())
            and bool((actual.abs() < 1).all().detach().cpu()), 'Stored FP32 coefficient domain; stop, never clamp')
    require(actual[0].detach().cpu().contiguous().numpy().tobytes()
            == c0[0].detach().cpu().contiguous().numpy().tobytes(), 'Old Gaussian row0 bytes')
    if bool((delta.detach() == 0).all().cpu()):
        require(actual.detach().cpu().contiguous().numpy().tobytes()
                == c0.detach().cpu().contiguous().numpy().tobytes(), 'Computed zero action is byte-exact; no bypass')
    if delta.requires_grad:
        require(actual.requires_grad and not actual.is_leaf and actual.grad_fn is not None,
                'The supplied nonleaf coefficient retains the actual delta graph')
    return actual, c0, headroom


def coefficients(delta):
    """Export this actual computed coefficient tensor, not a reconstructed theta."""
    return _coefficient_parts(delta)[0]


def card_class():
    global _CLASS, _PROOF
    if _CLASS is not None:
        return _CLASS
    base = runtime(); parent = _CS.card_class()
    inherited = _CS.source_proof()
    require(inherited['original_comp_step_unchanged']
            and inherited['inherited_step_inverse_byte_exact']
            and inherited['node_amplitude_and_endogenous_state_derivatives_live'], 'Original physical source proof')

    class ProposalCard(parent):
        def __init__(self, rank, *, P, A, delta=None, cstar=None, device='mps', weights=None):
            require((delta is None) != (cstar is None), 'Exactly one delta or sealed-coefficient input')
            if delta is not None:
                actual, c0, headroom = _coefficient_parts(delta)
                mode = 'relative_headroom_delta'
            else:
                require(isinstance(cstar, base.torch.Tensor) and not cstar.requires_grad
                        and cstar.grad is None
                        and base.torch.autograd.forward_ad.unpack_dual(cstar).tangent is None,
                        'Terminal validation uses nontrainable supplied sealed coefficients')
                actual, c0 = cstar, initial_coefficients(cstar.device)
                headroom = base.torch.ones_like(c0[1:]) - c0[1:].abs()
                mode = 'sealed_exported_coefficients'
            super().__init__(rank, P=P, A=A, cstar=actual, device=device, weights=weights)
            self.delta, self.c0, self.headroom = delta, c0, headroom
            self.proposal_mode = mode
            require(self.coefficients is actual and self.cstar is actual
                    and self.delta is delta and self.A is A
                    and all(self.P0[k] is v for k, v in P.items()), 'Actual supplied operands, no copied parameter')
            self.source_bindings.update(_BINDINGS)

        def diagnostics(self):
            result = super().diagnostics()
            result.update(allocated_parameters=PARAMETER_COUNT, active_parameters=ACTIVE_COUNT,
                proposal_mode=self.proposal_mode, proposal_only=True,
                frozen_initial_cstar=True, inactive_cstar_row=0,
                live_delta_allowed=self.delta is not None, exported_coefficient_count=60,
                native_gradient_qualified=False, no_native_gradient_claim=True,
                CS_numerical_qualification_passed=False, CS_failure_preserved=True,
                cstar_proposal_source_proof=dict(_PROOF))
            return result

    require(ProposalCard.step is parent.step
            and ProposalCard.layer_node_coefficients is parent.layer_node_coefficients
            and ProposalCard.run is parent.run, 'No physical, NODE, replay or checkpoint override')
    _PROOF = dict(parent=inherited, parent_model_sha256=PINS[CS/'model.py'],
        original_comp_step_unchanged=True, inherited_step_inverse_byte_exact=True,
        original_node_function_unchanged=True, original_birth_action_and_stored_plane_unchanged=True,
        relative_headroom_supplier_only=True, actual_delta_and_nonleaf_cstar_identities=True,
        original_cstar_bytes_not_retanh=True, row0_concatenated_unchanged=True,
        frozen_P_A_c0=True, terminal_uses_sealed_coefficients_without_inverse=True,
        native_gradient_qualified=False, CS_failure_preserved=True)
    _CLASS = ProposalCard
    return _CLASS


def card(rank, *, P, A, delta, device='mps', weights=None):
    return card_class()(rank, P=P, A=A, delta=delta, device=device, weights=weights)


def card_from_coefficients(rank, *, P, A, cstar, device='mps', weights=None):
    return card_class()(rank, P=P, A=A, cstar=cstar, device=device, weights=weights)


def reference_card(rank, *, P, A, device='mps', weights=None):
    runtime()
    return _CS.reference_card(rank, P=P, A=A, device=device, weights=weights)


def source_proof():
    require(_PROOF is not None, 'Construct the explicit card class before requesting its source proof')
    return dict(_PROOF)
