"""Inert exact-COMP adapter exposing only the existing exported cstar."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE, NEW = Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent
JOINT = NEW/'current_comp_joint_p_node_v1'
RB = NEW/'current_comp_receiver_birth_v1'
AUDIT = NEW/'CURRENT_COMP_CSTAR_FIXED_DIRECTION_SOURCE_AUDIT_20260910_V1.md'
COMP_SHA = '2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a'
PKEYS = ('msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b',
         'enc4_w', 'enc4_b', 'head', 'head_b', 'gain')
INACTIVE_KEYS = ('head', 'head_b', 'gain')
PARAMETER_COUNT, ACTIVE_COUNT = 60, 56
PINS = {
    JOINT/'model.py': 'cb49d7d598a76e430f388e6f3dc84b6b79d89eaf8a0e12cd3c19f8caca61cc86',
    JOINT/'PROTOCOL.md': '330f744e26f5672748ce94fc20f6e0f617358f83883d6a6c0c4b25c185d11830',
    NEW/'current_bd_p_coadaptation_v1/RESULT_V1.md':
        '010aabbf03ada6ad5d8b9509ebe1ce55a3fc02bc1048b66684fd92381d86ca09',
    RB/'information_v1/result.json':
        '73bec2488604faebd40ecd2024990969a0983df8fe6b8c9b9004bb04333cf3f9',
    RB/'information_watch_v1/result.json':
        '5cd65c4fbcd69f0a6daf0a3f82307edfa06bdf818e31e5345b283d230139478c',
}
_JOINT = _CO = _BASE = _CLASS = _BINDINGS = _PROOF = None


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
    require(spec is not None and spec.loader is not None, 'Pinned module specification')
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def authenticate():
    """Stdlib source admission; no model, numerical import or target access."""
    global _JOINT, _CO, _BINDINGS
    if _BINDINGS is not None:
        require(all(sha(p) == h for p, h in _BINDINGS.items()), 'All admitted bytes remain unchanged')
        return dict(_BINDINGS)
    require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
            'Source admission must precede numerical imports')
    pins = {str(p.resolve()): h for p, h in PINS.items()}
    require(all(sha(p) == h for p, h in pins.items()), 'Frozen cstar-adapter dependency hashes')
    for path in (Path(__file__).resolve(), HERE/'PROTOCOL.md', AUDIT):
        pins[str(path.resolve())] = sha(path)
    old = json.loads((RB/'information_v1/result.json').read_text())
    watch = json.loads((RB/'information_watch_v1/result.json').read_text())
    require(old['passed'] is False and old['completed'] is False
            and watch['passed'] is False, 'Historical RB numerical failure remains failed')
    _JOINT = module(JOINT/'model.py', 'cstar_information_exact_joint_owner')
    pins.update(_JOINT.authenticate())
    require(_JOINT.COMP_SHA == COMP_SHA and _JOINT.PKEYS == PKEYS
            and _JOINT.INACTIVE_KEYS == INACTIVE_KEYS, 'Exact retained COMP owner and P surface')
    _CO = _JOINT._CO
    require(_CO.coefficient_bytes(_JOINT._LITERALS['_BD_COEFFICIENT_LITERAL'])
            == _CO.coefficient_bytes(_CO.frozen_coefficients()), 'Exact exported native cstar, no retanh')
    _BINDINGS = pins
    return dict(pins)


def runtime():
    global _BASE
    require(_BINDINGS is not None and _JOINT is not None, 'Authenticate before explicit numerical runtime')
    if _BASE is None:
        _BASE = _JOINT.runtime()
    return _BASE


def frozen(device, *, trainable_cstar=True):
    """Exact COMP P/A plus the exact native cstar leaf; no theta conversion."""
    base = runtime()
    P, A = _JOINT.frozen(device, trainable_P=False, trainable_A=False)
    cstar = base.torch.tensor(_JOINT._LITERALS['_BD_COEFFICIENT_LITERAL'],
        dtype=base.torch.float32, device=device, requires_grad=bool(trainable_cstar))
    require(cstar.shape == (15, 4) and cstar.numel() == PARAMETER_COUNT, 'Existing60-value coefficient shape')
    return P, A, cstar


def _frozen_parents(P, A):
    base = runtime()
    require(set(P) == set(_JOINT._LITERALS['_GNN_LITERAL']), 'Complete exact COMP P mapping')
    for key, value in P.items():
        require(not value.requires_grad and value.grad is None
                and base.torch.autograd.forward_ad.unpack_dual(value).tangent is None
                and value.detach().cpu().contiguous().numpy().tobytes()
                == base.np.asarray(_JOINT._LITERALS['_GNN_LITERAL'][key], dtype=base.np.float32).tobytes(),
                'Every COMP P field remains byte-frozen: '+key)
    require(not A.requires_grad and A.grad is None
            and base.torch.autograd.forward_ad.unpack_dual(A).tangent is None
            and A.detach().cpu().contiguous().numpy().tobytes()
            == base.np.asarray(_JOINT._LITERALS['_BDNH_A_LITERAL'], dtype=base.np.float32).tobytes(),
            'Exact frozen COMP NODE A')


def card_class():
    global _CLASS, _PROOF
    if _CLASS is not None:
        return _CLASS
    base = runtime()
    parent = _JOINT.card_class()
    inherited = _JOINT.source_proof()
    require(inherited['inverse_source_byte_exact'] and inherited['original_node_changes_byte_exact']
            and inherited['variance_hook_count'] == 1 and inherited['physical_clamp_count'] == 2,
            'Exact original compiled COMP step and one domain observer')

    class CstarCard(parent):
        def __init__(self, rank, *, P, A, cstar, device='mps', weights=None):
            require(P is not None and A is not None and cstar is not None,
                    'Explicit current P/A and actual supplied cstar; no default state')
            super().__init__(rank, P=P, A=A, device=device, weights=weights)
            _frozen_parents(P, A)
            torch = base.torch
            require(isinstance(cstar, torch.Tensor) and cstar.shape == (15, 4)
                    and cstar.dtype == torch.float32 and cstar.device == self.w.device
                    and bool(torch.isfinite(cstar).all().detach().cpu())
                    and bool((cstar.abs() < 1).all().detach().cpu()),
                    'Finite exact-shape supplied FP32 cstar strictly inside the original coefficient domain')
            require(cstar[0].detach().cpu().contiguous().numpy().tobytes()
                    == self.coefficients[0].detach().cpu().contiguous().numpy().tobytes(),
                    'Gaussian layer0 coefficient bytes remain frozen')
            tangent = torch.autograd.forward_ad.unpack_dual(cstar).tangent
            require(tangent is None or (bool(torch.isfinite(tangent).all().detach().cpu())
                    and bool((tangent[0] == 0).all().detach().cpu())),
                    'Only finite active cstar forward tangents; row0 remains inactive')
            # The parent constructor has authenticated the original frozen literal.
            # Replace only its reference; never tensor-copy or detach the supplied
            # leaf/dual. The unmodified inherited NODE call reads this attribute.
            self.coefficients = cstar
            self.cstar = cstar
            require(self.coefficients is cstar and self.A is A
                    and all(self.P0[k] is v for k, v in P.items()), 'All actual operand identities retained')
            self.source_bindings.update(_BINDINGS)

        def diagnostics(self):
            result = super().diagnostics()
            result.update(allocated_parameters=PARAMETER_COUNT, active_parameters=ACTIVE_COUNT,
                frozen_parent_parameters=45456, frozen_P=True, frozen_NODE_A=True,
                frozen_cstar=False, live_cstar_allowed=True, inactive_cstar_row=0,
                live_P_and_A_allowed=False, no_receiver_geometry=True,
                native_gradient_qualified=False, no_native_gradient_claim=True,
                cstar_source_proof=dict(_PROOF))
            return result

    require(CstarCard.step is parent.step
            and CstarCard.layer_node_coefficients is parent.layer_node_coefficients
            and CstarCard.run is parent.run, 'No new step, NODE coefficient or checkpoint implementation')
    _PROOF = dict(parent=inherited, original_comp_step_unchanged=True,
        inherited_step_inverse_byte_exact=True, original_node_function_unchanged=True,
        original_birth_action_and_stored_plane_unchanged=True,
        original_checkpoint_domain_replay_unchanged=True,
        parameter_binding_only=True, supplied_cstar_identity_preserved=True,
        exact_exported_cstar_null=True, no_retanh_or_inverse_tanh=True,
        frozen_P_and_A=True, node_amplitude_and_endogenous_state_derivatives_live=True)
    _CLASS = CstarCard
    return _CLASS


def card(rank, *, P, A, cstar, device='mps', weights=None):
    return card_class()(rank, P=P, A=A, cstar=cstar, device=device, weights=weights)


def reference_card(rank, *, P, A, device='mps', weights=None):
    """Explicit same-backend unmodified JOINT null reference; no new run."""
    _frozen_parents(P, A)
    return _JOINT.card(rank, P=P, A=A, device=device, weights=weights)


def source_proof():
    require(_PROOF is not None, 'Explicit card-class construction precedes proof access')
    return dict(_PROOF)
