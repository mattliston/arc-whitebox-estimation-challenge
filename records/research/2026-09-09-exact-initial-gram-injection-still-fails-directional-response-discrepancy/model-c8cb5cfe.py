"""Inert, CPU-moment/MPS-product qualification adapter; no fit or targets."""
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CO = HERE.parent / 'current_bd_p_coadaptation_v1'
CO_SHA = '556f3b97613110e638f48127e40112ab203623fa9f6b076b0eab7f07d24658a0'
_OWNER = _CARD = None


def require(value, message):
    if not value:
        raise RuntimeError(message)


def owner():
    global _OWNER
    path = CO / 'model.py'
    require(hashlib.sha256(path.read_bytes()).hexdigest() == CO_SHA, 'Frozen CO source changed')
    if _OWNER is None:
        spec = importlib.util.spec_from_file_location('hybrid_cp_frozen_owner', path)
        _OWNER = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = _OWNER
        spec.loader.exec_module(_OWNER)
    return _OWNER


def authenticate():
    result = owner().authenticate()
    result[str(CO / 'model.py')] = CO_SHA
    for path in (Path(__file__).resolve(), HERE / 'PROTOCOL.md'):
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def card_class():
    global _CARD
    if _CARD is not None:
        return _CARD
    frozen = owner()
    authenticate()
    parent = frozen._card_class()
    base = frozen._base_module()
    torch = base.torch
    observed = base.replace_once(base.source, '    vv = torch.diagonal(Cpre)\n',
        '    vv = torch.diagonal(Cpre)\n    self._record_variance(l, vv)\n')
    changes = (
        ('    pp = w.T@p', '    pp = self._carrier_product(w.T, p)'),
        ('        subplane = ((left*left)@(right*lam).T).to(dt)',
         '        subplane = self._carrier_product(left*left, (right*lam).T).to(dt)'),
    )
    source = observed
    for old, new in changes:
        source = base.replace_once(source, old, new)
    inverse = source
    for old, new in reversed(changes):
        inverse = base.replace_once(inverse, new, old)
    require(inverse == observed, 'Hybrid two-site inverse differs')
    code, count = base.rewrite_clamps(source, str(__file__) + ':step')
    require(count == 2, 'Original clamp roster changed')
    env = dict(base.env)
    exec(code, env)

    class HybridCard(parent):
        def __init__(self, rank, P=None, weights=None, enabled=True):
            super().__init__(rank, P=P, device='cpu', weights=weights, audit_selections=True)
            self.enabled = bool(enabled)
            self.transferred_bytes = 0
            self.offloaded_products = 0
            self.hybrid_source_sha256 = hashlib.sha256(source.encode()).hexdigest()
            require(not self.enabled or torch.backends.mps.is_available(), 'MPS unavailable')

        def _carrier_product(self, left, right):
            require(left.dtype == right.dtype == torch.float32, 'Only existing FP32 products')
            require(left.device.type == right.device.type == 'cpu', 'CPU owned operands required')
            require(left.ndim == right.ndim == 2 and left.shape[1] == right.shape[0],
                    'Matrix shape mismatch')
            if not self.enabled or not left.numel() or not right.numel():
                return left @ right
            self.transferred_bytes += 4 * (left.numel() + right.numel() + left.shape[0] * right.shape[1])
            self.offloaded_products += 1
            return (left.to('mps') @ right.to('mps')).to('cpu')

        def step(self, layer, state, control, P):
            require(P is self.P0 and not bool((control.detach() != 0).any().cpu()),
                    'No payload override or exogenous control')
            self._layer, self._selection_slot = layer, 0
            return env['step'](self, layer, state, control, P)

        def run(self, *args, **kwargs):
            self.transferred_bytes = self.offloaded_products = 0
            return super().run(*args, **kwargs)

        def diagnostics(self):
            result = super().diagnostics()
            result.update(hybrid_enabled=self.enabled, hybrid_source_sha256=self.hybrid_source_sha256,
                          transferred_bytes=self.transferred_bytes,
                          transfer_accounting='Explicit forward-helper CPU/MPS copies including checkpoint replay; excludes automatic backward copies and allocator staging',
                          offloaded_products=self.offloaded_products,
                          source_inverse_byte_exact=True, moment_device='cpu', moment_dtype='float64',
                          parameter_device='cpu', carrier_product_device='mps' if self.enabled else 'cpu')
            return result

    _CARD = HybridCard
    return _CARD


def card(rank, P=None, weights=None, enabled=True):
    return card_class()(rank, P=P, weights=weights, enabled=enabled)
