"""Private late-only L12--15 encoder. No Torch import/model at module import.

Install on the fully configured current95c card AFTER private configure_v8's
install_m. The six cloned leaves are a separate parameter family; cd.P remains
the original frozen dictionary, including the W3 response metric encoder.
"""
import hashlib

SHAPES = {
    'enc0_w': (128, 126), 'enc0_b': (128,),
    'enc2_w': (128, 128), 'enc2_b': (128,),
    'enc4_w': (64, 128), 'enc4_b': (64,),
}
INITIAL_HASHES = {
    'enc0_w': '9284c1d601a9df393ede90929ae82e224753e0b2d78cd5c374b31c768dc8b14a',
    'enc0_b': 'ebe3e511277d2231d3d8ca56b98d3ec553ad610ffe8cbd37c5504115486755d1',
    'enc2_w': '7f4ceb7cd92e9d058d46cd1e56c94bbd4bf1de47c374607fec78833dcd97144a',
    'enc2_b': '7a6670bc65a34a666a51ecc9e8bf238cd121ac38c5c2f20214f2e1bead5ff01c',
    'enc4_w': '8e43cddec2d52ef030f00f87b488bc245cf938e6343f1a438dbb8bd6b9491586',
    'enc4_b': '1f513c531f3152dda6e3def79155eaf754aa59f2f2785c9ae011751c3eabcbdf',
}


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def tensors(value, torch):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item, torch)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item, torch)


class LateEncoderScope:
    def __init__(self, engine, torch, arrays=None):
        """Call after serve loads S56 params, HEADU and all request dials."""
        self.E, self.t = engine, torch
        assert not engine.UNFREEZE and not engine.LOSS_W
        assert not engine.M.DUMP15, 'DUMP15 has a separate diagnostic W3 path'
        self.original = engine.p
        self.original_hashes = {key: digest(engine.p[key]) for key in SHAPES}
        assert self.original_hashes == INITIAL_HASHES, 'Must bind frozen current S56 encoder'
        if arrays is not None:
            assert set(arrays) == set(SHAPES)
        self.leaves = {}
        for key, shape in SHAPES.items():
            source = engine.p[key] if arrays is None else arrays[key]
            assert source.dtype == torch.float32 and tuple(source.shape) == shape
            assert source.device.type == 'cpu' and bool(torch.isfinite(source).all())
            leaf = source.detach().clone().requires_grad_(True)
            assert leaf.is_leaf and leaf.data_ptr() != engine.p[key].data_ptr()
            self.leaves[key] = leaf
        self.frozen = []
        seen = set()
        collections = [engine.T, engine.DL, engine.PL.EPS]
        for name in ('HEADU', 'EDGEU', 'EDGEXS', 'RG_W'):
            collections.append(getattr(engine.M, name, None))
        for collection in collections:
            for value in tensors(collection, torch):
                if id(value) in seen:
                    continue
                seen.add(id(value))
                value.requires_grad_(False)
                value.grad = None
                self.frozen.append(value)
        engine.TRAINABLE = []  # late leaves never enter T/set_parameters' 28-array ABI
        self.cards = []

    def install_card(self, cd):
        """Wrap the final GNN, outside HEADF, without replacing cd.P."""
        assert cd.P is self.original
        assert cd.P['enc0_w'] is self.E.p['enc0_w']
        assert '_astra_current_w3_contract' in cd.env
        assert not hasattr(cd, '_o18_late_encoder_scope')
        parent = cd.env['gnn_step']
        leaves = self.leaves

        def late_gnn_step(l, x, sigma, W, state, P):
            # The original HEADF wrapper subsequently adds z @ HEADU to this
            # selected enc0_b, exactly matching native _hc_bias(l, late_bias).
            if 12 <= l <= 15:
                P = dict(P, **leaves)
            return parent(l, x, sigma, W, state, P)

        cd.env['gnn_step'] = late_gnn_step
        old_step = cd.step
        scope = self

        def scoped_step(layer, state):
            # Do not use no_grad here: current M requires its forward replay
            # records to be populated. Frozen inputs/parameters already give a
            # graph-free prefix, while grad mode remains enabled for records.
            if layer <= 11:
                assert not any(x.requires_grad for x in tensors(state, scope.t))
            result = old_step(layer, state)
            if layer <= 11:
                assert not any(x.requires_grad for x in tensors(result, scope.t))
            return result

        cd.step = scoped_step
        cd._o18_late_encoder_scope = self
        self.cards.append(cd)

    def gradients(self):
        result = {}
        for key, leaf in self.leaves.items():
            assert leaf.grad is not None and bool(self.t.isfinite(leaf.grad).all()), key
            result['late.' + key] = leaf.grad.detach().clone()
        assert all(value.grad is None and not value.requires_grad for value in self.frozen)
        assert {key: digest(self.original[key]) for key in SHAPES} == self.original_hashes
        assert any(bool(value.abs().max() > 0) for value in result.values())
        return result

    def arrays(self):
        """Exact six-array NPZ names consumed by export_late_arrays_v1.py."""
        return {key: leaf.detach().cpu().contiguous().numpy().copy()
                for key, leaf in self.leaves.items()}
