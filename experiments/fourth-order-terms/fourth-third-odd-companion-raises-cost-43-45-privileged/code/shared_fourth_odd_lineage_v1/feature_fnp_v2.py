"""V2: identical arithmetic, with isolated companion kernel-view metadata."""
from feature_fnp import (advance_u, epsilon_from_aux, gate_u, mean_feature,
                         observed_push, odd_diagonal, pair_gamma)
from feature_fnp import OddLineageTap as OriginalOddLineageTap


class CompanionViews:
    """Share paid parent arena/views; never initialize a parent metadata cache.

    Reshape/unstack view creation is billed by this runtime. A helper may use
    views that the parent has already paid for, and keeps its own newly paid
    views for this capture. Its cache additions never discount future parent
    work. No numeric transform, result cache, or W5 role grant is introduced.
    """
    NAMES = ('_VIEW_SPECS', '_COST_RULE_CACHE', '_PACKED_FRAME_CACHE',
             '_A_FRAME_CACHE', '_B_FRAME_CACHE', '_O_FRAME_CACHE',
             '_READ5_CHILD_CACHE', '_READ5_FRAME_CACHE')

    def __init__(self, mod):
        self.mod = mod
        self.private = {name:{} for name in self.NAMES}
        self.scratch = None
        self.active = False

    def call(self, function, *args, **kwargs):
        assert not self.active
        self.active = True
        parent = {name:getattr(self.mod, name) for name in self.NAMES}
        original_scratch = self.mod._SCRATCH
        before = {name:tuple((k, id(v)) for k, v in value.items()) for name, value in parent.items()}
        if self.scratch is None:
            self.scratch = list(original_scratch)
        assert self.scratch[0] is original_scratch[0]
        for index, value in enumerate(original_scratch):
            if value is not None:
                self.scratch[index] = value
        for name in self.NAMES:
            # Parent entries have already paid their initialization charge.
            self.private[name].update(parent[name])
            setattr(self.mod, name, self.private[name])
        self.mod._SCRATCH = self.scratch
        try:
            return function(*args, **kwargs)
        finally:
            self.scratch = self.mod._SCRATCH
            for name in self.NAMES:
                self.private[name] = getattr(self.mod, name)
                setattr(self.mod, name, parent[name])
                assert tuple((k, id(v)) for k, v in parent[name].items()) == before[name]
            self.mod._SCRATCH = original_scratch
            self.active = False

    def retire(self):
        assert not self.active
        self.private.clear()
        self.scratch = None


class OddLineageTap(OriginalOddLineageTap):
    def __init__(self, mod, ctx, observed):
        super().__init__(mod, ctx, observed)
        self.views = CompanionViews(mod)

    def advance(self):
        return self.views.call(super().advance)

    def detach(self):
        super().detach()
        self.views.retire()
