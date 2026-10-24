"""Read-only, metered quartic companion to actual stored cubic pair factors."""
import flopscope.numpy as fnp


def epsilon_from_aux(aux):
    return -aux['alpha'] * aux['pdf'] / (aux['sigma'] * aux['sigma'])


def pair_gamma(epsilon, selected, t, s):
    gap = t.astype(fnp.float64) - s.astype(fnp.float64)
    nonzero = gap != 0
    safe = fnp.where(nonzero, gap, 1.)
    return fnp.where(nonzero, epsilon[selected] / safe, 0.)


def paired_fourth_diagonal(p, cohorts):
    total = fnp.zeros(int(p.shape[0]), dtype=fnp.float64)
    stop = 0
    for start, gamma in cohorts:
        count = int(gamma.shape[0])
        assert start == stop and count > 0
        stop = start + 2 * count
        plus = p[:, start:start+count].astype(fnp.float64)
        minus = p[:, start+count:stop].astype(fnp.float64)
        quartic_difference = (plus-minus)*(plus+minus)*(plus*plus+minus*minus)
        total = total + fnp.sum(quartic_difference*gamma[None, :], axis=1)
    assert stop == int(p.shape[1])
    return total


def terminal_feature(aux, q4):
    sigma = aux['sigma']
    variance = sigma*sigma
    shape = sigma*(aux['alpha']*aux['alpha']-1)*aux['pdf']/24
    feature = shape*q4/(variance*variance)
    return shape, feature


class PairFourthTap:
    NAMES = ('_nonlinear_first_gaussian', 'nonlinear', '_shared_old384_push',
             'nonlinear_scalar', 'gnn_step')

    def __init__(self, mod, ctx):
        self.mod, self.ctx = mod, ctx
        self.original = {name: getattr(mod, name) for name in self.NAMES}
        self.extra_flops = 0
        self.cohorts = []
        self.pending = None
        self.layer = -1
        self.held = {}

    def measure(self, fun, *args):
        before = self.ctx.summary_dict()['flops_used']
        value = fun(*args)
        self.extra_flops += self.ctx.summary_dict()['flops_used']-before
        return value

    def remember(self, result):
        assert self.pending is None and 0 <= self.layer <= 14
        self.pending = (self.layer, self.measure(epsilon_from_aux, result[-1]))
        return result

    def first(self, *args, **kwargs):
        return self.remember(self.original['_nonlinear_first_gaussian'](*args, **kwargs))

    def pair(self, *args, **kwargs):
        return self.remember(self.original['nonlinear'](*args, **kwargs))

    def push(self, layer, w32, p, previous, newborn, lam, state):
        self.layer = layer
        answer = self.original['_shared_old384_push'](layer, w32, p, previous, newborn, lam, state)
        if layer:
            assert self.pending is not None and self.pending[0] == layer-1
            start, selected, unused_base, t, s = newborn
            assert start == sum(2*int(g.shape[0]) for _, g in self.cohorts)
            gamma = self.measure(pair_gamma, self.pending[1], selected, t, s)
            assert start + 2*int(gamma.shape[0]) == int(lam.shape[0])
            self.cohorts.append((start, gamma))
            self.pending = None
        else:
            assert not self.cohorts and self.pending is None
        if layer == 15:
            assert len(self.cohorts) == 15
            self.held['q4'] = self.measure(paired_fourth_diagonal, answer, self.cohorts)
        return answer

    def scalar(self, *args, **kwargs):
        result = self.original['nonlinear_scalar'](*args, **kwargs)
        assert self.layer == 15 and 'q4' in self.held
        self.held['shape'], self.held['feature'] = self.measure(terminal_feature, result[1], self.held['q4'])
        return result

    def gnn(self, l, x, sigma, w, state, params):
        result = self.original['gnn_step'](l, x, sigma, w, state, params)
        if l == 15:
            self.held['hidden'] = result[1]
            self.held['sigma'] = sigma
        return result

    def attach(self):
        for name, value in zip(self.NAMES, (self.first, self.pair, self.push, self.scalar, self.gnn)):
            setattr(self.mod, name, value)

    def detach(self):
        for name, value in self.original.items():
            setattr(self.mod, name, value)
        assert all(getattr(self.mod, name) is value for name, value in self.original.items())
