"""v2 (trainer, written 2026-10-12T23:46:25Z; lead 23:3xZ / physicist): + the family k21k, a K21 scale on the KINK rows of the covariance-output knob
(cov_pm: dK21 = iso kink (k3 / v) Cpre + k21n (1 - kink) K21 + k21k kink K21, layers 2-14, base 0: the kink rows (|alpha| < 1.5) had only the
k3 form iso and no K21 scale; the always-on / off rows have k21n). k21k = 0 adds an exact zero (certified bit for bit with the other new
families before any use). Workers alias it as otr_pl (worker_generated_v15.py). v1 docstring follows.
TAG opus-trainer (2026-10-01): PER-LAYER versions of the physics dials on the torch card (lead's task of 20:45 / 21:20 / 22:05 UTC).
Every family is a vector over the 16 layers that is ADDED to today's value: effective value at layer l = today's value + EPS[family][l],
with EPS a zero vector (a torch leaf). With EPS = 0 the card is unchanged (certified bit for bit by pl_certify.py); the gradient of
the loss with respect to EPS[family][l] is the gradient of a per-layer dial initialised at today's global value. Families and where
they act (today's value in brackets):
  k4       per-unit k4 scale at its definition, pair layers 1-14 [bands k4a 1-7 / k4b 8-11 / k4c 12-14] and the final layer 15 [1]
  bscale   carried b x scale after the b-fix, layers 0-14 [dial bscale]
  s_carry  weights of the carried columns x scale, layers 1-14 [dial s_carry];   s_new: newborn columns, layers 0-14 [dial s_new]
  c0, c1   K31 row factor c0 + c1 mu k3 / v^2, shared by the pair map and the covariance-output K31 remainder, layers 1-14 [dials c0, c1]
  iso, k21n, k31   covariance-output knobs, layers 2-14 [-0.094, -0.028, -0.212];   edge: message edge x (1 + edge), layers 8-14 [0.164]
  rc       recentring strength 1 + rc, layers 1-14 [-0.08]
  X0, X1, X2   birth-leg dials, layers 0-14 [dials X0, X1, X2] (X3 is cut from the card: the j3 leg is not built)
  kappa    birth weight d (1 + kappa radial / 3), layers 1-14 [1];   bsrc: source of the b carrier, bsrc radial d v / 6, layers 1-14 [1]
How: the card's hooks are replaced by copies that read today's value + EPS[family][layer]; the birth-leg update is the card's own
make_bd source with three text replacements; kappa and bsrc need the card's step, whose source is taken from the class and patched
in two lines. Every use of a value gets its own addition node, so the gradients of the global dials accumulate as before.
A request may carry non-zero values (set_values): the card then runs with today's value + those shifts, for forward scans of a
per-layer or global change; without them the shifts are reset to zero.
usage (inside otr_engine when TRAINER_PL=1): otr_pl.install(engine module, card) after the star is installed; otr_pl.collect() after
backward -> {family: 16 gradients}; otr_pl.set_values({family: 16 shifts}) before a run."""
import inspect
import sys
import textwrap
import types

import torch

FAMILIES = ('k4', 'bscale', 's_carry', 's_new', 'c0', 'c1', 'iso', 'k21n', 'k31', 'edge', 'rc', 'X0', 'X1', 'X2', 'kappa', 'bsrc', 'k21k')      # v2: + k21k (the K21 scale on the kink rows)
GLOBAL = dict(k4={'k4a': range(1, 8), 'k4b': range(8, 12), 'k4c': range(12, 15)}, bscale={'bscale': range(16)},
              s_carry={'s_carry': range(16)}, s_new={'s_new': range(16)}, c0={'c0': range(16)},
              X0={'X0': range(16)}, X1={'X1': range(16)}, X2={'X2': range(16)})   # family -> global dial -> layers it sums over
EPS = {family: torch.zeros(16, dtype=torch.float64, requires_grad=True) for family in FAMILIES}
_STEP = {}


def set_values(values):
    """Shifts for the next run: {family: 16 numbers}; families left out go back to zero."""
    assert set(values) <= set(FAMILIES), sorted(values)
    with torch.no_grad():
        for family, leaf in EPS.items():
            leaf.copy_(torch.tensor(values.get(family, [0.0] * 16), dtype=torch.float64))


def _patched_step(card_class):
    """The card's step with a per-layer kappa in the birth weight and a per-layer scale on the b source (two lines replaced)."""
    if card_class not in _STEP:
        source = textwrap.dedent(inspect.getsource(card_class.step))
        for old, new in (('birth = d * (1 + radial / 3)\n', 'birth = d * (1 + self.pl_kappa(layer) * radial / 3)\n'),
                         ('b = g * bpre + radial * d * vv / 6\n', 'b = g * bpre + self.pl_bsrc(layer) * radial * d * vv / 6\n')):
            assert source.count(old) == 1, old
            source = source.replace(old, new)
        namespace = dict(vars(sys.modules[card_class.__module__]))
        exec(compile(source, '<%s.step with per-layer kappa and b source>' % card_class.__name__, 'exec'), namespace)
        _STEP[card_class] = namespace['step']
    return _STEP[card_class]


def install(E, cd):
    D, V3 = E.DL, E.LC.JT.gn3.V3
    for value in EPS.values():
        value.grad = None
    eps = EPS
    layer = lambda: cd.call_layer

    # 1. k4: the pair call and ctx['k4'] (read by the covariance-output hook) both see the scaled k4; final layer separately
    band = lambda l: D['k4a'] if l <= 7 else (D['k4b'] if l <= 11 else D['k4c'])
    def pair_in(l, pin, ctx):
        if l == 0:
            return pin
        ctx['k4'] = ctx['k4'] * (band(l) + eps['k4'][l])
        return (pin[0], pin[1], pin[2], pin[3] * (band(l) + eps['k4'][l]), pin[4], pin[5])
    cd.hooks['pair_in'] = pair_in
    cd.hooks['k4_15'] = lambda l, k4, ctx: k4 * (1.0 + eps['k4'][15])

    # 2. carried b and column weights
    cd.hooks['b_post'] = lambda l, b, ctx: b * (D['bscale'] + eps['bscale'][l])
    def state_out(l, st, ctx):
        if l == 15:
            return st
        lam, stops = st[3], st[12]
        n_old = stops[-2] if len(stops) >= 2 else 0
        return st[:3] + (torch.cat((lam[:n_old] * (D['s_carry'] + eps['s_carry'][l]),
                                    lam[n_old:] * (D['s_new'] + eps['s_new'][l]))),) + st[4:]
    cd.hooks['state_out'] = state_out

    # 3. K31 row factor, one function for the pair map and for the covariance-output remainder
    c0, c1 = cd.env['_ROWF'][0].__defaults__
    cd.env['_ROWF'][0] = lambda mu, k3, v: (c0 + eps['c0'][layer()]) + (c1 + eps['c1'][layer()]) * mu * k3 / (v * v)

    # 4. covariance-output knobs: gn_linear_hook.install's hook with per-layer strengths and the row factor read at call time
    derivatives = cd.env['derivatives']
    t = torch
    def cov_pm(l, C, ctx):
        if l < 2 or l > 14:
            return C
        iso, k21n, k31 = V3['iso'] + eps['iso'][l], V3['k21n'] + eps['k21n'][l], V3['k31'] + eps['k31'][l]
        k21k = eps['k21k'][l]      # v2: base 0 (the kink rows had no K21 scale)
        rowf = cd.env['_ROWF'][0]
        mu, Cpre, K21, k4, aux = ctx['mu'], ctx['Cpre'], ctx['k21'], ctx['k4'], ctx['aux']
        s = aux['sigma']; a = aux['alpha']
        v = t.diagonal(Cpre)
        A, _unused = derivatives(a, aux['pdf'], aux['gate'], aux['gaussian_mean'] / s, 5)
        A1, A2, A3, A4, A5 = [x.to(t.float32) for x in A[1:6]]
        ss = t.outer(s, s)
        rho = (Cpre / ss).to(t.float32)
        rho = rho - t.diag(t.diagonal(rho))
        rho2 = rho * rho / 2
        k3 = t.diagonal(K21)
        kink = (a.abs() < 1.5).to(t.float64)
        dK21 = ((iso * kink) * k3 / v)[:, None] * Cpre + (k21n * (1 - kink))[:, None] * K21 + (k21k * kink)[:, None] * K21      # v2: + k21k on the kink rows
        dT21 = (dK21 / (v[:, None] * s[None, :])).to(t.float32)
        t4 = k4.expand(v.shape) / v ** 2
        r31 = rowf(mu, k3, v) if rowf is not None else None
        dT31 = (k31 * (t4 * r31 if r31 is not None else t4)).to(t.float32)[:, None] * rho
        t21a = t.outer(A2, A1) + rho * t.outer(A3, A2) + rho2 * t.outer(A4, A3)
        t31a = t.outer(A3, A1) + rho * t.outer(A4, A2) + rho2 * t.outer(A5, A3)
        X = dT21 * t21a / 2 + dT31 * t31a / 6
        dcov = (X + X.T).to(t.float64) * ss
        dcov = dcov - t.diag(t.diagonal(dcov))
        return C + dcov
    cd.hooks['cov_pm'] = cov_pm
    cd.hooks['edge'] = lambda l, v, ctx: v * (1 + (V3['edge'] + eps['edge'][l]))
    def delta(l, value, ctx):   # the step reads knobs['recenter'] right after this hook
        cd.knobs['recenter'] = 1 + (V3['rc'] + eps['rc'][l])
        return value
    cd.hooks['delta'] = delta

    # 5. birth legs: the card's own make_bd source (with the j3 cut of the reference trainer) and per-layer X0, X1, X2
    source = E.bd_source
    for old, new in (('for j in range(4):', 'for j in range(3):'),
                     ("change = change + direction * D['X1']", "change = change + direction * (D['X1'] + EPS['X1'][LAYER()])"),
                     ('change = change + direction * c[j] + raw * add[j]',
                      "change = change + direction * c[j] + raw * (add[j] + EPS['X%d' % j][LAYER()])")):
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    namespace = dict(vars(E.LC.JT), EPS=eps, LAYER=layer)
    exec(compile(source, '<make_bd with per-layer X0, X1, X2>', 'exec'), namespace)
    cd.env['_bd_update'] = namespace['make_bd'](torch, D)

    # 6, 7. birth weight and b source inside the step
    cd.pl_kappa = lambda l: 1.0 + eps['kappa'][l]
    cd.pl_bsrc = lambda l: 1.0 + eps['bsrc'][l]
    cd.step = types.MethodType(_patched_step(type(cd)), cd)


def collect():
    return {family: (value.grad.clone() if value.grad is not None else value.detach().clone() * 0) for family, value in EPS.items()}
