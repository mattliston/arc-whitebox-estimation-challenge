"""TAG opus_engineer a15 K4L v6 unit check (light, no estimator): fragment v6 (the young cohorts' hub terms as one block of
operations, a strength per age folded into the coefficients at the birth) against fragment v5 on the same random data, once with
one strength for all ages and once with a strength per age:
  birth: the four coefficients per pair (v6's lazy tuple and its stacked array) against v5's; the per-age stacks against
         strength x v5's coefficients;
  local: the per-row sum at layer 1 (one cohort), layers 2 and 3 (two cohorts, with older columns in front) and the final
         layer's call without squares and cubes: v6 against v5 and against a float64 numpy evaluation;
  requests (logged operations + views, as the platform counts round trips) and bill of each.
usage: unit_k4l_v6.py [<fragment file>]   (fragment_k4l_v7.py: v5's arithmetic with the blocks by splits; its row sums must equal v5's bit for bit)"""
from pathlib import Path
import math
import sys
import numpy as np
import flopscope as flops
from flopscope import _ndarray
fnp = flops.numpy
HERE = Path(__file__).resolve().parent
VIEWS = [0]
_getitem = _ndarray.FlopscopeArray.__getitem__


def counted_getitem(self, key):
    VIEWS[0] += 1
    return _getitem(self, key)


_ndarray.FlopscopeArray.__getitem__ = counted_getitem


class _E2Hold(list):
    def __call__(self, x):
        self.append(x)
        return x


def load(name, strength):
    ns = dict(fnp=fnp, _E2Hold=_E2Hold)
    text = (HERE/name).read_text()
    assert text.count('_K4L_C = 1.0\n') == 1
    exec(compile(text.replace('_K4L_C = 1.0\n', '_K4L_C = %r\n' % (strength,)), name, 'exec'), ns)
    return ns


def measured(ctx, call):
    start = (len(ctx._op_log), VIEWS[0], int(ctx.summary_dict()['flops_used']))
    out = call()
    ops = list(ctx._op_log)[start[0]:]
    return out, len(ops) + VIEWS[0] - start[1] - sum(1 for x in ops if 'getitem' in x.op_name), int(ctx.summary_dict()['flops_used']) - start[2]


ok = True
for strength in (0.86, (0.9, 0.6)):
    NEW = sys.argv[1] if len(sys.argv) > 1 else 'fragment_k4l_v6.py'
    old, new = load('fragment_k4l_v5.py', strength), load(NEW, strength)
    ages = strength if isinstance(strength, tuple) else (1.0, 1.0)   # one number: the strength is applied by _k4l_dk in both versions
    rng = np.random.default_rng(1)
    n = 1024
    print('strength', strength)
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        roster, at = [], 0
        for l, K in ((0, 96), (1, 128), (2, 160), (3, 420)):
            alpha = rng.standard_normal(n) * (0.0 if l == 0 else 1.3)
            sigma = rng.uniform(0.6, 1.8, n)
            gate = np.array([0.5 * math.erfc(-x / math.sqrt(2)) for x in alpha]); pdf = np.exp(-alpha ** 2 / 2) / math.sqrt(2 * math.pi)
            host = sigma * (alpha * gate + pdf) * (1 + 0.03 * rng.standard_normal(n))
            t = rng.uniform(0.4, 3.0, n) * sigma ** 2; s = -rng.uniform(0.4, 3.0, n) * sigma ** 2
            sel = rng.permutation(n)[:K]
            aux = dict(alpha=fnp.asarray(alpha), sigma=fnp.asarray(sigma), gate=fnp.asarray(gate), pdf=fnp.asarray(pdf))
            args = (l, aux, fnp.asarray(host), fnp.asarray(sel), fnp.asarray(t), fnp.asarray(s))
            counts = []
            for ns in (old, new):
                ns['_k4l_const']()
                if 'one' not in ns['_K4L_CONST']:
                    ns['_K4L_CONST']['one'] = fnp.ones(n, dtype=fnp.float64)
                counts.append(measured(ctx, lambda: ns['_k4l_birth'](*args))[1:])
            a = np.stack([np.asarray(x) for x in old['_K4L_B'][l]])
            lazy = np.stack([np.asarray(x) for x in new['_K4L_B'][l]])
            raw = np.asarray(new['_K4L_RAW'][l]) if '_K4L_RAW' in new else lazy
            rel = np.abs(a.astype(np.float64) - raw.astype(np.float64)).max() / np.abs(a).max()
            per_age = max(float(np.abs(np.asarray(new['_K4L_E'][l][i], dtype=np.float64) - c * a.astype(np.float64)).max() / (abs(c) * np.abs(a).max())) for i, c in enumerate(ages)) if '_K4L_E' in new else 0.0
            ok = ok and rel < 1e-6 and per_age < 1e-6 and bool((lazy == raw).all())
            print('  birth layer %d, %d hubs: coefficients against v5 %.1e of the largest (%d of %d bit-identical), per-age stacks against strength x v5 %.1e; requests v5 %d, v6 %d; flops v5 %d, v6 %d'
                  % (l, K, rel, int((a == raw).sum()), a.size, per_age, counts[0][0], counts[1][0], counts[0][1], counts[1][1]))
            roster.append((at + 64 if l == 0 else at, 2 * K)); at = roster[-1][0] + 2 * K   # 64 older columns in front of the first cohort
        state = dict(cohorts=roster)
        for l, rows, with_powers in ((1, n, True), (2, n, True), (3, n, True), (4, 640, False)):
            pp = (rng.standard_normal((rows, at)) * 0.4).astype(np.float32)
            young = [j for j in range(max(l - 2, 0), l)]
            want = np.zeros(rows)
            for j in young:                                         # the pair's columns differ by a multiple of the hub's edge: D is small against p
                k0, K = roster[j][0], roster[j][1] // 2
                pp[:, k0 + K:k0 + 2 * K] = pp[:, k0:k0 + K] + (rng.standard_normal((rows, K)) * 0.05).astype(np.float32)
                p = pp[:, k0:k0 + K].astype(np.float64); D = p - pp[:, k0 + K:k0 + 2 * K].astype(np.float64)
                e1, e2, e3, e4 = (np.asarray(x, dtype=np.float64) for x in old['_K4L_B'][j])
                want += ages[l - j - 1] * (D * (e1 * p ** 3 + D * (e2 * p ** 2 + D * (e3 * p + e4 * D)))).sum(1)
            P = fnp.asarray(pp); P2 = P * P; P3 = P2 * P
            got = []
            for ns in (old, new):
                got.append(measured(ctx, lambda: ns['_k4l_local'](l, P, P2 if with_powers else None, P3 if with_powers else None, state)))
            (a, ra, fa), (b, rb, fb) = got
            same_bits = bool((np.asarray(a) == np.asarray(b)).all())
            a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
            scale = np.abs(want).max()
            d_new, d_old, between = np.abs(b - want).max() / scale, np.abs(a - want).max() / scale, np.abs(a - b).max() / scale
            pairs = sum(roster[j][1] // 2 for j in young)
            ok = ok and d_new < max(2 * d_old, 2e-6) and between < 5e-6
            print('  local at layer %d (%d cohort%s, %d rows, squares and cubes %s): new against float64 %.1e of the largest row sum (v5 %.1e), new against v5 %.1e (%s); requests v5 %d, new %d; flops per element of rows x pairs v5 %.1f, new %.1f'
                  % (l, len(young), 's' if len(young) > 1 else '', rows, 'given' if with_powers else 'not given', d_new, d_old, between, 'bit-identical' if same_bits else 'not bit-identical', ra, rb, fa / (rows * pairs), fb / (rows * pairs)))
print('PASS' if ok else 'FAIL')
