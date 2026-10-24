"""TAG opus_engineer a42 hc unit check (light, small arrays, no estimator): the fragment's column against a numpy transcription of the
trainer's definition (O/trainer/otr_m.py columns, column 6: least squares over the response rows on four regressors, residual over
tr C v, standardised over the rows with n - 1, clipped at 8, 0 elsewhere), for both layouts of the fold's regression rows (with the
constant: target row 4; table regression: target row 3), with outliers that reach the clip; the first layer's bias term enc0_b + z u';
the used-once rule; requests and bill of one layer.
usage: unit_hc.py"""
from pathlib import Path
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


rng = np.random.default_rng(11)
n, nrows = 256, 200
rows = np.sort(rng.choice(n, nrows, replace=False)); mask = np.zeros(n); mask[rows] = 1.0
v = rng.uniform(0.4, 2.0, n); mu = rng.standard_normal(n) * 0.7
A = rng.standard_normal((n, 40)); C = A @ A.T / 40 * 0.1; np.fill_diagonal(C, v); cmu = C @ mu; trC = float(np.trace(C))
diag = 3e-4 * (0.8 * trC * v + 0.3 * mu * mu - 0.1 * 2 * mu * cmu + 0.05 + 0.02 * rng.standard_normal(n) * v)      # kap x the matrix diagonal
diag[rows[:3]] += 3e-4 * np.array([4.0, -5.0, 6.0]); diag[rows[3:5]] += 3e-4 * np.array([0.9, -1.4])                                                                 # outliers that reach the clip
diag = diag * mask


def reference():
    X = np.stack([trC * v, np.ones(n), mu * mu, 2 * mu * cmu], 1)[rows]; y = (diag / 3e-4)[rows]          # the torch side has B_ii itself (no kap)
    resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    x = resid / (trC * v[rows])
    z = np.zeros(n); z[rows] = np.clip((x - x.mean()) / x.std(ddof=1), -8, 8)
    return z.astype(np.float32)


ok = True
u = (rng.standard_normal(128) * 0.02).astype(np.float32)
bias = rng.standard_normal(128).astype(np.float32)
text = (HERE/'fragment_hc_v1.py').read_text()
assert text.count('_HC_U = None ') == 1
for target in (4, 3):
    ns = dict(fnp=fnp, _E2Hold=_E2Hold)
    exec(compile(text.replace('_HC_U = None ', '_HC_U = %r ' % (tuple(float(x) for x in u),)), 'fragment_hc_v1.py', 'exec'), ns)
    regressors = [mask * v] + ([mask] if target == 4 else []) + [mask * mu * mu, mask * 2 * mu * cmu]
    Yn = np.stack(regressors + [mask * diag, mask * mu / v, np.ones(n), mask])                              # then rows the fold uses for its own outputs
    with flops.BudgetContext(flop_budget=2**40, quiet=True) as ctx:
        Y, vc, fm = fnp.asarray(Yn), fnp.asarray(v), fnp.asarray(mask)
        ns['_hc_column'](5, Y, target, vc, fm, nrows)
        ns['_hc_bias'](5, fnp.asarray(bias))
        mark = (len(ctx._op_log), VIEWS[0], int(ctx.summary_dict()['flops_used']))
        ns['_hc_column'](5, Y, target, vc, fm, nrows)
        z = np.asarray(ns['_HC_STATE']['z'])
        b = fnp.asarray(bias)
        out = ns['_hc_bias'](5, b)
        log = list(ctx._op_log)[mark[0]:]
        requests = len(log) + VIEWS[0] - mark[1] - sum(1 for x in log if 'getitem' in x.op_name) - 1        # 1 = this script's asarray of the bias
        used = int(ctx.summary_dict()['flops_used']) - mark[2]
        again = ns['_hc_bias'](5, b)                                                                        # used once: the second call returns the plain bias
        other = (ns['_hc_column'](5, Y, target, vc, fm, nrows), ns['_hc_bias'](6, b))[1]                    # a column of layer 5 is not used by layer 6
        ns['_hc_column'](0, Y, target, vc, fm, nrows); none0 = ns['_HC_STATE']['z'] is None
        ns['_hc_column'](15, Y, target, vc, fm, nrows); none15 = ns['_HC_STATE']['z'] is None
    zr = reference()
    e_z = float(np.abs(z - zr).max())
    e_b = float(np.abs(np.asarray(out) - (bias[None, :] + np.outer(zr, u))).max())
    clipped = int((np.abs(zr) == 8).sum())
    good = e_z < 2e-5 and e_b < 1e-6 and again is b and other is b and none0 and none15 and str(ns['_HC_STATE']['z']) == 'None' and clipped >= 1 and int((np.abs(zr) > 4).sum()) >= 3 and z.dtype == np.float32
    ok &= bool(good)
    print(f'regression rows with target {target}: column {e_z:.1e} from the reference (largest entry {np.abs(zr).max():.1f}, {clipped} units at the clip, {int((zr != 0).sum())} of {n} non-zero), '
          f'bias term {e_b:.1e}; used once {again is b}, not by another layer {other is b}, no column at layers 0 and 15 {none0 and none15}; requests {requests}, bill {used} flops' + ('' if good else '   <-- FAIL'))
print('PASS' if ok else 'FAIL')
