"""TAG opus_engineer a42 hc v2 unit check (light, small arrays, no estimator):
  (1) with the one column 'diag', v2's bias term is bit for bit v1's (same inputs, both layouts of the regression rows);
  (2) with three columns (the diagonal's residual on the response rows, two all-unit vectors) the bias term against a numpy
      transcription of the trainer's rule (each column standardised over its own units with n - 1, clipped at 8, 0 elsewhere; bias +
      Z' U), with units at the clip;
  (3) a column that a layer does not put counts as 0 there; a column is used once and never by another layer; requests and bill
      of one layer;
  (4) the helpers of columns 9, 12, 13 (_hc_pdiag, _hc_k3) against numpy.
usage: unit_hc_v2.py"""
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
diag = 3e-4 * (0.8 * trC * v + 0.3 * mu * mu - 0.1 * 2 * mu * cmu + 0.05 + 0.02 * rng.standard_normal(n) * v)
diag[rows[:3]] += 3e-4 * np.array([4.0, -5.0, 6.0]); diag = diag * mask
extra = [rng.standard_normal(n) ** 5, rng.standard_normal(n) * v]          # a heavy-tailed column (reaches the clip) and a plain one, all units
bias = rng.standard_normal(128).astype(np.float32)


def standard(x, sel):
    z = np.zeros(n); y = x[sel]; z[sel] = np.clip((y - y.mean()) / y.std(ddof=1), -8, 8)
    return z.astype(np.float32)


def diag_reference():
    X = np.stack([trC * v, np.ones(n), mu * mu, 2 * mu * cmu], 1)[rows]; y = (diag / 3e-4)[rows]
    resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    full = np.zeros(n); full[rows] = resid / (trC * v[rows])
    return standard(full, rows)


def load(name, columns, U):
    text = (HERE/name).read_text()
    ns = dict(fnp=fnp, _E2Hold=_E2Hold)
    if name.endswith('v1.py'):
        text = text.replace('_HC_U = None ', '_HC_U = %r ' % (tuple(float(x) for x in U[0]),))
    else:
        assert text.count("_HC_COLUMNS = ('diag',) ") == 1 and text.count('_HC_U = None ') == 1
        text = text.replace("_HC_COLUMNS = ('diag',) ", '_HC_COLUMNS = %r ' % (columns,)).replace('_HC_U = None ', '_HC_U = %r ' % (tuple(tuple(float(x) for x in u) for u in U),))
    exec(compile(text, name, 'exec'), ns)
    return ns


ok = True
U = (rng.standard_normal((3, 128)) * 0.02).astype(np.float32)
for target in (4, 3):
    regressors = [mask * v] + ([mask] if target == 4 else []) + [mask * mu * mu, mask * 2 * mu * cmu]
    Yn = np.stack(regressors + [mask * diag, mask * mu / v, np.ones(n), mask])
    with flops.BudgetContext(flop_budget=2**40, quiet=True) as ctx:
        Y, vc, fm, b = fnp.asarray(Yn), fnp.asarray(v), fnp.asarray(mask), fnp.asarray(bias)
        one, two = load('fragment_hc_v1.py', ('diag',), U[:1]), load('fragment_hc_v2.py', ('diag',), U[:1])
        one['_hc_column'](5, Y, target, vc, fm, nrows); two['_hc_column'](5, Y, target, vc, fm, nrows)
        a1, a2 = np.asarray(one['_hc_bias'](5, b)), np.asarray(two['_hc_bias'](5, b))
        same = np.array_equal(a1, a2)
        # three columns
        three = load('fragment_hc_v2.py', ('diag', 'heavy', 'plain'), U)
        maskb = fnp.asarray(mask.astype(bool))
        for _ in range(2):
            mark = (len(ctx._op_log), VIEWS[0], int(ctx.summary_dict()['flops_used']))
            three['_hc_column'](5, Y, target, vc, fm, nrows)
            three['_hc_put'](5, 'heavy', fnp.asarray(extra[0])); three['_hc_put'](5, 'plain', fnp.asarray(extra[1]), maskb, nrows)      # 'plain' counts on the response rows only, and is NOT zero outside them when put
            out = three['_hc_bias'](5, b)
            log = list(ctx._op_log)[mark[0]:]
            requests = len(log) + VIEWS[0] - mark[1] - sum(1 for x in log if 'getitem' in x.op_name) - 2      # 2 = this script's asarray calls
            used = int(ctx.summary_dict()['flops_used']) - mark[2]
        again = three['_hc_bias'](5, b)
        three['_hc_column'](6, Y, target, vc, fm, nrows); three['_hc_put'](6, 'heavy', fnp.asarray(extra[0]))
        lacking = np.asarray(three['_hc_bias'](6, b))                             # 'plain' is missing at layer 6: it counts as 0
        three['_hc_put'](5, 'heavy', fnp.asarray(extra[0])); stale = three['_hc_bias'](6, b)      # a column of layer 5 is not used by layer 6
        # the helpers of columns 12, 13 and 9
        cols, old = 60, 16
        pp = rng.standard_normal((n, cols)).astype(np.float32); lam = rng.standard_normal(cols).astype(np.float32)
        wc = (pp * pp * pp * lam[None, :]).astype(np.float32)
        before = wc.sum(1).astype(np.float64); after = before + 0.01 * rng.standard_normal(n)
        k3 = load('fragment_hc_v2.py', ('k3_young', 'k3_old', 'k31l_diag_rows'), U)
        k3['_hc_k3'](7, fnp.asarray(wc), old, fnp.asarray(before), fnp.asarray(after), vc, fnp.asarray(np.sqrt(v)))
        W = 24
        value = rng.standard_normal((nrows, W)).astype(np.float32); own = rng.standard_normal((nrows, W)).astype(np.float32); ly = rng.standard_normal((1, W)).astype(np.float32)
        k3['_hc_pdiag'](7, fnp.asarray(value), fnp.asarray(own), fnp.asarray(ly[0]), fnp.asarray(rows), fm, vc, n)
        got = {k: np.asarray(x[0]) for k, x in k3['_HC_STATE']['raw'].items()}
        counts = {k: (x[1] is None, x[2]) for k, x in k3['_HC_STATE']['raw'].items()}
    Z = np.stack([diag_reference(), standard(extra[0], np.arange(n)), standard(extra[1], rows)])
    want = bias[None, :] + Z.T @ U
    err = float(np.abs(np.asarray(out) - want).max())
    Z0 = Z.copy(); Z0[2] = 0.0
    err0 = float(np.abs(lacking - (bias[None, :] + Z0.T @ U)).max())
    clipped = int((np.abs(Z) == 8).sum())
    old_stored = wc[:, :old].sum(1).astype(np.float64); young = before - old_stored; vs = v * np.sqrt(v)
    pd = np.zeros(n); pd[rows] = (value * own) @ ly[0]
    e_k3 = max(float(np.abs(got['k3_young'] - young / vs).max() / np.abs(young / vs).max()), float(np.abs(got['k3_old'] - (after - young) / vs).max() / np.abs((after - young) / vs).max()))
    e_pd = float(np.abs(got['k31l_diag_rows'] - pd / (v * v)).max() / np.abs(pd / (v * v)).max())
    helpers = e_k3 < 1e-6 and e_pd < 1e-6 and counts['k3_young'] == (True, n) and counts['k31l_diag_rows'] == (False, nrows)
    good = same and err < 2e-6 and err0 < 2e-6 and again is b and stale is b and clipped >= 2 and helpers
    ok &= bool(good)
    print(f'regression rows with target {target}: one column, v2 equal to v1 bit for bit {same}; three columns {err:.1e} from the reference ({clipped} entries at the clip); '
          f'a missing column counts as 0: {err0:.1e}; used once {again is b}, not by another layer {stale is b}; helpers k3 {e_k3:.1e}, plane diagonal {e_pd:.1e}; '
          f'requests {requests}, bill {used} flops' + ('' if good else '   <-- FAIL'))
print('PASS' if ok else 'FAIL')
