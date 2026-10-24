"""TAG opus_engineer a42 hcf unit check (light, small arrays, no estimator): the final-layer column 'k4_local15' and the bias term of
the final head against a numpy transcription of the rule (standardised over the kept rows with n - 1, clipped at 8, 0 elsewhere;
b0 + Z' U); a missing column counts as 0; without weights nothing changes; requests and bill of one call.
usage: unit_hcf.py"""
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


rng = np.random.default_rng(7)
n, kept = 256, 205
rows = np.sort(rng.choice(n, kept, replace=False))
v = rng.uniform(0.4, 2.0, n)
dk = np.zeros(n); dk[rows] = rng.standard_normal(kept) ** 5 * v[rows] ** 2 * 1e-3      # heavy tail: reaches the clip
U = (rng.standard_normal((2, 128)) * 0.02).astype(np.float32); b0 = rng.standard_normal(128).astype(np.float32)


def load(columns, U_):
    text = (HERE/'fragment_hcf_v1.py').read_text()
    assert text.count('_HCF_COLUMNS = () ') == 1 and text.count('_HCF_U = None ') == 1
    if U_ is not None:
        text = text.replace('_HCF_COLUMNS = () ', '_HCF_COLUMNS = %r ' % (columns,)).replace('_HCF_U = None ', '_HCF_U = %r ' % (tuple(tuple(float(x) for x in u) for u in U_),))
    ns = dict(fnp=fnp, _E2Hold=_E2Hold, _K4L_FINAL=[None], _k4l_final=lambda mu, v: 0.0, _hc_bias=lambda l, bias: bias)   # stand-ins for the K4L and HC fragments
    exec(compile(text, 'fragment_hcf_v1.py', 'exec'), ns)
    return ns


ok = True
with flops.BudgetContext(flop_budget=2**40, quiet=True) as ctx:
    b = fnp.asarray(b0)
    off = load((), None)
    off['_hcf_k4'](fnp.asarray(dk), fnp.asarray(rows), fnp.asarray(v))
    untouched = off['_hc_bias'](15, b) is b
    one = load(('k4_local15', 'other'), U)
    for _ in range(2):
        mark = (len(ctx._op_log), VIEWS[0], int(ctx.summary_dict()['flops_used']))
        one['_hcf_k4'](fnp.asarray(dk), fnp.asarray(rows), fnp.asarray(v))
        not14 = one['_hc_bias'](14, b) is b                     # other layers: the columns stay for layer 15
        out = np.asarray(one['_hc_bias'](15, b))
        log = list(ctx._op_log)[mark[0]:]
        requests = len(log) + VIEWS[0] - mark[1] - sum(1 for x in log if 'getitem' in x.op_name) - 3
        used = int(ctx.summary_dict()['flops_used']) - mark[2]
    again = one['_hc_bias'](15, b) is b
x = dk / (v * v); z = np.zeros(n); y = x[rows]; z[rows] = np.clip((y - y.mean()) / y.std(ddof=1), -8, 8)
Z = np.stack([z, np.zeros(n)]).astype(np.float32)
want = b0[None, :] + Z.T @ U
err = float(np.abs(out - want).max()); clipped = int((np.abs(z) == 8).sum())
ok = untouched and again and not14 and err < 2e-6 and clipped >= 1
print(f'final-layer column k4_local15 + a missing column: {err:.1e} from the reference ({clipped} units at the clip); without weights untouched {untouched}; '
      f'used once {again}; requests {requests}, bill {used} flops')
print('PASS' if ok else 'FAIL')
