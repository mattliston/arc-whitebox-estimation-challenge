"""TAG opus_engineer a43 ro unit check (light, small arrays, no estimator): the readouts (o2) carried b and (o3) radial against a numpy
transcription of the trainer's lines (O/trainer/otr_m.py, READOUTS: b + sigma * rms(b / sigma) * (state @ B[l]); radial * (1 + mean(state)
@ RAD[l])); off without weights; only for the layer whose state gnn_step recorded; requests and bill of one layer.
usage: unit_ro.py"""
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


rng = np.random.default_rng(3)
n = 256
state = rng.standard_normal((n, 64)).astype(np.float32)
A = rng.standard_normal((n, n)); C = A @ A.T / n + np.diag(rng.uniform(0.3, 2.0, n))
b = rng.standard_normal(n) * 0.05
B = rng.standard_normal((16, 64)) * 0.01; RAD = rng.standard_normal((16, 64)) * 0.01
radial = 0.0123


def load(withw):
    text = (HERE/'fragment_ro_v1.py').read_text()
    if withw:
        assert text.count('_RO_B = None ') == 1 and text.count('_RO_RAD = None ') == 1
        text = text.replace('_RO_B = None ', '_RO_B = %r ' % (tuple(tuple(float(x) for x in row) for row in B),)).replace('_RO_RAD = None ', '_RO_RAD = %r ' % (tuple(tuple(float(x) for x in row) for row in RAD),))
    ns = dict(fnp=fnp, _E2Hold=_E2Hold, gnn_step=lambda l, x, sigma, W, s, P: (None, fnp.asarray(state)))
    exec(compile(text, 'fragment_ro_v1.py', 'exec'), ns)
    return ns


ok = True
with flops.BudgetContext(flop_budget=2**40, quiet=True) as ctx:
    off = load(False); on = load(True)
    bb, CC = fnp.asarray(b), fnp.asarray(C)
    off['gnn_step'](5, None, None, None, None, None)
    same = off['_ro_b'](5, bb, CC) is bb and off['_ro_radial'](5, radial) == radial
    on['gnn_step'](5, None, None, None, None, None)
    other = on['_ro_b'](6, bb, CC) is bb                                       # the recorded state is layer 5's
    mark = (len(ctx._op_log), VIEWS[0], int(ctx.summary_dict()['flops_used']))
    nb = np.asarray(on['_ro_b'](5, bb, CC)); nr = float(np.asarray(on['_ro_radial'](5, radial)))
    log = list(ctx._op_log)[mark[0]:]
    requests = len(log) + VIEWS[0] - mark[1] - sum(1 for x in log if 'getitem' in x.op_name)
    used = int(ctx.summary_dict()['flops_used']) - mark[2]
s = np.sqrt(np.maximum(np.diag(C), 1e-12)); st = state.astype(np.float64)
want_b = b + s * np.sqrt(np.mean((b / s) ** 2)) * (st @ B[5]); want_r = radial * (1 + st.mean(0) @ RAD[5])
eb = float(np.abs(nb - want_b).max() / np.abs(want_b - b).max()); er = abs(nr - want_r) / abs(want_r - radial)
ok = same and other and eb < 1e-12 and er < 1e-12
print(f'carried b {eb:.1e} of its change, radial {er:.1e} of its change against the transcription; off without weights {same}; not for another layer {other}; '
      f'requests {requests} (both readouts, including the constants once), bill {used} flops')
print('PASS' if ok else 'FAIL')
