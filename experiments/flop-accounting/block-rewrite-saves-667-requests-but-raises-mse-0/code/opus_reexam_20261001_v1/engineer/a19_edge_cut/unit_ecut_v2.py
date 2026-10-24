"""TAG opus_engineer a19 edge cut unit check (light, small arrays, single thread, no estimator): the fragment's kept block against
the parent's full edge tensor on the same rows and columns, and what the dropped part weighs for Gaussian-distributed mu/sigma.
v2: the same check on fragment_ecut_v2.py, and its output against v1's bit for bit, with the requests (logged operations + index
calls) and the bill of both.
usage: unit_ecut_v2.py"""
from pathlib import Path
import math
import numpy as np
import flopscope as flops
from flopscope import _ndarray
fnp = flops.numpy
VIEWS = [0]
_getitem = _ndarray.FlopscopeArray.__getitem__


def counted_getitem(self, key):
    VIEWS[0] += 1
    return _getitem(self, key)


_ndarray.FlopscopeArray.__getitem__ = counted_getitem
_ndarray.FlopscopeArray.T = property(lambda self: (VIEWS.__setitem__(0, VIEWS[0] + 1), self.transpose())[1])
HERE = Path(__file__).resolve().parent


class _E2Hold(list):
    def __call__(self, x):
        self.append(x)
        return x


rng = np.random.default_rng(11)
n, ch = 384, 16
alpha = rng.standard_normal(n) * 1.3
gate = np.array([0.5 * math.erfc(-x / math.sqrt(2)) for x in alpha]); pdf = np.exp(-alpha ** 2 / 2) / math.sqrt(2 * math.pi)
alive = gate >= 0.01
resp = alive & (np.minimum(gate, 1 - gate) >= 0.01)
J = np.argsort(-gate, kind='stable')[:int(alive.sum())]                       # the live roster: descending gate
I = np.argsort(-np.where(alive, np.minimum(gate, 1 - gate), -1.0), kind='stable')[:int(resp.sum())]
left = (rng.standard_normal((n, ch)) * 0.8).astype(np.float32); right = (rng.standard_normal((n, ch)) * 0.8).astype(np.float32)
t = dict(bias=(rng.standard_normal(ch) * 0.3).astype(np.float32), output=(rng.standard_normal(ch) * 0.4).astype(np.float32))
Rm = (rng.standard_normal((n, n)) * 0.5).astype(np.float32); Sm = (rng.standard_normal((n, n)) * 0.5).astype(np.float32)
e = (rng.standard_normal((3, ch)) * 0.5).astype(np.float32)
pf, gf = pdf.astype(np.float32), gate.astype(np.float32)
# the parent's arithmetic on the full tensor
h = left[:, None, :] + right[None, :, :] + t['bias']
h = h + Rm[:, :, None] * e[0] + Sm[:, :, None] * e[1] + Sm.T[:, :, None] * e[2]
T = np.tanh(h) @ t['output']
raw_full = T * pf[:, None] * gf[None, :]
results = {}
for standin, name in ((False, 'fragment_ecut.py'), (True, 'fragment_ecut.py'), (False, 'fragment_ecut_v2.py'), (True, 'fragment_ecut_v2.py')):
    ns = dict(fnp=fnp, _E2Hold=_E2Hold, _E2_LAYER=dict(ecut_rows=fnp.asarray(I), ecut_cols=fnp.asarray(J)))
    exec(compile((HERE/name).read_text().replace('_ECUT_STANDIN = False', '_ECUT_STANDIN = %r' % standin), name, 'exec'), ns)
    k = dict(R=fnp.asarray(Rm), S=fnp.asarray(Sm), e0=fnp.asarray(e[0]), e1=fnp.asarray(e[1]), e2=fnp.asarray(e[2]))
    tt = {key: fnp.asarray(val) for key, val in t.items()}
    hold = []
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        operands = (fnp.asarray(left), fnp.asarray(right), fnp.asarray(pf), fnp.asarray(gf))
        mark = (len(ctx._op_log), VIEWS[0], int(ctx.summary_dict()['flops_used']))
        out32 = ns['_ecut_raw'](operands[0], operands[1], tt, k, operands[2], operands[3], hold)
        log = list(ctx._op_log)[mark[0]:]
        used = int(ctx.summary_dict()['flops_used']) - mark[2]; ops = len(log)
        requests = ops + VIEWS[0] - mark[1] - sum(1 for x in log if 'getitem' in x.op_name)
        raw = np.asarray(out32, dtype=np.float64)
    results[(standin, name)] = (np.asarray(out32).copy(), requests, used)
    mask = np.zeros((n, n), dtype=bool); mask[np.ix_(I, J)] = True
    kept_err = np.abs(raw[mask] - raw_full[mask]).max() / np.abs(raw_full[mask]).max()
    full_bill = n * n * ch * 26
    print(f"{name}, stand-in {standin}: {requests} requests; kept block against the full tensor, max difference / max |raw| = {kept_err:.2e}; outside the block exactly 0: {bool((raw[~mask] == 0).all())};"
          f" rows {len(I)} of {n}, columns {len(J)} of {n}; billed {used / 1e6:.2f} Mflop against about {full_bill / 1e6:.2f} for the full tensor ({100 * used / full_bill:.0f}%); {ops} charged ops")
sym_full = 0.5 * (raw_full + raw_full.T)
cut = np.where(mask, raw_full, 0.0); sym_cut = 0.5 * (cut + cut.T)
print(f'dropped part of the symmetrised output: {100 * np.linalg.norm(sym_full - sym_cut) / np.linalg.norm(sym_full):.2f}% of its norm ({100 * ((sym_full - sym_cut) ** 2).sum() / (sym_full ** 2).sum():.3f}% of its energy);'
      f' largest dropped entry {np.abs(sym_full - sym_cut).max() / np.abs(sym_full).max():.3f} of the largest entry')
for standin in (False, True):
    (a, ra, fa), (b, rb, fb) = results[(standin, 'fragment_ecut.py')], results[(standin, 'fragment_ecut_v2.py')]
    print(f'stand-in {standin}: v2 against v1: {"bit-identical" if (a == b).all() and a.dtype == b.dtype else "NOT bit-identical"}; requests {ra} -> {rb}; bill {fa} -> {fb} flops ({fb - fa:+d})')
