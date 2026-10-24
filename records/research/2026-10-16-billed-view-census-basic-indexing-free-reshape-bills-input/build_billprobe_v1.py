"""[opus-viewbill] DIAGNOSTIC (not deployable): ~S3 (release's S1b preview bd238e1c + rqs2 piece 5 + _GSAFE5_LIMIT = 0 = b8d531a3)
+ a wrapper on BudgetContext._charge_op recording, per network, (op, candidate call site function:line) -> (calls, flop_cost x op weight
x dtype rate). Dumped at exit to $BP_OUT (json). Question: which billed view / data-movement sites (reshape, getitem, copyto, ...) carry
the bill, and which of them a free view op (unstack / split / transpose / moveaxis / expand_dims / squeeze / broadcast_to) could replace."""
import hashlib
from pathlib import Path
D = Path(__file__).resolve().parent; R = D.parent
src = (R/'opus_rqs2_v1/cand_s1bp5_g0_v3.py').read_text(); assert hashlib.sha256(src.encode()).hexdigest().startswith('b8d531a3')
frag = '''

# BEGIN OPUS-VIEWBILL PROBE (diagnostic only)
import sys as _bp_sys, os as _bp_os, json as _bp_json, atexit as _bp_atexit, collections as _bp_col
from flopscope import _budget as _bp_budget
_BP = dict(net=-1, rows=_bp_col.defaultdict(lambda: [0, 0.0]))
_BP_FILE = __file__ if '__file__' in globals() else None
_BP_W = None
_BP_CHARGE = _bp_budget.BudgetContext._charge_op


def _bp_charge(self, op_name, flop_cost, subscripts, shapes, *a, **kw):
    global _BP_W
    if _BP_W is None:
        import flopscope as _fs, json as _j
        from pathlib import Path as _P
        d = _j.loads((_P(_fs.__file__).parent/'data'/'default_weights.json').read_text())
        _BP_W = (d['weights'], d['dtype_rates'])
    w, rates = _BP_W
    dts = kw.get('dtypes') or ()
    rate = max([rates.get(str(t), 1.0) for t in dts] or [1.0])
    f = _bp_sys._getframe(1); site = '?'
    while f is not None:
        if f.f_code.co_filename == _BP_FILE:
            site = '%s:%d' % (f.f_code.co_name, f.f_lineno); break
        f = f.f_back
    e = _BP['rows'][(_BP['net'], op_name, site)]
    e[0] += 1; e[1] += float(flop_cost) * float(w.get(op_name, 1.0) or 0.0) * float(rate)
    return _BP_CHARGE(self, op_name, flop_cost, subscripts, shapes, *a, **kw)


_bp_budget.BudgetContext._charge_op = _bp_charge
_BP_PREDICT = Estimator.predict


def _bp_predict(self, *args, **kwargs):
    _BP['net'] += 1
    return _BP_PREDICT(self, *args, **kwargs)


Estimator.predict = _bp_predict


def _bp_dump():
    p = _bp_os.environ.get('BP_OUT')
    if p:
        with open(p, 'w') as fh:
            _bp_json.dump([[k[0], k[1], k[2], v[0], v[1]] for k, v in _BP['rows'].items()], fh)


_bp_atexit.register(_bp_dump)
# END OPUS-VIEWBILL PROBE
'''
o = D/'cand_s3ish_billprobe_v1.py'
with o.open('x') as f: f.write(src + frag)
compile(o.read_text(), str(o), 'exec'); print(o.name, hashlib.sha256(o.read_bytes()).hexdigest()[:8])
