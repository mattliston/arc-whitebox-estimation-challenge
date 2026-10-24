"""[opus-batchtree] DIAGNOSTIC (not deployable): S4 452bda56 + recorders for the batched-tree feasibility census:
 - every live product entry (k-split: _ks_product / _hz_ks_product / _ubx_ks_product; trees: _ipt_product_ab, _hzg_product, _hz7_product,
   _d7_product_kpad, _r7_product, _cp_d6_shared, _push6_padded, _mm) -> (net, seq, name, shape args, plan sizes if any, charged ops,
   caller) with nesting depth, so chunk passes and same-key runs can be grouped offline;
 - every forward transform (_ipt_forward_a_ab / _ipt_forward_b_ab / _hzt_forward_a / _hzd_forward_b) -> (net, seq, role, params, ops,
   enclosing product). Dumped at exit to $BT_OUT (json)."""
import hashlib
from pathlib import Path
D = Path(__file__).resolve().parent; R = D.parent
src = (R/'opus_passrel_v1/stack_v1/cand_s4.py').read_text(); assert hashlib.sha256(src.encode()).hexdigest().startswith('452bda56')
frag = '''

# BEGIN OPUS-BATCHTREE PROBE (diagnostic only)
import sys as _bt_sys, os as _bt_os, json as _bt_json, atexit as _bt_atexit
from flopscope import _budget as _bt_budget
_BT = dict(net=-1, seq=0, rows=[], stack=[], ops=[0])
_BT_CHARGE = _bt_budget.BudgetContext._charge_op


def _bt_charge(self, *a, **kw):
    _BT['ops'][0] += 1
    return _BT_CHARGE(self, *a, **kw)


_bt_budget.BudgetContext._charge_op = _bt_charge


def _bt_shape(x):
    s = getattr(x, 'shape', None)
    return list(map(int, s)) if s is not None else (x if isinstance(x, (int, float)) else None)


def _bt_wrap(name, parent, kind):
    def wrapped(*args, **kw):
        f = _bt_sys._getframe(1)
        caller = '%s:%d' % (f.f_code.co_name, f.f_lineno)
        _BT['seq'] += 1
        seq = _BT['seq']; o0 = _BT['ops'][0]
        enclosing = _BT['stack'][-1] if _BT['stack'] else None
        _BT['stack'].append((name, seq))
        try:
            res = parent(*args, **kw)
        finally:
            _BT['stack'].pop()
        info = [_bt_shape(a) for a in args[:7]]
        plan = None
        for a in args:
            if isinstance(a, tuple) and len(a) == 3 and isinstance(a[2], tuple):
                plan = [a[0], a[1], list(a[2])]
        _BT['rows'].append(dict(net=_BT['net'], seq=seq, kind=kind, name=name, args=info, plan=plan, ops=_BT['ops'][0] - o0,
                                caller=caller, enclosing=enclosing, depth=len(_BT['stack'])))
        return res
    return wrapped


for _bt_name in ('_ks_product', '_hz_ks_product', '_ubx_ks_product', '_ipt_product_ab', '_hzg_product', '_hz7_product', '_d7_product_kpad',
                 '_r7_product', '_cp_d6_shared', '_push6_padded', '_mm'):
    if _bt_name in globals():
        globals()[_bt_name] = _bt_wrap(_bt_name, globals()[_bt_name], 'product')
for _bt_name in ('_ipt_forward_a_ab', '_ipt_forward_b_ab', '_hzt_forward_a', '_hzd_forward_b', '_hzd_forward_a', '_hz_leaf', '_ks_unwind', '_ipt_leaf_unwind_ab'):
    if _bt_name in globals():
        globals()[_bt_name] = _bt_wrap(_bt_name, globals()[_bt_name], 'forward')
_ASTRA_DENSE_CONTEXT = tuple((_n, globals()[_n]) for _n, _f in _ASTRA_DENSE_CONTEXT)
_BT_PREDICT = Estimator.predict


def _bt_predict(self, *args, **kwargs):
    _BT['net'] += 1
    return _BT_PREDICT(self, *args, **kwargs)


Estimator.predict = _bt_predict


def _bt_dump():
    p = _bt_os.environ.get('BT_OUT')
    if p:
        with open(p, 'w') as fh:
            _bt_json.dump(dict(rows=_BT['rows'], region={str(k): list(v) for k, v in _IPT_REGION.items()}, arena=int(_SCRATCH[0].shape[0]) if _SCRATCH is not None else None), fh)


_bt_atexit.register(_bt_dump)
# END OPUS-BATCHTREE PROBE
'''
o = D/'cand_s4_btprobe_v2.py'
with o.open('x') as f: f.write(src + frag)
compile(o.read_text(), str(o), 'exec'); print(o.name, hashlib.sha256(o.read_bytes()).hexdigest()[:8])
