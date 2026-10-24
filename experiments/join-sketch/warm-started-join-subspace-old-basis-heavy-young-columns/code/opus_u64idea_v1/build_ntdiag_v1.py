"""[opus-u64idea] DIAGNOSTIC (not deployable): NTB32 (cand_s2tr_ntb32_v2 9b7aab59) + flop probes at each join: billed cost of the actual
(rank+p)-column sketch product and of discarded alternative sketch products (384 sorted / 416 / 448 / 32 / 64 columns; J part and old-tier
part), of the NT truncation and of Fj. Probes read flopscope's active budget (flops_used); the probe products are discarded."""
import hashlib
from pathlib import Path
D = Path(__file__).resolve().parent
src = (D/'cand_s2tr_ntb32_v2.py').read_text()
assert hashlib.sha256(src.encode()).hexdigest().startswith('9b7aab59')
fn = src.index('def _astra_bcc_push_impl(')


def patch(old, new):
    global src
    i = src.rindex(old); assert i > fn and src.index('\ndef ', fn + 10) > i, old[:50]
    src = src[:i] + new + src[i + len(old):]


patch("""            Y = _shared_old384_mm(J, _e2_H(_e2_H(weight[:, None]) * _e2_H(_e2_H(J[_jn_rows, :]).T)))
            if old_columns:
                Y = Y + _e2_H(_shared_old384_mm(Qp, _e2_H(_shared_old384_mm(state['core'], _e2_H(_e2_H(Qp[_jn_rows, :]).T)))))
            del _jn_d""", """            _ntd_alt(layer, J, weight, Qp if old_columns else None, state['core'] if old_columns else None, _jn_d, rank)
            _ntd_f0 = _ntd_flops()
            Y = _shared_old384_mm(J, _e2_H(_e2_H(weight[:, None]) * _e2_H(_e2_H(J[_jn_rows, :]).T)))
            if old_columns:
                Y = Y + _e2_H(_shared_old384_mm(Qp, _e2_H(_shared_old384_mm(state['core'], _e2_H(_e2_H(Qp[_jn_rows, :]).T)))))
            _ntd_log(layer, 'Yactual%d' % int(Y.shape[1]), _ntd_flops() - _ntd_f0)
            del _jn_d""")
patch("""        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
        core = None
        if _nt_p:
            Qn, core = _nt_truncate(layer, rank, Qn, unused_R, _jn_rows, _need_core)""", """        _ntd_f0 = _ntd_flops()
        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
        _ntd_log(layer, 'QR%d' % int(Y.shape[1]), _ntd_flops() - _ntd_f0)
        core = None
        if _nt_p:
            _ntd_f0 = _ntd_flops()
            Qn, core = _nt_truncate(layer, rank, Qn, unused_R, _jn_rows, _need_core)
            _ntd_log(layer, 'TRUNC', _ntd_flops() - _ntd_f0)""")
patch("""        Fj = _shared_old384_mm(_e2_H(Qn.T), J)""", """        _ntd_f0 = _ntd_flops()
        Fj = _shared_old384_mm(_e2_H(Qn.T), J)
        _ntd_log(layer, 'Fj', _ntd_flops() - _ntd_f0)""")
src += '''

# BEGIN OPUS-U64IDEA NT DIAGNOSTIC PROBES (not deployable)
from flopscope._budget import get_active_budget as _ntd_budget


def _ntd_flops():
    return int(_ntd_budget().flops_used)


def _ntd_log(layer, label, f):
    print('[ntd] layer %d %s %d' % (layer, label, f), file=_nt_sys.stderr, flush=True)


def _ntd_alt(layer, J, weight, Qp, core_old, d, rank):
    order = fnp.argsort(-d.astype(fnp.float64), kind='stable')
    for label, rows in (('384sorted', fnp.sort(order[:rank])), ('416', order[:rank + 32]), ('448', order[:rank + 64]),
                        ('32', order[rank:rank + 32]), ('64', order[rank:rank + 64])):
        f0 = _ntd_flops()
        Y = _shared_old384_mm(J, (weight[:, None] * (J[rows, :]).T))
        f1 = _ntd_flops()
        _ntd_log(layer, 'J' + label, f1 - f0)
        if Qp is not None:
            Z = _shared_old384_mm(Qp, _shared_old384_mm(core_old, (Qp[rows, :]).T))
            _ntd_log(layer, 'O' + label, _ntd_flops() - f1)
            del Z
        del Y
# END OPUS-U64IDEA NT DIAGNOSTIC PROBES
'''
o = D/'cand_s2tr_ntb32diag_v1.py'
with o.open('x') as f:
    f.write(src)
compile(o.read_text(), str(o), 'exec'); print(o.name, hashlib.sha256(o.read_bytes()).hexdigest()[:8])
