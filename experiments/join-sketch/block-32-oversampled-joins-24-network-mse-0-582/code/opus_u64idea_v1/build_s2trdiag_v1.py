"""[opus-u64idea] DIAGNOSTIC (not deployable): S2tr 46e9315d + flop probes on TODAY's join (sketch product, QR, core solve, Fj) and a
per-layer bill trace (flops_used at each push entry), to price NT against the real baseline."""
import hashlib
from pathlib import Path
D = Path(__file__).resolve().parent; R = D.parent
src = (R/'opus_s2tr_v1/candidate_S2tr_v1.py').read_text()
assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
fn = src.index('def _astra_bcc_push_impl(')


def patch(old, new):
    global src
    i = src.rindex(old); assert i > fn and src.index('\ndef ', fn + 10) > i, old[:50]
    src = src[:i] + new + src[i + len(old):]


patch("""            Y = _shared_old384_mm(J, _e2_H(_e2_H(weight[:, None]) * _e2_H(_e2_H(J[_jn_rows, :]).T)))
            if old_columns:
                Y = Y + _e2_H(_shared_old384_mm(Qp, _e2_H(_shared_old384_mm(state['core'], _e2_H(_e2_H(Qp[_jn_rows, :]).T)))))
            del _jn_d""", """            _ntd_f0 = _ntd_flops()
            Y = _shared_old384_mm(J, _e2_H(_e2_H(weight[:, None]) * _e2_H(_e2_H(J[_jn_rows, :]).T)))
            if old_columns:
                Y = Y + _e2_H(_shared_old384_mm(Qp, _e2_H(_shared_old384_mm(state['core'], _e2_H(_e2_H(Qp[_jn_rows, :]).T)))))
            _ntd_log(layer, 'Yactual%d' % int(Y.shape[1]), _ntd_flops() - _ntd_f0)
            del _jn_d""")
patch("""        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
        core = None""", """        _ntd_f0 = _ntd_flops()
        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
        _ntd_log(layer, 'QR%d' % int(Y.shape[1]), _ntd_flops() - _ntd_f0)
        _ntd_f0 = _ntd_flops()
        core = None""")
patch("""        del K, unused_R, Y, Omega, _jn_rows""", """        _ntd_log(layer, 'CORE', _ntd_flops() - _ntd_f0)
        del K, unused_R, Y, Omega, _jn_rows""")
patch("""        Fj = _shared_old384_mm(_e2_H(Qn.T), J)""", """        _ntd_f0 = _ntd_flops()
        Fj = _shared_old384_mm(_e2_H(Qn.T), J)
        _ntd_log(layer, 'Fj', _ntd_flops() - _ntd_f0)""")
patch("""    _UVP_CUR[0] = layer
""", """    _UVP_CUR[0] = layer
    _ntd_log(layer, 'PUSH_ENTRY_CUM', _ntd_flops())
""")
src += '''

# BEGIN OPUS-U64IDEA DIAGNOSTIC PROBES (not deployable)
import sys as _ntd_sys
from flopscope._budget import get_active_budget as _ntd_budget


def _ntd_flops():
    return int(_ntd_budget().flops_used)


def _ntd_log(layer, label, f):
    print('[ntd] layer %d %s %d' % (layer, label, f), file=_ntd_sys.stderr, flush=True)
# END OPUS-U64IDEA DIAGNOSTIC PROBES
'''
o = D/'cand_s2trdiag_v1.py'
with o.open('x') as f:
    f.write(src)
compile(o.read_text(), str(o), 'exec'); print(o.name, hashlib.sha256(o.read_bytes()).hexdigest()[:8])
