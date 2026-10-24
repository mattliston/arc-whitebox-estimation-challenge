"""TAG fable-lead 2026-10-20: CMODE v3 = v2 (per-layer rank-one common-mode covariance correction along mu, str/int keys) + a
_CMODE_DEBUG print of per-layer common-mode OBSERVABLES on stderr (card quantities only: mu^T Cpre mu/(|mu|^2 mean v) = the
covariance common-mode share, the card's radial of the previous layer, mean v, |mu|^2, kink fraction |a|<1, mean gate), for the
per-network sign study of the optimum coefficient. {} and _CMODE_DEBUG False = parent bit for bit. usage: make_cmode_v3.py <base sci> <out>"""
import sys
from pathlib import Path
base, out = Path(sys.argv[1]), Path(sys.argv[2])
assert not out.exists(), out
text = base.read_text()
anchor = "            bpre = wT @ b\n            vv = fnp.diag(Cpre)\n"
i = text.rindex(anchor)
hook = '''            if _CMODE_DEBUG and 6 <= int(l) <= 14:   # [fable-lead CMODE v3] per-layer common-mode observables (card quantities only)
                _cmd_mu = mu.astype(fnp.float64); _cmd_n2 = float(fnp.sum(_cmd_mu * _cmd_mu)); _cmd_v = fnp.diag(Cpre).astype(fnp.float64); _cmd_vm = float(fnp.mean(_cmd_v))
                _cmd_q = float(_cmd_mu @ (Cpre.astype(fnp.float64) @ _cmd_mu)) / (_cmd_n2 * _cmd_vm)
                _cmd_a = _cmd_mu / fnp.sqrt(fnp.maximum(_cmd_v, 1e-12)); _cmd_kink = float(fnp.mean((fnp.abs(_cmd_a) < 1.0).astype(fnp.float64)))
                _cmd_r = float(radial) if not hasattr(radial, 'shape') or tuple(radial.shape) == () else float(fnp.mean(radial))
                import sys as _cmd_sys
                print('CMODE_OBS L%d mumode %.5f radial_prev %.6f meanv %.5f mu2 %.2f kink1 %.4f gate %.4f' % (int(l), _cmd_q, _cmd_r, _cmd_vm, _cmd_n2, _cmd_kink, float(fnp.mean(flops.stats.norm.cdf(_cmd_a)))), file=_cmd_sys.stderr, flush=True)
                del _cmd_mu, _cmd_v, _cmd_a
            if _CMODE and (int(l) in _CMODE or str(int(l)) in _CMODE):   # [fable-lead 2026-10-20 CMODE v3] common-mode covariance correction
                _cm_old = Cpre
                Cpre, _plane_cov32 = _cmode_apply(l, mu, Cpre, _plane_cov32)
                if Bpre is _cm_old:
                    Bpre = Cpre
                del _cm_old
'''
text = text[:i] + hook + text[i:]
knobs = '''

# [fable-lead 2026-10-20] CMODE v3 science arm knobs (make_cmode_v3.py): per-layer rank-one common-mode covariance correction along mu.
_CMODE = {}
_CMODE_DEBUG = False


def _cmode_apply(l, mu, Cpre, cov32):
    c = _CMODE.get(int(l)) or _CMODE.get(str(int(l)))
    if not c:
        return Cpre, cov32
    mu64 = mu.astype(fnp.float64)
    scale = (float(c) * fnp.mean(fnp.diag(Cpre).astype(fnp.float64))) / fnp.sum(mu64 * mu64)
    corr = fnp.outer(mu64, mu64) * scale
    Cpre = Cpre + corr.astype(Cpre.dtype)
    if cov32 is not None:
        cov32 = cov32 + corr.astype(cov32.dtype)
    return Cpre, cov32
'''
result = text + knobs
compile(result, str(out), 'exec')
out.write_text(result)
print('written', out, 'lines', result.count('\n'))
