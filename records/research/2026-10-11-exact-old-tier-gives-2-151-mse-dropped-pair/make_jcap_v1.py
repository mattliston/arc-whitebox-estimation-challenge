"""TAG opus_engineer a62 (2026-10-11 23:2xZ, lead: 'the native vs torch join's captured lambda^2-weighted energy at L7/L10/L13 on one network'):
MEASUREMENT build (never a release file): S48b (8378970c) with one call inserted in the ACTIVE old-tier push right after the join's QR
and Nystrom core (before their del), which copies the join's operands to local numpy and logs, per join, against the joined Gram
G = J diag(lam^2) J' (+ Qp core_old Qp' when an old frame is there; rows = previous):
  proj = tr(Qn' G Qn) / tr G    the energy of G inside the native basis (Nystrom rows = the top Gram-diagonal rows)
  nys  = tr(core_new) / tr G    what the native Nystrom core represents (L7 / L10; the lazy L13 join has no core)
  rf   = the same as proj for the torch card's range finder (Y = G Omega, Omega = w32[previous, :rank], QR)
  opt  = the top-rank eigenvalues of G / tr G (the best any rank-r basis can keep)
Predictions are the parent's (the copies are local). The log is printed at the end of each prediction and kept in _JCAP_LOG.
usage: make_jcap_v1.py <out name>   -> a62_modules/s49/<out name>/candidate.py"""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE/'s48/s48b/candidate.py'
name = sys.argv[1]
out = HERE/'s49'/name
out.mkdir(parents=True)
text = SRC.read_text()
assert hashlib.sha256(text.encode()).hexdigest().startswith('8378970c')
anchor = "# Approximate range-finder core, original orthogonal factor projection.\ndef _shared_old384_push(layer, w32, p, previous, newborn, lam, state, rows=None):\n"
assert text.count(anchor) == 1
at = text.index(anchor)
end = text.index('\ndef ', at + len(anchor))
body = text[at:end]
old = "        del K, unused_R, Y, Omega, _jn_rows\n"
assert body.count(old) == 1
new = ("        _jcap_measure(layer, J, weight, Qp if old_columns else None, state.get('core') if old_columns else None, Qn, core, Omega, rank)   # opus-engineer a62 JCAP (measurement only)\n"
       + old)
text = text[:at] + body.replace(old, new) + text[end:]
text = text.rstrip('\n') + '''


# BEGIN OPUS-ENGINEER a62 JCAP (measurement only; never a release file): see O/engineer/a62_modules/make_jcap_v1.py.
import numpy as _jc_np
_JCAP_LOG = []


def _jcap_measure(layer, J, weight, Qp, core_old, Qn, core_new, Omega, rank):
    try:
        f = lambda x: _jc_np.asarray(x, dtype=_jc_np.float64)
        Jl, wl, Qnl = f(J), f(weight), f(Qn)
        JW = Jl * _jc_np.sqrt(wl)[None, :]
        G = JW @ JW.T
        if Qp is not None:
            Qpl = f(Qp)
            G = G + Qpl @ f(core_old) @ Qpl.T
        tot = float(_jc_np.trace(G))
        proj = float(_jc_np.sum((Qnl.T @ G) * Qnl.T)) / tot
        Qr = _jc_np.linalg.qr(G @ f(Omega), mode='reduced')[0]
        rf = float(_jc_np.sum((Qr.T @ G) * Qr.T)) / tot
        ev = _jc_np.linalg.eigvalsh(G)[::-1]
        opt = float(ev[:rank].sum()) / tot
        nys = float(_jc_np.trace(f(core_new))) / tot if core_new is not None else float('nan')
        row = dict(layer=int(layer), rows=int(G.shape[0]), joined=int(Jl.shape[1]), rank=int(rank), old=Qp is not None,
                   proj=proj, nys=nys, rf=rf, opt=opt)
    except Exception as e:
        row = dict(layer=int(layer), error=repr(e))
    _JCAP_LOG.append(row)
    print('JCAP', row, flush=True)
# END OPUS-ENGINEER a62 JCAP
'''
(out/'candidate.py').write_text(text)
print(name, hashlib.sha256(text.encode()).hexdigest()[:8])
