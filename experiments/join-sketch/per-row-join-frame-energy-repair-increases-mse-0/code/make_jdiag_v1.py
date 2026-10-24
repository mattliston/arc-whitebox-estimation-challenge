"""TAG fable-lead 2026-10-20: build the JDIAG science arm = candidate_rel_v4a_sci_v1 + a per-row ENERGY REPAIR of the join's
projection. At a join layer the young cohorts' columns J (and the old tier Qp core Qp^T) are replaced by their projection on the
rank-384 frame Qn (J ~ Qn Fj); the projection drops the Schur part of the carrier Gram G = J w J^T + old, so the per-unit energy
diag(G) (already computed as _jn_d for the landmark ranking) is under/over-represented row by row. The repair rescales the rows of
the frame by s_i = sqrt(diag(G)_i / diag(Qn M Qn^T)_i), M = Fj w Fj^T + overlap core overlap^T, before the frame is pushed, so the
old tier's pushed columns carry the exact per-unit energy. Knobs (module globals, read at call time): _JDIAG_ON, _JDIAG_LAYERS,
_JDIAG_MIN/_MAX (clip of the ratio), _JDIAG_POWER (strength), _JDIAG_DEBUG (stderr trace).
usage: make_jdiag_v1.py <base sci candidate> <out>"""
import sys
from pathlib import Path
base, out = Path(sys.argv[1]), Path(sys.argv[2])
assert not out.exists(), out
text = base.read_text()
cut = text.rindex('def _astra_bcc_push_impl(')
head, tail = text[:cut], text[cut:]

def once(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:60])
    return s.replace(old, new)

tail = once(tail, "        Omega = w32[previous, :rank]\n", "        Omega = w32[previous, :rank]\n        _jd_dfull = None   # [fable-lead JDIAG]\n")
tail = once(tail, "            del _jn_d\n        else:\n", "            _jd_dfull = _jn_d; del _jn_d   # [fable-lead JDIAG] keep diag(G)\n        else:\n")
block = '''        _jd_s = None   # [fable-lead 2026-10-20 JDIAG] per-row energy repair of the join projection (see make_jdiag_v1.py)
        if _JDIAG_ON and layer in _JDIAG_LAYERS and _jd_dfull is not None:
            _jd_M = fnp.matmul(Fj * weight[None, :], Fj.T)
            if old_columns:
                _jd_ov = fnp.matmul(Qn.T, Qp)
                _jd_M = _jd_M + fnp.matmul(fnp.matmul(_jd_ov, state['core']), _jd_ov.T)
                del _jd_ov
            _jd_proj = fnp.sum(fnp.matmul(Qn, _jd_M) * Qn, axis=1)
            _jd_ratio = _jd_dfull.astype(fnp.float64) / fnp.maximum(_jd_proj.astype(fnp.float64), 1e-30)
            _jd_s = fnp.sqrt(fnp.clip(_jd_ratio, _JDIAG_MIN, _JDIAG_MAX)) ** _JDIAG_POWER
            if _JDIAG_DEBUG:
                import sys as _jd_sys
                print('JDIAG L%d width %d rows %d old %d trace full %.4e proj %.4e ratio min %.3f mean %.3f max %.3f' % (layer, int(Fj.shape[1]), int(Qn.shape[0]), int(old_columns), float(fnp.sum(_jd_dfull)), float(fnp.sum(_jd_proj)), float(fnp.min(_jd_ratio)), float(fnp.mean(_jd_ratio)), float(fnp.max(_jd_ratio))), file=_jd_sys.stderr, flush=True)
            _jd_s = _jd_s.astype(fnp.float32)
            del _jd_M, _jd_proj, _jd_ratio
'''
anchor = "            core = core * (fnp.sum(fnp.matmul(Fj * Fj, weight)) / fnp.trace(core))\n"
tail = once(tail, anchor, anchor + block)
tail = once(tail, "        active_basis = Qn\n        del J, weight, coefficient, Fj, joined_core, factors, core\n",
            "        if _jd_s is not None:   # [fable-lead JDIAG] the repaired frame is what gets pushed and stored\n            Qn = Qn * _jd_s[:, None]\n            del _jd_s\n        active_basis = Qn\n        del J, weight, coefficient, Fj, joined_core, factors, core\n")
knobs = '''

# [fable-lead 2026-10-20] JDIAG science arm knobs (make_jdiag_v1.py): per-row energy repair of the join projection.
_JDIAG_ON = True
_JDIAG_LAYERS = (10,)
_JDIAG_MIN = 1.0
_JDIAG_MAX = 2.0
_JDIAG_POWER = 1.0
_JDIAG_DEBUG = False
'''
result = head + tail + knobs
compile(result, str(out), 'exec')
out.write_text(result)
print('written', out, 'lines', result.count('\n'), 'base lines', text.count('\n'))
