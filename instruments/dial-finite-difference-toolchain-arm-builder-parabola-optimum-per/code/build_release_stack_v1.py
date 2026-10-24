"""TAG fable-lead 2026-10-19: build a RELEASE form (and its science copy) = base + zero-bill dial moves + optional conditional k4 dial.
  --dials <file>: lines of python assignments appended verbatim (e.g. `_BSK = 0.434`), read at predict time, bill unchanged.
  --k4cv <json> : {"b2": [c0, c1, c2], "b3": [c0, c1, c2]} -> the two-band conditional dial (cv_vv at L3/L8/L12), patched at the four
                  k4 band sites exactly as candidate_rel_k4cv2_v1.py.
Writes <out>.py (release form) and <out>_sci_v1.py (release form + `_FAST_RESCUE_CAP_S = 1e9`, for the concurrent-safe judge).
usage: build_release_stack_v1.py <base release candidate> <out stem> <note> [--dials file] [--k4cv json] [--k4shape json (k4shape_coef_v1.json; first_layer 8)] [--k3shape json (k3shape_coef_v1.json) [--k3centered]] [--k3l15 json (k3l15_coef_v1.json) [--k3l15-strength S] [--k3l15-scale X]]"""
import json
import sys
from pathlib import Path

args = sys.argv[1:]
base, stem, note = Path(args[0]), args[1], args[2]
dials = Path(args[args.index('--dials') + 1]) if '--dials' in args else None
k4cv = Path(args[args.index('--k4cv') + 1]) if '--k4cv' in args else None
k4shape = Path(args[args.index('--k4shape') + 1]) if '--k4shape' in args else None
k3shape = Path(args[args.index('--k3shape') + 1]) if '--k3shape' in args else None
k3centered = '--k3centered' in args
k3l15 = Path(args[args.index('--k3l15') + 1]) if '--k3l15' in args else None
k3l15_strength = float(args[args.index('--k3l15-strength') + 1]) if '--k3l15-strength' in args else 1.0
k3l15_scale = float(args[args.index('--k3l15-scale') + 1]) if '--k3l15-scale' in args else 1.0
out, out_sci = Path(stem + '.py'), Path(stem + '_sci_v1.py')
assert not out.exists() and not out_sci.exists(), (out, out_sci)
src = base.read_text().rstrip('\n') + '\n'
src += '\n# BEGIN FABLE-LEAD RELEASE STACK 2026-10-19 (%s)\n' % note
if dials:
    moves = [l.rstrip() for l in dials.read_text().split('\n') if l.strip() and not l.startswith('#')]
    src += '# zero-bill dial moves (read at predict time):\n' + '\n'.join(moves) + '\n'
if k4cv:
    c = json.load(open(k4cv)); b2, b3 = [round(float(x), 4) for x in c['b2']], [round(float(x), 4) for x in c['b3']]
    old = "                k4 = k4 * (_S19_K4BAND[0] if l <= 7 else (_S19_K4BAND[1] if l <= 11 else _S19_K4BAND[2]))\n"
    assert src.count(old) == 4, src.count(old)
    new = old + ("                if _K4CV_ON:   # fable-lead 10-19: conditional band-2/3 k4 scale from the net's own variance dispersion at L3/L8/L12\n"
                 "                    _k4cv_rec(l, vv)\n                    if 8 <= l <= 11:\n                        k4 = k4 * _k4cv_factor()\n"
                 "                    elif 12 <= l <= 14:\n                        k4 = k4 * _k4cv_factor3()\n")
    src = src.replace(old, new)
    src += '''# CONDITIONAL K4 BAND-2 + BAND-3 DIAL (zero bill): per-network band scales from the variance dispersion cv_l = std(diag Cpre_l) /
# mean(diag Cpre_l) at layers 3, 8, 12: d2 = C2[0] + C2[1] cv8 + C2[2] cv3 ; d3 = C3[0] + C3[1] cv8 + C3[2] cv12 (clipped to +-0.07);
# band scales become _S19_K4BAND[1] (1 + d2), _S19_K4BAND[2] (1 + d3). Coefficients: %s
_K4CV_ON = True
_K4CV = {}
_K4CV_COEF = (%r, %r, %r)
_K4CV3_COEF = (%r, %r, %r)
_K4CV_CLIP = 0.07


def _k4cv_rec(l, vv):
    if l == 1:
        _K4CV.clear()
    if l in (3, 8, 12):
        m = float(fnp.mean(vv))
        _K4CV[l] = (float(fnp.std(vv)) / m) if m > 0 else 0.0


def _k4cv_factor():
    if 3 in _K4CV and 8 in _K4CV:
        d = _K4CV_COEF[0] + _K4CV_COEF[1] * _K4CV[8] + _K4CV_COEF[2] * _K4CV[3]
        return 1.0 + max(-_K4CV_CLIP, min(_K4CV_CLIP, d))
    return 1.0


def _k4cv_factor3():
    if 8 in _K4CV and 12 in _K4CV:
        d = _K4CV3_COEF[0] + _K4CV3_COEF[1] * _K4CV[8] + _K4CV3_COEF[2] * _K4CV[12]
        return 1.0 + max(-_K4CV_CLIP, min(_K4CV_CLIP, d))
    return 1.0
''' % (str(k4cv), b2[0], b2[1], b2[2], b3[0], b3[1], b3[2])
if k4shape:
    t = json.load(open(k4shape)); table = {int(l): round(-float(v[0]), 3) for l, v in t['coef'].items() if int(l) >= int(t.get('first_layer', 8))}
    K4L = "                k4 = (_e2_append_inline((_e2_value_inline := k4)) or _e2_value_inline) + _hc_k4l\n"
    assert src.count(K4L) == 4, src.count(K4L)
    src = src.replace(K4L, K4L + "                k4 = k4 + _k4shape(l, vv, bpre)   # fable-lead 10-19: per-unit k4 shape correction fitted to Monte Carlo truth\n")
    src += """# PER-UNIT K4 SHAPE CORRECTION (zero bill): Monte Carlo per-unit cumulant truth on fresh nets shows the card's per-unit k4 at L8-14
# is skew-coupled: err_i = model - truth ~ b1 vv_i^2 (k3n_i^2 - mean), k3n = 3 bpre / sqrt(vv) (the card's carried skew proxy).
# Correction dk4_i = c_l vv_i^2 (k3n_i^2 - mean k3n^2) with c_l = -b1, centered (layer-mean k4 and the band dials untouched), after K4L.
_K4SHAPE_C = %r


def _k4shape(l, vv, bpre):
    c = _K4SHAPE_C.get(int(l))
    if not c:
        return 0.0
    k3n2 = (3.0 * bpre) * (3.0 * bpre) / vv
    return c * (vv * vv * (k3n2 - fnp.mean(k3n2)))
""" % (table,)
if k3shape:
    t = json.load(open(k3shape)); tab = {int(l): [round(float(v), 6) for v in c] for l, c in t['coef'].items() if int(l) >= 8}
    FILL = "            fnp.fill_diagonal(K21, _ps_kdiag)\n"
    assert src.count(FILL) == 4, src.count(FILL)
    src = src.replace(FILL, "            _ps_kdiag = _k3mod(l, _ps_kdiag, mu, vv)   # fable-lead 10-19: per-unit k3 shape correction fitted to Monte Carlo truth\n" + FILL)
    src += """# PER-UNIT K3 SHAPE CORRECTION (zero bill): Monte Carlo per-unit cumulant truth on fresh nets shows the card's per-unit pre-activation
# k3 (diag K21) at L8-14 has a predictable error: card - truth ~ a + v^1.5 (b z + c z^2 + d k3n^2), z = mu/sqrt(v), k3n = k3/v^1.5
# (coefficients stable across nets; 55-70 percent of the per-unit error variance leave-one-net-out). The diagonal is corrected by -fit.
_K3SHAPE = %r
_K3SHAPE_CENTERED = %r


def _k3mod(l, kd, mu, vv):
    c = _K3SHAPE.get(int(l))
    if not c:
        return kd
    s = fnp.sqrt(vv)
    z = mu / s
    v15 = vv * s
    k3n2 = (kd / v15) * (kd / v15)
    fit = c[0] + v15 * (c[1] * z + c[2] * (z * z) + c[3] * k3n2)
    if _K3SHAPE_CENTERED:
        fit = fit - fnp.mean(fit)
    return kd - fit
""" % (tab, k3centered)
if k3l15:
    c = tuple(round(float(x), 7) for x in json.load(open(k3l15))['coef'])
    T15 = "                k3 = k3 + (_e2_append_inline((_e2_value_inline := ((_e2_append_inline((_e2_value_inline := (3 * active_v))) or _e2_value_inline) * bpre))) or _e2_value_inline)\n"
    assert src.count(T15) == 4, src.count(T15)
    src = src.replace(T15, T15 + "                k3 = _k3mod15(k3, mu, v)   # fable-lead 10-19: terminal-layer per-unit k3 correction fitted to Monte Carlo truth\n")
    src += """# TERMINAL-LAYER PER-UNIT K3 CORRECTION (zero bill): Monte Carlo per-unit cumulant truth on fresh nets shows the card's L15 per-unit
# pre-activation k3 has rms error ~47 percent of the rms truth, dominated by a missing skew-mean coupling (corr(err, z) -0.8):
# card - truth ~ a + v^1.5 (b z + c z^2 + d k3n^2), 72-78 percent of the error variance leave-one-net-out. k3 -> (k3 - S fit) x SCALE.
_K3L15_COEF = %r
_K3L15_STRENGTH = %r
_K3L15_SCALE = %r


def _k3mod15(k3, mu, v):
    if not _K3L15_STRENGTH and _K3L15_SCALE == 1.0:
        return k3
    s = fnp.sqrt(v)
    z = mu / s
    v15 = v * s
    k3n2 = (k3 / v15) * (k3 / v15)
    c = _K3L15_COEF
    fit = c[0] + v15 * (c[1] * z + c[2] * (z * z) + c[3] * k3n2)
    return (k3 - _K3L15_STRENGTH * fit) * _K3L15_SCALE
""" % (c, k3l15_strength, k3l15_scale)
src += '# END FABLE-LEAD RELEASE STACK\n'
compile(src, str(out), 'exec')
out.write_text(src)
sci = src + '_FAST_RESCUE_CAP_S = 1e9   # science variant: concurrent judges must not trip the rescue\n'
compile(sci, str(out_sci), 'exec')
out_sci.write_text(sci)
print('written', out.name, 'and', out_sci.name)
