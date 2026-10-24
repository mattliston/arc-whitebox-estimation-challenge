"""[opus-diet2b] request diet 2b v2 = v1 minus the _gop2_gnn_step edit (client-server census crash: c['rms'] reaches the server as a
Python float, the 09-25 client-mode trap) and _fz3 refusing every 0-d operand that _fzk did not make.
v1: EINSUM-CHAIN FUSION of 2-multiply chains in the physics functions of the clean base LR2 (9ef4b5e5),
by line-number edits of LIVE lines (each line's text is asserted) plus helpers appended LAST.
09-25 einsum-chain laws: einsum(optimize=False) with <= 3 operands equals the left-deep multiply chain bit for bit (f32 and f64,
broadcast forms, 0-d constants, bool masks) and bills exactly the separate multiplies; distinct operand OBJECTS only; no float32 prefix
that later meets float64; no value from integer indexing (client mode returns a Python scalar); never out= (bills 2x); 0-d constants
are made once per process (1 request, 2 flops). _fz3 checks all of it at run time (client-side metadata only, no request) and falls
back to the plain chain otherwise. Commuted forms are used only where the original multiplies the same two values ((x*y)*z == z*(x*y)).
usage: build_diet2b_v1.py <base.py> <base sha prefix> <out.py>"""
import hashlib, sys
from pathlib import Path
base = Path(sys.argv[1]).read_text(); assert hashlib.sha256(base.encode()).hexdigest().startswith(sys.argv[2])
L = base.split('\n')
EDITS = [
    # _nonlinear_pair_batch (14 calls a network)
    (872, "    P4 = _e2_H(_e2_H(mu ** 4) + _e2_H(_e2_H(_e2_H(6 * mu) * mu) * vc)) + _e2_H(v3 * vc)",
          "    P4 = _e2_H(_e2_H(mu ** 4) + _e2_H(_fz3(_e2_H(6 * mu), mu, vc))) + _e2_H(v3 * vc)"),
    (874, "    G4 = _e2_H(_e2_H(fnp.stack((mu, q1, mq3, P4))) * g) + _e2_H(_e2_H(_e2_H(fnp.stack((st['one'], mu, q2, mq5))) * s) * pdf)",
          "    G4 = _e2_H(_e2_H(fnp.stack((mu, q1, mq3, P4))) * g) + _e2_H(_fz3(_e2_H(fnp.stack((st['one'], mu, q2, mq5))), s, pdf))"),
    (886, "    post_k3 = _e2_H(r3 - _e2_H(_e2_H(3 * m) * r2)) + _e2_H(2 * _e2_H(m ** 3))",
          "    post_k3 = _e2_H(r3 - _e2_H(_fz3(_fzk(3.0, m), m, r2))) + _e2_H(2 * _e2_H(m ** 3))"),
    (887, "    post_k4 = _e2_H(_e2_H(_e2_H(r4 - _e2_H(_e2_H(4 * m) * r3)) + _e2_H(_e2_H(_e2_H(6 * m) * m) * r2)) - _e2_H(3 * _e2_H(m ** 4))) - _e2_H(_e2_H(3 * var) * var)",
          "    post_k4 = _e2_H(_e2_H(_e2_H(r4 - _e2_H(_fz3(_fzk(4.0, m), m, r3))) + _e2_H(_fz3(_e2_H(6 * m), m, r2))) - _e2_H(3 * _e2_H(m ** 4))) - _e2_H(_e2_H(3 * var) * var)"),
    (892, "    aa = fnp.stack((mgs, g, pdf, A3, A4, _e2_H(_e2_H(-h5) * pdf), _e2_H(h6 * pdf)))",
          "    aa = fnp.stack((mgs, g, pdf, A3, A4, _e2_H(_fz3(_fzk(-1.0, h5), h5, pdf)), _e2_H(h6 * pdf)))"),
    # _g5 covariance factor (7 calls)
    (7796, "    return _e2_H(_e2_H(_e2_H(0.5 * _e2_H(raw + _e2_H(raw.T))) * _e2_H(fnp.outer(sf, sf))) * t['cov_factor'])",
           "    return _e2_H(_fz3(_e2_H(0.5 * _e2_H(raw + _e2_H(raw.T))), _e2_H(fnp.outer(sf, sf)), t['cov_factor']))"),
    # _f4c_step (14)
    (10827, "    _e2_H(fnp.put(beta, live_indices, _e2_H(t1 + _e2_H(_e2_H(2 * gl) * t2))))   # float32 values, widened by the scatter; the division is in float64 as before",
            "    _e2_H(fnp.put(beta, live_indices, _e2_H(t1 + _e2_H(_fz3(_fzk(2.0, gl), gl, t2)))))   # float32 values, widened by the scatter; the division is in float64 as before"),
    (10830, "    _e2_H(fnp.put(diagonal, live_indices, _e2_H(_e2_H(_e2_H(3 * gl) * gl) * dg)))",
            "    _e2_H(fnp.put(diagonal, live_indices, _e2_H(_fz3(_e2_H(3 * gl), gl, dg))))"),
    # _k4l_dk (15)
    (11010, "    wi = _e2_H(_e2_H(_e2_H(q * q) + 1e-12) * pushed) * iso",
            "    wi = _fz3(_e2_H(_e2_H(q * q) + 1e-12), pushed, iso)"),
    # _b6_m2 (14): (t3*t3)*(he4*ph) == (he4*ph)*(t3*t3)
    (11245, "    _hc_put(l, 'm2', hold(hold(hold(t3 * t3) * hold(he4 * ph)) * (1.0 / 72.0)), response_mask, nrows)",
            "    _hc_put(l, 'm2', hold(hold(_fz3(he4, ph, hold(t3 * t3))) * (1.0 / 72.0)), response_mask, nrows)"),
    # (Ut @ vector) * (3 * rho) == (3 * rho) * (Ut @ vector)
    (14924, "    return ((Ut @ vector) * (3 * rho)).astype(fnp.float64)",
            "    return _fz3(_fzk(3.0, rho), rho, Ut @ vector).astype(fnp.float64)"),
    # _bd_update (15)
    (15032, "    delta = _e2_H(_e2_H(_e2_H(new32 + old32) * difference) * weight) + _e2_H(fnp.einsum('i,ji->ij', quadratic_weight, difference, optimize=False))",
            "    delta = _e2_H(_fz3(_e2_H(new32 + old32), difference, weight)) + _e2_H(fnp.einsum('i,ji->ij', quadratic_weight, difference, optimize=False))"),
    # _astra_fhm_core (15)
    (17536, "            b = _e2_H(gq * bpre) + _e2_H(_e2_H(_e2_H(radial * d) * vv) / _S19_BSRC_DIV) if _FULL_LOWER_MOMENTS else _e2_H(_e2_H(gq * bpre) + _e2_H(_e2_H(_e2_H(radial * d) * vv) / _S19_BSRC_DIV)) * alive",
            "            b = _e2_H(gq * bpre) + _e2_H(_e2_H(_fz3(radial, d, vv)) / _S19_BSRC_DIV) if _FULL_LOWER_MOMENTS else _e2_H(_e2_H(gq * bpre) + _e2_H(_e2_H(_fz3(radial, d, vv)) / _S19_BSRC_DIV)) * alive"),
    # Bdiag * (2 * b) == (2 * b) * Bdiag
    (17558, "            _ps_trace_diag = _ps_Bdiag * b + _ps_Bdiag * (2 * b)",
            "            _ps_trace_diag = _ps_Bdiag * b + _fz3(_fzk(2.0, b), b, _ps_Bdiag)"),
]
for n, old, new in EDITS:
    assert L[n - 1] == old, (n, L[n - 1][:120])
    L[n - 1] = new
HELPERS = '''

# BEGIN [opus-diet2b] EINSUM-CHAIN FUSION v1 helpers (2026-10-16; append LAST). Byte-identical by the 09-25 einsum-chain laws; see
# lead/runs/opus_passrel_v1/diet2b_v1/build_diet2b_v1.py for the fused lines. _fz3(a, b, c) == (a * b) * c bit for bit.
_FZ_STATS = dict(fused=0, fallback=0, consts=0)
_FZ_K = {}
_FZ_IDS = set()   # id() of the 0-d constants _fzk made (they live in _FZ_K for the whole process)


def _fzk(value, like):
    """A cached 0-d constant of like's dtype (made once per process: 1 request, 2 flops); like's own value if it is not an array."""
    if not (hasattr(like, 'shape') and hasattr(like, 'dtype')):
        return value
    key = (value, str(like.dtype))
    got = _FZ_K.get(key)
    if got is None:
        got = _FZ_K[key] = fnp.asarray(value, dtype=like.dtype)
        _FZ_IDS.add(id(got))
        _FZ_STATS['consts'] += 1
    return got


def _fz3(a, b, c):
    """(a * b) * c as ONE einsum(optimize=False) when bit-identical: three distinct array OBJECTS, one float dtype (bool masks allowed),
    every operand's shape a right-aligned suffix of the result's (no size-1 broadcasting); otherwise the plain left-deep chain."""
    ops = (a, b, c)
    if a is b or b is c or a is c or not all(hasattr(x, 'shape') and hasattr(x, 'dtype') for x in ops):
        _FZ_STATS['fallback'] += 1
        return (a * b) * c
    shapes = [tuple(int(d) for d in x.shape) for x in ops]
    if any(not s_ and id(x) not in _FZ_IDS for s_, x in zip(shapes, ops)):   # a 0-d value not made by _fzk may be a bare float on the server
        _FZ_STATS['fallback'] += 1
        return (a * b) * c
    out = max(shapes, key=len)
    if (len({str(x.dtype) for x in ops} - {'bool'}) != 1 or 1 in out or len(out) > 6
            or any(s != out[len(out) - len(s):] for s in shapes)):
        _FZ_STATS['fallback'] += 1
        return (a * b) * c
    letters = 'abcdef'[:len(out)]
    _FZ_STATS['fused'] += 1
    return fnp.einsum(','.join(letters[len(out) - len(s):] for s in shapes) + '->' + letters, a, b, c, optimize=False)
# END [opus-diet2b] EINSUM-CHAIN FUSION v1 helpers
'''
s = '\n'.join(L) + HELPERS
compile(s, 'diet2b', 'exec')
with open(sys.argv[3], 'x') as f:
    f.write(s)
print(hashlib.sha256(s.encode()).hexdigest()[:8], len(EDITS), 'edits')
