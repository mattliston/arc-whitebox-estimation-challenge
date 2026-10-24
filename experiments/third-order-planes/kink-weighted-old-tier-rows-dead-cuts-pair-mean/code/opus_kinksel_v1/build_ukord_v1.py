"""[opus-kinksel] (c) UVP/LCR leading-r_u coordinate order by KINK-WEIGHTED energy, on S2tr + JORD (c0458f0e, the 24-seed JORD screen file).
JORD orders the Nystrom join rows (the QR frame's leading coordinates, whose first r_u = 256 carry the u-leg pair means) by Gram energy
_jn_d. Here the SET stays JORD's (top-rank by _jn_d: bill-neutral, same rows) and only its ORDER becomes _jn_d x (1 + beta t / mean t),
t_j = sum_{i kink at this layer, |a_i| < c} W[j, i]^2 (the hunter's option-B weight; a = _E2_LAYER['a'], this layer's own mu / sigma),
so the leading r_u coordinates prefer rows feeding this layer's kink units. Joins 10 and 13 (release's B arm layers), c = 1.0.
Arms: ukord_b1 (beta 1), ukord_b2 (beta 2). Text edit inside the LAST _astra_bcc_push_impl + a value block appended LAST."""
import hashlib
from pathlib import Path
D = Path(__file__).resolve().parent; R = D.parent
src = (R/'opus_grow_v1/cand_s2tr_grow_jord_v1.py').read_text(); assert hashlib.sha256(src.encode()).hexdigest().startswith('c0458f0e')
i0 = src.rfind('\ndef _astra_bcc_push_impl(')
OLD = "            _jn_rows = _e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank]   # [opus-grow] JORD: energy order (UVP leading r_u coordinates)\n"
NEW = OLD + """            if layer in _UK_LAYERS:   # [opus-kinksel] (c): same row set, kink-weighted order
                _uk_k = fnp.nonzero(fnp.abs(_E2_LAYER['a']) < _UK_C)[0]
                if int(_uk_k.shape[0]) > 0:
                    _uk_t = fnp.sum(w32[previous][:, _uk_k] ** 2, axis=1)
                    _uk_key = _e2_H(_e2_H(_jn_d * (1.0 + _UK_BETA * _uk_t / fnp.mean(_uk_t)))[_jn_rows])
                    _jn_rows = _e2_H(_jn_rows[fnp.argsort(_e2_H(-_e2_H(_uk_key.astype(fnp.float64))), kind='stable')])
"""
j = src.find(OLD, i0); assert j > i0 and src.find(OLD, j + 1) == -1
body = src[:j] + NEW + src[j + len(OLD):]
for tag, beta in (('ukord_b1', 1.0), ('ukord_b2', 2.0)):
    out = body + '\n\n# BEGIN [opus-kinksel] ARM %s\n_UK_C = 1.0\n_UK_BETA = %r\n_UK_LAYERS = (10, 13)\n# END [opus-kinksel] ARM %s\n' % (tag, beta, tag)
    o = D/f'cand_s2trj_{tag}_v1.py'
    with o.open('x') as f: f.write(out)
    compile(out, str(o), 'exec'); print(o.name, hashlib.sha256(out.encode()).hexdigest()[:8])
