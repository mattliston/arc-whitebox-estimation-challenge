"""[opus-tank] option B on S2tr + JORD (48ca245e, apply_jord_v1.py): the hunter's task-aware Nystrom row block (lead/runs/opus_ladder2h_v1/
build_tjoin_v1.py, the same text) inserted just before the JORD _jn_rows line inside the LAST def _astra_bcc_push_impl: rows feeding this
layer's kink units (|a| < c, a = _E2_LAYER['a']) get energy x (1 + beta t_j / mean t), t_j = sum_{i kink} W[j, i]^2. Lead spec: joins
L10 / L13 only, beta 0.5 / 1 / 2 at c = 1.0 (the oracle smoke's kink set |a| < 1). Value-only arms."""
import hashlib
from pathlib import Path
D = Path(__file__).resolve().parent
src = (D/'cand_s2tr_jord_v1.py').read_text(); assert hashlib.sha256(src.encode()).hexdigest().startswith('48ca245e')
i0 = src.rfind('\ndef _astra_bcc_push_impl(')
OLD = "            _jn_rows = _e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank]   # JORD: energy order (UVP leading r_u coordinates)\n"
NEW = """            _tj_a = _E2_LAYER['a']   # [opus-tank] option B (hunter's tjoin block): task-aware row energy at the listed joins
            _tj_k = fnp.nonzero(fnp.abs(_tj_a) < _L2H_TJ_C)[0]
            if int(_tj_k.shape[0]) > 0 and layer in _L2H_TJ_LAYERS:
                _tj_t = fnp.sum(w32[previous][:, _tj_k] ** 2, axis=1)
                _jn_d = _jn_d * (1.0 + _L2H_TJ_BETA * _tj_t / fnp.mean(_tj_t))
""" + OLD
j = src.find(OLD, i0)
assert j > i0 and src.find(OLD, j + 1) == -1
body = src[:j] + NEW + src[j + len(OLD):]
for tag, beta in (('b05', 0.5), ('b10', 1.0), ('b20', 2.0)):
    out = body + '\n\n# BEGIN [opus-tank] ARM %s\n_L2H_TJ_C = 1.0\n_L2H_TJ_BETA = %r\n_L2H_TJ_LAYERS = (10, 13)\n# END [opus-tank] ARM %s\n' % (tag, beta, tag)
    p = D/f'cand_s2tr_jord_tank_{tag}_v1.py'
    with p.open('x') as f: f.write(out)
    compile(out, str(p), 'exec'); print(tag, hashlib.sha256(out.encode()).hexdigest()[:8])
