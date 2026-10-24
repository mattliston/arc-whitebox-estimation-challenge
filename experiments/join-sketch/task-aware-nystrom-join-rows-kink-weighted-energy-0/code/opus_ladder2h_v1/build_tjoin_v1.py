# [opus-tjoin] TASK-AWARE Nystrom rows at the joins (option B): the live join (_astra_bcc_push_impl) ranks rows by Gram energy _jn_d;
# here _jn_d_j x (1 + beta * t_j / mean(t)), t_j = sum_{i kink at this layer} W[j, i]^2 over the layer's kink units |a_i| < c
# (a = _E2_LAYER['a'], the layer's own mu / sigma, known before the push), so rows feeding kink units are preferred. Same rank, same order
# rule (index-sorted, as S2tr). Zero bill (~6 small vector ops per join). S2tr 46e9315d; text edit inside the LAST _astra_bcc_push_impl.
import hashlib, sys
O = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1'
H = O + '/lead/runs/opus_ladder2h_v1'
src = open(O + '/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py').read()
assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
i0 = src.rfind('\ndef _astra_bcc_push_impl(')
OLD = "            _jn_rows = fnp.sort(_e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank])\n"
NEW = """            _tj_a = _E2_LAYER['a']
            _tj_k = fnp.nonzero(fnp.abs(_tj_a) < _L2H_TJ_C)[0]
            if int(_tj_k.shape[0]) > 0 and layer in _L2H_TJ_LAYERS:
                _tj_t = fnp.sum(w32[previous][:, _tj_k] ** 2, axis=1)
                _jn_d = _jn_d * (1.0 + _L2H_TJ_BETA * _tj_t / fnp.mean(_tj_t))
""" + OLD
j = src.find(OLD, i0)
assert j > i0 and src.find(OLD, j + 1) == -1
body = src[:j] + NEW + src[j + len(OLD):]
for tag, c, beta, layers in (('tj05b2', 0.5, 2.0, (7, 10, 13)), ('tj10b1', 1.0, 1.0, (7, 10, 13))):
    out = body + '\n\n# BEGIN [opus-tjoin] ARM %s\n_L2H_TJ_C = %r\n_L2H_TJ_BETA = %r\n_L2H_TJ_LAYERS = %r\n# END [opus-tjoin] ARM %s\n' % (tag, c, beta, layers, tag)
    p = H + '/cand_s2tr_l2h_%s_v1.py' % tag
    open(p, 'w').write(out)
    print(tag, hashlib.sha256(out.encode()).hexdigest()[:8])
