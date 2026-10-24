# [opus-ladder2] DEEPER ECUT (EDGE0 = edge off: xMSE +0.76% SE .60 for bill -0.54% -> the kept block carries accuracy, so cut inside it):
# ecutc = the tanh stand-in (10 ops / element) -> hard tanh clip(h, -1, 1) (1 op / element), rosters unchanged;
# ecutp = tighter rosters inside the edge only: rows pdf >= phi(1.5) (|mu/sigma| <= 1.5, was 2.33), columns gate >= 0.05 (was 0.01).
import hashlib, sys
O = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1'
H = O + '/lead/runs/opus_ladder2h_v1'
src = open(O + '/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py').read()
assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
lines = src.split('\n')
i0 = next(i for i, l in enumerate(lines) if l.startswith('def _ecut_raw('))
i1 = next(i for i in range(i0 + 1, len(lines)) if lines[i].startswith('# END OPUS-ENGINEER EDGE CUT'))
body = '\n'.join(lines[i0:i1]).rstrip() + '\n'
stand = body[body.index('    if _ECUT_STANDIN:\n'):body.index("    block = fnp.einsum(")]
assert stand.count('fnp.tanh(h, out=h)') == 1
ECUTC = body.replace(stand, "    hold.append(fnp.clip(h, -1.0, 1.0, out=h))   # [opus-ladder2] ECUTC: hard tanh\n")
ECUTP = '''_L2H_ECUT_PARENT = _ecut_raw
_L2H_PDF_FLOOR = float(0.3989422804014327 * __import__('math').exp(-0.5 * 1.5 * 1.5))
_L2H_GATE_FLOOR = 0.05


def _ecut_raw(left, right, t, k, pf, gf, hold):
    rosters = (_E2_LAYER['ecut_rows'], _E2_LAYER['ecut_cols'])
    I, J = rosters
    I2 = I[pf[I] >= _L2H_PDF_FLOOR]
    J2 = J[gf[J] >= _L2H_GATE_FLOOR]
    _E2_LAYER.update(ecut_rows=I2, ecut_cols=J2)
    try:
        return _L2H_ECUT_PARENT(left, right, t, k, pf, gf, hold)
    finally:
        _E2_LAYER.update(ecut_rows=rosters[0], ecut_cols=rosters[1])
'''
arms = dict(ecutc=ECUTC, ecutp=ECUTP)
for tag in (sys.argv[1:] or arms):
    out = src + '\n\n# BEGIN [opus-ladder2] ARM %s (append LAST to S2tr 46e9315d; deeper / cheaper, value-only)\n' % tag + arms[tag] + '# END [opus-ladder2] ARM %s\n' % tag
    p = H + '/cand_s2tr_l2h_%s_v1.py' % tag
    open(p, 'w').write(out)
    print(tag, hashlib.sha256(out.encode()).hexdigest()[:8], p)
