"""[opus-ladder2] release share: three EXACT arms on S2tr 46e9315d, each a fragment appended LAST:
  S1X   the star's s1 term restored exactly (E2-S1Z dropped it): every live sandwich call (_dc_cov_star, layers 1-14) adds
        (2/3) s1 off the diagonal, s1 = ((W' A) o W') W in float32 through _mm (the diagonal the parent fills is kept), to the
        float64 star result and the retained float32 star plane: sub = (2 s1 + s2) / 3 as before E2-S1Z (accuracy oracle, ~+2 MM/layer)
  DCVX  the dead-cut sandwich off: _DCV_FLOOR = 0.0 -> every roster is the whole width -> the parent full sandwich (exact)
  PSDMX the inner-size psdm kernel off: _PSDM_ON = False -> the padded 1,024 symmetric product (rounding-level twin)
Writes cand_s2tr_l2r_<arm>_v1.py and prints their sha256."""
import hashlib
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
D = Path(__file__).resolve().parent
src = (O/'lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py').read_text()
assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
S1X = '''

# BEGIN [opus-ladder2] S1X v1 (accuracy oracle, NOT for release): the star's s1 term restored exactly on every live sandwich call
_S1X_PARENT_DC = _dc_cov_star
_S1X_STATS = dict(calls=0)


def _dc_cov_star(w, C, A, gate, floor, retain32=False):
    out = _S1X_PARENT_DC(w, C, A, gate, floor, retain32)
    wt = w.T
    A32 = A if str(A.dtype) == 'float32' else A.astype(fnp.float32)
    pushed = _mm(fnp.ascontiguousarray(wt), A32)
    s1 = _mm(pushed * wt, w) * (2.0 / 3.0)
    fnp.fill_diagonal(s1, 0.0)
    result = out[1]
    fnp.add(result, s1, out=result)
    if retain32:
        fnp.add(out[3], s1, out=out[3])
    _S1X_STATS['calls'] += 1
    return out
# END [opus-ladder2] S1X v1
'''
ARMS = {'S1X': S1X, 'DCVX': "\n\n# [opus-ladder2] DCVX v1: the dead-cut covariance sandwich off (exact full sandwich)\n_DCV_FLOOR = 0.0\n",
        'PSDMX': "\n\n# [opus-ladder2] PSDMX v1: the inner-size psdm sandwich kernel off (padded 1,024 symmetric product)\n_PSDM_ON = False\n"}
for name, frag in ARMS.items():
    t = src + frag; compile(t, name, 'exec')
    p = D/f'cand_s2tr_l2r_{name}_v1.py'; p.write_text(t)
    print(name, hashlib.sha256(t.encode()).hexdigest()[:12])
