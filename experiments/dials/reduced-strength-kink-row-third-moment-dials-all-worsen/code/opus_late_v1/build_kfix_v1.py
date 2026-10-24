"""[opus-late] (1) KFIX: fixed excess-kurtosis offset in the post-variance formula at L12/13/14 (beta +0.0014/+0.0022/+0.0034, derived on
7103248-51 only by var_share_v1.py). v_post <- v_post * exp(-beta_l * h(z)), h(z) = first-order Edgeworth kurtosis sensitivity of log Var[ReLU]
at the unit's standardized pre-activation z = aux['alpha'] (gate = Phi(z), pdf = phi(z) from gaussian_raw), applied only where z >= -2 (the
fitted range). One asserted insertion in the LIVE _astra_fhm_core right before the post-variance is written onto C's diagonal."""
import hashlib
from pathlib import Path
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
src = (O/'lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py').read_text(); assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
OLD = "                _ps_post_cov32 = C\n                C = C.astype(fnp.float64)\n                fnp.fill_diagonal(C, _ps_Cdiag)\n"
NEW = ("                if l in _LKF_BETA:\n"
       "                    _ps_Cdiag = _lkf_post_var(_ps_Cdiag, aux, l)   # [opus-late] KFIX: fixed excess-kurtosis offset in the post variance\n" + OLD)
d = src.rindex('\ndef _astra_fhm_core('); e = src.find('\ndef ', d + 1)
assert src.count(OLD) >= 1 and src[d:e].count(OLD) == 1, 'post-variance write not unique in the live _astra_fhm_core'
i = src.index(OLD, d); out = src[:i] + NEW + src[i + len(OLD):]
helper = '''

# BEGIN [opus-late] KFIX (fixed excess-kurtosis offset in the post variance, L12-14; derived on 7103248-51 only)
_LKF_BETA = {12: 0.0014, 13: 0.0022, 14: 0.0034}


def _lkf_post_var(vdiag, aux, l):
    z, P, p = aux['alpha'], aux['gate'], aux['pdf']
    zz = z * z
    m1 = z * P + p
    m2 = (zz + 1.0) * P + z * p
    var = fnp.maximum(m2 - m1 * m1, 1e-30)
    dm2 = (-1.0 / 12.0) * (z * p)
    dm1 = (1.0 / 24.0) * ((zz - 1.0) * p)
    h = fnp.where(z >= -2.0, (dm2 - 2.0 * (m1 * dm1)) / var, 0.0)
    f = fnp.exp(h * (-_LKF_BETA[l]))
    return vdiag * (f if str(f.dtype) == str(vdiag.dtype) else f.astype(vdiag.dtype))
# END [opus-late] KFIX
'''
out = out + helper
(O/'lead/runs/opus_late_v1/cand_s2tr_late_kfix_v1.py').write_text(out); print('kfix', hashlib.sha256(out.encode()).hexdigest()[:12])
