"""[opus-k4unit] CAPTURE build (diagnostic, not deployable): S2tr 46e9315d + stashes of (a) the L14 pair map's post22 / post_k4 / cov /
post21 / mean / pre-activation v and mu (inside _nonlinear_pair_batch when _PHYS_LAYER[0] == 14) and (b) the L15 readout's mu, v, k3,
k4 (= radial v^2 + _k4l_final, evaluated once, just before the same nonlinear_scalar call), the K4L part, radial, the pre-head mean,
the post covariance C entering L15 and W15. Arithmetic unchanged (the k4 argument is computed one statement earlier)."""
import hashlib, re
from pathlib import Path
D = Path(__file__).resolve().parent; R = D.parent
src = (R/'opus_s2tr_v1/candidate_S2tr_v1.py').read_text()
assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
old = """    fnp.fill_diagonal(post22, post_k4)
    fnp.fill_diagonal(cov, var)
    fnp.fill_diagonal(post21, post_k3)
"""
assert src.count(old) == 1
src = src.replace(old, old + """    if _PHYS_LAYER[0] == 14:   # opus-k4unit capture
        _K4U_CAP['L14'] = dict(post22=_k4u_np.array(post22, dtype=_k4u_np.float64), post_k4=_k4u_np.array(post_k4, dtype=_k4u_np.float64),
                               cov=_k4u_np.array(cov, dtype=_k4u_np.float64), post21=_k4u_np.array(post21, dtype=_k4u_np.float64),
                               m=_k4u_np.array(m, dtype=_k4u_np.float64), pre_v=_k4u_np.array(v, dtype=_k4u_np.float64), pre_mu=_k4u_np.array(mu, dtype=_k4u_np.float64))
""")
pat = re.compile(r'^( +)m, aux = nonlinear_scalar\(mu, v, k3, _e2_H\(_e2_H\(radial \* v\) \* v\) \+ _e2_H\(_k4l_final\(mu, v\)\)\)(.*)$', re.M)
def rep(mo):
    ind = mo.group(1)
    return (f"{ind}_k4u_k4l = _k4l_final(mu, v)   # opus-k4unit capture: same value, evaluated one statement earlier\n"
            f"{ind}_k4u_k4 = _e2_H(_e2_H(radial * v) * v) + _e2_H(_k4u_k4l)\n"
            f"{ind}m, aux = nonlinear_scalar(mu, v, k3, _k4u_k4){mo.group(2)}\n"
            f"{ind}_K4U_CAP['L15'] = dict(mu=_k4u_np.array(mu, dtype=_k4u_np.float64), v=_k4u_np.array(v, dtype=_k4u_np.float64), k3=_k4u_np.array(k3, dtype=_k4u_np.float64),\n"
            f"{ind}                       k4=_k4u_np.array(_k4u_k4, dtype=_k4u_np.float64), k4l=_k4u_np.array(_k4u_k4l, dtype=_k4u_np.float64) * _k4u_np.ones(len(mu)), radial=_k4u_np.array(float(radial)),\n"
            f"{ind}                       m=_k4u_np.array(m, dtype=_k4u_np.float64), C=_k4u_np.array(C, dtype=_k4u_np.float64), W=_k4u_np.array(W[l], dtype=_k4u_np.float64))")
src, n = pat.subn(rep, src); assert n >= 1, n
head = "import numpy as _k4u_np\n_K4U_CAP = {}\n"
i = src.index('\nimport ') + 1
src = src[:i] + head + src[i:]
o = D/'cand_s2tr_k4ucap_v1.py'
with o.open('x') as f:
    f.write(src)
compile(src, str(o), 'exec'); print(o.name, hashlib.sha256(src.encode()).hexdigest()[:8], 'final sites', n)
