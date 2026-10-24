"""Fixed coefficient-one terminal k3-square completion; S32 prefix unchanged."""

def nonlinear_scalar(mu, v, k3, k4):
    _e2_H = _E2Hold()   # E2: temporaries of this call, released together (one hidden free trip)
    s,a,pdf,g,mg,r2g,r3g,r4g = gaussian_raw(mu,v)
    d = pdf/s; e = -a*pdf/v; f = (a*a-1)*pdf/(v*s)
    m = _e2_H(mg + _e2_H(_e2_H(k3 * e) / 6)) + _e2_H(_e2_H(k4 * f) / 24)
    r2 = _e2_H(r2g + _e2_H(_e2_H(k3 * d) / 3)) + _e2_H(_e2_H(k4 * e) / 12)
    r3 = _e2_H(r3g + _e2_H(k3 * g)) + _e2_H(_e2_H(k4 * d) / 4)
    r4 = _e2_H(r4g + _e2_H(_e2_H(4 * k3) * mg)) + _e2_H(k4 * g)
    # BEGIN fixed terminal k3-square raw-moment completion
    _tk3_q = k3 * k3 * pdf
    _tk3_a2 = a * a
    _tk3_h2 = _tk3_a2 - 1
    _tk3_h3 = a * (_tk3_a2 - 3)
    _tk3_h4 = _tk3_a2 * _tk3_a2 - 6 * _tk3_a2 + 3
    m = m + _tk3_q * _tk3_h4 / (72 * s ** 5)
    r2 = r2 - _tk3_q * _tk3_h3 / (36 * s ** 4)
    r3 = r3 + _tk3_q * _tk3_h2 / (12 * s ** 3)
    r4 = r4 - _tk3_q * a / (3 * s ** 2)
    # END fixed terminal k3-square raw-moment completion
    var = r2 - _e2_H(m * m)
    post_k3 = _e2_H(r3 - _e2_H(_e2_H(3 * m) * r2)) + _e2_H(2 * _e2_H(m ** 3))
    post_k4 = _e2_H(_e2_H(_e2_H(r4 - _e2_H(_e2_H(4 * m) * r3)) + _e2_H(_e2_H(_e2_H(6 * m) * m) * r2)) - _e2_H(3 * _e2_H(m ** 4))) - _e2_H(_e2_H(3 * var) * var)
    aux = dict(sigma=s,alpha=a,pdf=pdf,gate=g,gaussian_mean=mg,pre_v=v,pre_k3=k3,
               pre_k4=k4,post_v=var,post_k3=post_k3,post_k4=post_k4)
    return m,aux
