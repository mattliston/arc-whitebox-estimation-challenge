"""Public-FNP Gaussian tail and centered moments; fixed numerical repair only.

Rational coefficients: https://www.netlib.org/fdlibm/s_erf.c .
Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
Developed at SunSoft, a Sun Microsystems, Inc. business.
Permission to use, copy, modify, and distribute this software is freely
granted, provided that this notice is preserved.

Adaptation: directly return positive erfc, instead of subtracting saturated
erf from one. Split exponent with an FP32-rounded high part (not fdlibm's
bit truncation). All operand arithmetic, gathers, copies and scatters are
public metered FNP operations. No host/private special-function call.
"""
import flopscope.numpy as fnp

PP = (1.28379167095512558561e-1, -3.25042107247001499370e-1,
      -2.84817495755985104766e-2, -5.77027029648944159157e-3, -2.37630166566501626084e-5)
QQ = (1., 3.97917223959155352819e-1, 6.50222499887672944485e-2,
      5.08130628187576562776e-3, 1.32494738004321644526e-4, -3.96022827877536812320e-6)
PA = (-2.36211856075265944077e-3, 4.14856118683748331666e-1,
      -3.72207876035701323847e-1, 3.18346619901161753674e-1,
      -1.10894694282396677476e-1, 3.54783043256182359371e-2, -2.16637559486879084300e-3)
QA = (1., 1.06420880400844228286e-1, 5.40397917702171048937e-1,
      7.18286544141962662868e-2, 1.26171219808761642112e-1,
      1.36370839120290507362e-2, 1.19844998467991074170e-2)
RA = (-9.86494403484714822705e-3, -6.93858572707181764372e-1,
      -1.05586262253232909814e1, -6.23753324503260060396e1,
      -1.62396669462573470355e2, -1.84605092906711035994e2,
      -8.12874355063065934246e1, -9.81432934416914548592)
SA = (1., 1.96512716674392571292e1, 1.37657754143519042600e2,
      4.34565877475229228821e2, 6.45387271733267880336e2,
      4.29008140027567833386e2, 1.08635005541779435134e2,
      6.57024977031928170135, -6.04244152148580987438e-2)
RB = (-9.86494292470009928597e-3, -7.99283237680523006574e-1,
      -1.77579549177547519889e1, -1.60636384855821916062e2,
      -6.37566443368389627722e2, -1.02509513161107724954e3, -4.83519191608651397019e2)
SB = (1., 3.03380607434824582924e1, 3.25792512996573918826e2,
      1.53672958608443695994e3, 3.19985821950859553908e3,
      2.55305040643316442583e3, 4.74528541206955367215e2, -2.24409524465858183362e1)


def _poly(x, coefficients):
    value = coefficients[-1]
    for c in reversed(coefficients[:-1]):
        value = value * x + c
    return value


def lower_tail(a):
    """Phi(-a), finite nonnegative vector a. Regions use direct positive erfc."""
    if not bool(fnp.all(fnp.isfinite(a) & (a >= 0))):
        raise ValueError('Finite nonnegative lower-tail argument required')
    x = a / 1.4142135623730951
    out = fnp.zeros_like(x)
    regions = ((x < .84375), ((x >= .84375) & (x < 1.25)),
               ((x >= 1.25) & (x < 1 / .35)), ((x >= 1 / .35) & (x < 28)))
    for region, mask in enumerate(regions):
        indices = fnp.flatnonzero(mask)
        if not len(indices):
            continue
        t = x[indices]
        if region == 0:
            ratio = _poly(t * t, PP) / _poly(t * t, QQ)
            erfc = .5 - ((t - .5) + t * ratio)
        elif region == 1:
            u = t - 1
            erfc = (1 - .845062911510467529297) - _poly(u, PA) / _poly(u, QA)
        else:
            u = 1 / (t * t)
            p, q = (RA, SA) if region == 2 else (RB, SB)
            correction = _poly(u, p) / _poly(u, q)
            high = t.astype(fnp.float32).astype(fnp.float64)
            erfc = (fnp.exp(-high * high - .5625)
                    * fnp.exp((high - t) * (high + t) + correction) / t)
        fnp.put(out, indices, erfc / 2)
    # x>=28 has a true erfc below the FP64 subnormal range; zero is underflow,
    # not a negative-moment clamp. No finite value in the fixed grid is here.
    return out


def tail_moments(a, q, phi):
    """Positive raw moments E[ReLU(N(-a,1))**k], k1..3.

For a>=5 only, use fixed64 backward ratios I_k/I_(k-1)=k/(a+R_(k+1)).
This prevents O(a**6) relative amplification from the direct r3 subtraction.
The branch choice/count is fixed numerics, not a model gate or fitted parameter.
"""
    r1, r2, r3 = fnp.zeros_like(a), fnp.zeros_like(a), fnp.zeros_like(a)
    indices = fnp.flatnonzero(a < 5)
    if len(indices):
        t, qt = a[indices], q[indices]
        one = phi[indices] - t * qt
        two = qt - t * one
        three = 2 * one - t * two
        for out, value in ((r1, one), (r2, two), (r3, three)):
            fnp.put(out, indices, value)
    indices = fnp.flatnonzero(a >= 5)
    if len(indices):
        t = a[indices]
        ratio = fnp.zeros_like(t)
        ratios = {}
        for k in range(64, 0, -1):
            ratio = k / (t + ratio)
            if k <= 3:
                ratios[k] = ratio
        one = q[indices] * ratios[1]
        two = one * ratios[2]
        three = two * ratios[3]
        for out, value in ((r1, one), (r2, two), (r3, three)):
            fnp.put(out, indices, value)
    return r1, r2, r3


def gaussian_raw(mu, variance):
    """The same Gaussian moments as v1, evaluated without tail cancellation."""
    if not bool(fnp.all(fnp.isfinite(mu) & fnp.isfinite(variance))):
        raise ValueError('Nonfinite Gaussian input')
    v = fnp.maximum(variance, 1e-10)  # SAME pre-variance convention as v1.
    s = fnp.sqrt(v)
    alpha = mu / s
    a = fnp.abs(alpha)
    phi = fnp.exp(-a * a / 2) / 2.5066282746310002
    q = lower_tail(a)
    r1, r2, r3 = tail_moments(a, q, phi)
    positive = alpha >= 0
    gate = fnp.where(positive, 1 - q, q)
    mean = fnp.where(positive, a + r1, r1)
    var_negative = r2 - r1 * r1
    k_negative = r3 - 3 * r1 * r2 + 2 * r1 * r1 * r1
    lost_second = 2 * a * r1 + r2
    var_positive = 1 - lost_second - r1 * r1
    k_positive = ((3 * a * a * r1 + 3 * a * r2 + r3)
                  - 3 * r1 * (1 - lost_second) + 2 * r1 * r1 * r1)
    post_v = fnp.where(positive, var_positive, var_negative) * v
    k3 = fnp.where(positive, k_positive, k_negative) * v * s
    return dict(v=v, s=s, a=alpha, pdf=phi, g=gate, m=mean * s,
                post_v=post_v, k3=k3, d=phi / s, e=-alpha * phi / v)
