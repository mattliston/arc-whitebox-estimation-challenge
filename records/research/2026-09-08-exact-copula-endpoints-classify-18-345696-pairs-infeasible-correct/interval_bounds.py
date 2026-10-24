"""Small outward endpoint reference; no experiment IO or execution at import."""
import importlib.util
import math
from decimal import Decimal
from pathlib import Path
import sys

import numpy as np


def load_decimal(path):
    spec = importlib.util.spec_from_file_location('_copula_endpoint_decimal', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def scalar_cdf(reference, value):
    """128-term certified series at one literal double with abs(value)<=4."""
    value = float(value)
    if not math.isfinite(value) or abs(value) > 4:
        raise ValueError('Outside fixed exact-endpoint scalar domain')
    point, Interval = reference.point, reference.Interval
    x = point(abs(value))
    term, total = x, point(0)
    x2 = x.square()
    for k in range(128):
        total = total + term if k % 2 == 0 else total - term
        term = term * x2 * (2*k+1) / (2*(k+1)*(2*k+3))
    total = total + Interval(Decimal(0), term.hi)
    cdf = point(1)/2 + total/(2*reference.pi_interval()).sqrt()
    return cdf if value >= 0 else 1-cdf


class Bounds:
    """Every arithmetic op encloses exact real operations on binary64 bounds."""
    def __init__(self, lo, hi=None):
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = self.lo if hi is None else np.asarray(hi, dtype=np.float64)
        if not (np.isfinite(self.lo).all() and np.isfinite(self.hi).all()
                and np.all(self.lo <= self.hi)):
            raise ArithmeticError('Invalid outward bounds')

    @staticmethod
    def cast(other):
        return other if isinstance(other, Bounds) else Bounds(other)

    def __getitem__(self, item):
        return Bounds(self.lo[item], self.hi[item])

    def __neg__(self):
        return Bounds(-self.hi, -self.lo)

    def __add__(self, other):
        other = self.cast(other)
        return Bounds(np.nextafter(self.lo+other.lo, -np.inf),
                      np.nextafter(self.hi+other.hi, np.inf))

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-self.cast(other))

    def __rsub__(self, other):
        return self.cast(other) + (-self)

    def __mul__(self, other):
        other = self.cast(other)
        p = (self.lo*other.lo, self.lo*other.hi,
             self.hi*other.lo, self.hi*other.hi)
        return Bounds(np.nextafter(np.minimum.reduce(p), -np.inf),
                      np.nextafter(np.maximum.reduce(p), np.inf))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.cast(other)
        if np.any((other.lo <= 0) & (other.hi >= 0)):
            raise ZeroDivisionError('Outward denominator contains zero')
        p = (self.lo/other.lo, self.lo/other.hi,
             self.hi/other.lo, self.hi/other.hi)
        return Bounds(np.nextafter(np.minimum.reduce(p), -np.inf),
                      np.nextafter(np.maximum.reduce(p), np.inf))


def select(mask, yes, no):
    yes, no = Bounds.cast(yes), Bounds.cast(no)
    return Bounds(np.where(mask, yes.lo, no.lo), np.where(mask, yes.hi, no.hi))


def from_decimal(values):
    return Bounds(np.nextafter(np.array([float(x.lo) for x in values]), -np.inf),
                  np.nextafter(np.array([float(x.hi) for x in values]), np.inf))


def endpoint_arrays(alpha_i, alpha_j, phi_i, phi_j, cdf_i, cdf_j,
                    scale, sign):
    a, b = Bounds(alpha_i), Bounds(alpha_j)
    smaller_i = np.asarray(alpha_i) <= np.asarray(alpha_j)
    cdf_min = select(smaller_i, cdf_i, cdf_j)
    phi_min = select(smaller_i, phi_i, phi_j)
    largest = Bounds(np.maximum(alpha_i, alpha_j))
    mean_product = (phi_i+a*cdf_i)*(phi_j+b*cdf_j)
    plus = (1+a*b)*cdf_min+largest*phi_min-mean_product
    overlap = np.asarray(alpha_i) > -np.asarray(alpha_j)
    product_minus = (a*b-1)*(cdf_j+cdf_i-1)+b*phi_i+a*phi_j
    minus = select(overlap, product_minus, 0)-mean_product
    return (scale*sign*select(sign > 0, minus, plus),
            scale*sign*select(sign > 0, plus, minus))


def scalar_endpoint_by_integrals(reference, a, b, tau_i, tau_j,
                                 sigma_i, sigma_j, sign):
    """Independent expanded truncated-normal I0,I1,I2 integration."""
    P = reference.point
    aa, bb = P(float(a)), P(float(b))
    ca, cb = scalar_cdf(reference, a), scalar_cdf(reference, b)
    pa, pb = reference.normal_pdf(aa), reference.normal_pdf(bb)
    mean_product = (pa+aa*ca)*(pb+bb*cb)
    small = min(a, b)
    cc, pc = scalar_cdf(reference, small), reference.normal_pdf(P(float(small)))
    i0, i1, i2 = cc, pc, cc-P(float(small))*pc
    plus = i2+(aa+bb)*i1+aa*bb*i0-mean_product
    if a > -b:
        i0 = ca+cb-1
        i1 = pa-pb
        i2 = i0-bb*pb-aa*pa
        raw_minus = -i2+(bb-aa)*i1+aa*bb*i0
    else:
        raw_minus = P(0)
    minus = raw_minus-mean_product
    scale = (P(tau_i)/P(sigma_i))*(P(tau_j)/P(sigma_j))
    return ((scale*minus, scale*plus) if sign > 0
            else (-scale*plus, -scale*minus))


def certify(reference):
    """Fixed fixtures; runs before any causal archive is decoded."""
    count = 0
    cache = {}
    for a in (-3., -1., 0., .5, 2., 3.):
        cache[a] = (scalar_cdf(reference, a), reference.normal_pdf(reference.point(a)))
    for a in cache:
        for b in cache:
            for sign in (-1., 1.):
                ti, tj, si, sj = .3, 1.7, .6, 1.1
                ca, pa = cache[a]
                cb, pb = cache[b]
                scale = (Bounds(ti)/si)*(Bounds(tj)/sj)
                actual = endpoint_arrays(np.array([a]), np.array([b]),
                    from_decimal([pa]), from_decimal([pb]),
                    from_decimal([ca]), from_decimal([cb]), scale, np.array([sign]))
                expected = scalar_endpoint_by_integrals(reference, a, b, ti, tj, si, sj, sign)
                for got, want in zip(actual, expected):
                    if not (Decimal.from_float(float(got.lo[0])) <= want.lo
                            and Decimal.from_float(float(got.hi[0])) >= want.hi):
                        raise AssertionError(('Endpoint enclosure fixture', a, b, sign))
                    count += 1
    # Known exact alpha=0 sentinels, independent of the product expansion.
    c, p = cache[0.]
    pi = reference.pi_interval()
    for sign in (-1., 1.):
        got = endpoint_arrays(np.array([0.]), np.array([0.]),
            from_decimal([p]), from_decimal([p]), from_decimal([c]), from_decimal([c]),
            Bounds(1.), np.array([sign]))
        positive, negative = reference.point(1)/2-reference.point(1)/(2*pi), -reference.point(1)/(2*pi)
        expected = (negative, positive) if sign > 0 else (-positive, -negative)
        for actual, target in zip(got, expected):
            if not (Decimal.from_float(float(actual.lo[0])) <= target.lo
                    and Decimal.from_float(float(actual.hi[0])) >= target.hi):
                raise AssertionError('Reflection sentinel')
            count += 1
    return dict(passed=True, scalar_endpoint_enclosures=count,
                decimal_precision=160, cdf_terms=128, cdf_abs_domain=4,
                target_arrays_accessed=False)
