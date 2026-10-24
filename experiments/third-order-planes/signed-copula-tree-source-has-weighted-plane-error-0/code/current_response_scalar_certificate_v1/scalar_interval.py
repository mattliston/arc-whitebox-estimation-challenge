"""Inert independent outward-interval reference for current scalar feature21.

No source/NPZ/file IO, NumPy, Jet implementation, model, sampling, or arithmetic
at import time.  ``evaluate`` treats each supplied Python float as its EXACT
binary value.  It uses fixed Decimal precision160 and private Context objects;
decimal.getcontext() is never changed or used for arithmetic.

Basic operations use directed rounding.  Decimal Context.exp/sqrt are assumed
to satisfy their documented correctly-rounded ROUND_HALF_EVEN contracts; each
endpoint is widened by one adjacent precision160 Decimal.  Underflow, subnormal
results, overflow and invalid operations FAIL, rather than silently returning
an uncertified enclosure.  This is an interval reference to the declared real
polynomial/CDF scalar block, not a derivative of a discontinuous float program.

Pi is enclosed by Machin's identity with alternating-series tails.  For
|alpha|<=12, Phi uses the integrated exponential alternating series with512
terms and its certified next-term remainder.  The initial terms need not
decrease; all terms from index512 onward do, since their successive ratio is
at most72/(k+1)<1.  Outside this range, the positive Gaussian Mills inequalities
phi(x)*x/(x*x+1) < 1-Phi(x) < phi(x)/x provide an enclosure.  No unqualified
mpmath/scipy special function or adaptive precision is used.

API: evaluate(mu, rawv, sigma, k3, k4, order=5,
              feature21_denominator=D_float)
returns tuples of Interval objects for feature21 value/alpha derivatives and
mu derivatives.  Supply D_float=(sigma*sigma)*(sigma*sigma) from the original
float64 feature normalizer.  The exact-real sigma**4/D_float factor is retained.
``point`` and ``outward_l2`` support the separate diagnostic runner.  A numerical
diagnosis does not change the frozen finite-difference FAIL or admit gate truth.
"""
from dataclasses import dataclass
from decimal import (Context, Decimal, DivisionByZero, FloatOperation,
                     InvalidOperation, Overflow, ROUND_CEILING, ROUND_FLOOR,
                     ROUND_HALF_EVEN, Subnormal, Underflow)
from functools import lru_cache
from math import factorial


PRECISION = 160
CDF_SERIES_TERMS = 512
MACHIN_TERMS_5 = 256
MACHIN_TERMS_239 = 64


@lru_cache(maxsize=3)
def _context(rounding):
    """Private contexts: no dependence on or mutation of global Decimal state."""
    context = Context(prec=PRECISION, rounding=rounding,
                      Emin=-999999, Emax=999999)
    for condition in (DivisionByZero, InvalidOperation, Overflow,
                      Underflow, Subnormal, FloatOperation):
        context.traps[condition] = True
    return context


def _decimal(value):
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, float):
        result = Decimal.from_float(value)
    elif isinstance(value, int) and not isinstance(value, bool):
        result = Decimal(value)
    else:
        raise TypeError('Exact scalar must be a Python float, int, or Decimal')
    if not result.is_finite():
        raise ValueError('Nonfinite exact scalar')
    return result


@dataclass(frozen=True)
class Interval:
    lo: Decimal
    hi: Decimal

    def __post_init__(self):
        lo, hi = _decimal(self.lo), _decimal(self.hi)
        if lo > hi:
            raise ValueError('Reversed interval')
        object.__setattr__(self, 'lo', lo)
        object.__setattr__(self, 'hi', hi)

    def __add__(self, other):
        other = _interval(other)
        return Interval(_context(ROUND_FLOOR).add(self.lo, other.lo),
                        _context(ROUND_CEILING).add(self.hi, other.hi))

    __radd__ = __add__

    def __neg__(self):
        # Decimal unary minus/abs may consult the global context: copy_* do not.
        return Interval(self.hi.copy_negate(), self.lo.copy_negate())

    def __sub__(self, other):
        other = _interval(other)
        return Interval(_context(ROUND_FLOOR).subtract(self.lo, other.hi),
                        _context(ROUND_CEILING).subtract(self.hi, other.lo))

    def __rsub__(self, other):
        return _interval(other) - self

    def __mul__(self, other):
        other = _interval(other)
        pairs = ((self.lo, other.lo), (self.lo, other.hi),
                 (self.hi, other.lo), (self.hi, other.hi))
        lower = [_context(ROUND_FLOOR).multiply(a, b) for a, b in pairs]
        upper = [_context(ROUND_CEILING).multiply(a, b) for a, b in pairs]
        return Interval(min(lower), max(upper))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = _interval(other)
        if other.lo <= 0 <= other.hi:
            raise ZeroDivisionError('Interval denominator contains zero')
        pairs = ((self.lo, other.lo), (self.lo, other.hi),
                 (self.hi, other.lo), (self.hi, other.hi))
        lower = [_context(ROUND_FLOOR).divide(a, b) for a, b in pairs]
        upper = [_context(ROUND_CEILING).divide(a, b) for a, b in pairs]
        return Interval(min(lower), max(upper))

    def __rtruediv__(self, other):
        return _interval(other) / self

    def square(self):
        low = [
            _context(ROUND_FLOOR).multiply(self.lo, self.lo),
            _context(ROUND_FLOOR).multiply(self.hi, self.hi)]
        high = [
            _context(ROUND_CEILING).multiply(self.lo, self.lo),
            _context(ROUND_CEILING).multiply(self.hi, self.hi)]
        lower = Decimal(0) if self.lo <= 0 <= self.hi else min(low)
        return Interval(lower, max(high))

    def __pow__(self, power):
        if not isinstance(power, int) or isinstance(power, bool):
            raise TypeError('Only exact integer interval powers are supported')
        if power < 0:
            return point(1) / (self ** (-power))
        if power == 0:
            return point(1)
        if power == 1:
            return self
        if power % 2 == 0:
            return (self ** (power // 2)).square()
        return self * (self ** (power - 1))

    def __abs__(self):
        if self.lo <= 0 <= self.hi:
            return Interval(Decimal(0), max(self.lo.copy_abs(), self.hi.copy_abs()))
        endpoints = (self.lo.copy_abs(), self.hi.copy_abs())
        return Interval(min(endpoints), max(endpoints))

    def exp(self):
        context = _context(ROUND_HALF_EVEN)
        lower = context.exp(self.lo).next_minus(context=context)
        upper = context.exp(self.hi).next_plus(context=context)
        return Interval(max(Decimal(0), lower), upper)

    def sqrt(self):
        if self.lo < 0:
            raise ValueError('Square-root interval has negative lower endpoint')
        context = _context(ROUND_HALF_EVEN)
        lower = (Decimal(0) if self.lo == 0 else
                 context.sqrt(self.lo).next_minus(context=context))
        upper = (Decimal(0) if self.hi == 0 else
                 context.sqrt(self.hi).next_plus(context=context))
        return Interval(max(Decimal(0), lower), upper)

    def as_strings(self):
        return dict(lo=str(self.lo), hi=str(self.hi))


def point(value):
    """Exact enclosure of the given float/int/Decimal; no rounded decimal literal."""
    exact = _decimal(value)
    return Interval(exact, exact)


def _interval(value):
    return value if isinstance(value, Interval) else point(value)


def outward_l2(values):
    """Outward Euclidean norm, including intervals that straddle zero."""
    total = point(0)
    for value in values:
        total = total + _interval(value).square()
    return total.sqrt()


def _arctan_inverse(denominator, terms):
    """Alternating arctan(1/q), enclosing its exact rational next-term tail."""
    total = point(0)
    for k in range(terms):
        term = point(1) / ((2 * k + 1) * denominator ** (2 * k + 1))
        total = total + term if k % 2 == 0 else total - term
    next_term = point(1) / ((2 * terms + 1) * denominator ** (2 * terms + 1))
    remainder = Interval(Decimal(0), next_term.hi)
    return total + remainder if terms % 2 == 0 else total - remainder


@lru_cache(maxsize=1)
def pi_interval():
    """Machin: pi=16 atan(1/5)-4 atan(1/239), with outward rational tails."""
    return (16 * _arctan_inverse(5, MACHIN_TERMS_5)
            - 4 * _arctan_inverse(239, MACHIN_TERMS_239))


@lru_cache(maxsize=1)
def _inverse_root_two_pi():
    return point(1) / (2 * pi_interval()).sqrt()


def normal_pdf(alpha):
    alpha = _interval(alpha)
    return (-alpha.square() / 2).exp() * _inverse_root_two_pi()


def _cdf_positive(x):
    """CDF enclosure at one exact nonnegative Decimal endpoint."""
    if x < 0:
        raise ValueError('Expected nonnegative CDF argument')
    if x == 0:
        return point(1) / 2
    a = point(x)
    if x <= 12:
        a2 = a.square()
        term, integral = a, point(0)
        for k in range(CDF_SERIES_TERMS):
            integral = integral + term if k % 2 == 0 else integral - term
            term = term * a2 * (2 * k + 1) / (2 * (k + 1) * (2 * k + 3))
        # 512 is even. The remaining alternating series starts positively,
        # and every term thereafter decreases, uniformly for x<=12.
        integral = integral + Interval(Decimal(0), term.hi)
        result = point(1) / 2 + _inverse_root_two_pi() * integral
    else:
        density = normal_pdf(a)
        tail_lower = density * a / (a.square() + 1)
        tail_upper = density / a
        result = Interval((1 - tail_upper).lo, (1 - tail_lower).hi)
    # These intersections use exact known probability bounds, not a heuristic
    # numerical clipping rule. A disjoint enclosure raises in Interval().
    return Interval(max((point(1) / 2).lo, result.lo),
                    min(Decimal(1), result.hi))


def _cdf_endpoint(x):
    return _cdf_positive(x) if x >= 0 else 1 - _cdf_positive(x.copy_abs())


def normal_cdf(alpha):
    """Use exact CDF monotonicity for an outward alpha interval."""
    alpha = _interval(alpha)
    lower, upper = _cdf_endpoint(alpha.lo), _cdf_endpoint(alpha.hi)
    return Interval(lower.lo, upper.hi)


def _polynomial_add(*polynomials):
    size = max(len(p) for p in polynomials)
    output = []
    for k in range(size):
        coefficient = point(0)
        for polynomial in polynomials:
            if k < len(polynomial):
                coefficient = coefficient + polynomial[k]
        output.append(coefficient)
    return output


def _polynomial_derivative(polynomial):
    return ([k * polynomial[k] for k in range(1, len(polynomial))]
            or [point(0)])


def _polynomial_value(polynomial, alpha):
    value = point(0)
    for coefficient in reversed(polynomial):
        value = value * alpha + coefficient
    return value


def _moment_derivatives(p, q, alpha, cdf, pdf, order):
    """Independent P_next=P', Q_next=P+Q'-alpha*Q recurrence."""
    values = []
    for k in range(order + 1):
        values.append(_polynomial_value(p, alpha) * cdf
                      + _polynomial_value(q, alpha) * pdf)
        if k < order:
            p, q = (_polynomial_derivative(p),
                    _polynomial_add(p, _polynomial_derivative(q),
                                    [point(0)] + [-a for a in q]))
    return tuple(values)


def _convolve(a, b, order):
    output = []
    for r in range(order + 1):
        value = point(0)
        for k in range(r + 1):
            value = value + a[k] * b[r - k]
        output.append(value)
    return tuple(output)


def evaluate(mu, rawv, sigma, k3, k4, order=5, feature21_denominator=None):
    """Outward current scalar law at one fixed point, never a numerical retry.

    Returns ``feature21_alpha_derivatives[r]`` and
    ``feature21_mu_derivatives[r]`` for r=0..order.  The zeroth entry is the
    mathematical scalar VALUE, before any native additive anchor.  Additive
    anchors do not change its derivatives.  The diagnostic must separately
    reproduce original floating anchors and bound their evaluation error.

    For Python-float sigma only, an omitted denominator is the literal binary
    float sequence v=sigma*sigma; D=v*v.  Supplying the authenticated original
    D explicitly is preferred and mandatory for non-float sigma.  No incoming
    variance/sigma-floor ratio is silently set to one.  Extreme inputs that
    exceed Decimal's supported normal range fail without a precision fallback.
    """
    if not isinstance(order, int) or isinstance(order, bool) or not 0 <= order <= 5:
        raise ValueError('Fixed derivative order must lie in 0..5')
    mu0, v0, s, incoming3, incoming4 = map(point, (mu, rawv, sigma, k3, k4))
    if s.lo <= 0 or v0.lo == 0:
        raise ValueError('Require positive supplied sigma and nonzero raw variance')
    if feature21_denominator is None:
        if not isinstance(sigma, float):
            raise TypeError('Supply actual fixed denominator for non-float sigma')
        rounded_v = sigma * sigma
        feature21_denominator = rounded_v * rounded_v
    denominator = point(feature21_denominator)
    if denominator.lo <= 0:
        raise ValueError('Feature21 denominator must be finite and positive')
    floor_value = max(v0.lo, Decimal.from_float(1e-10))
    variance_floor = point(floor_value)
    s2, s3, s4 = s.square(), s ** 3, s ** 4
    alpha = mu0 / s
    u, r = variance_floor / s2, s2 / v0
    t3, t4 = incoming3 / s3, incoming4 / s4
    zero, one = point(0), point(1)
    polynomials = (
        ([zero, one],
         [1 - r * t4 / 24, -r * t3 / 6, r * t4 / 24]),
        ([u, zero, one],
         [t3 / 3, 1 - r * t4 / 12]),
        ([t3, 3 * u, zero, one],
         [2 * u + t4 / 4, zero, one]),
        ([3 * u.square() + t4, 4 * t3, 6 * u, zero, one],
         [4 * t3, 5 * u, zero, one]))
    cdf, pdf = normal_cdf(alpha), normal_pdf(alpha)
    moments = tuple(_moment_derivatives(p, q, alpha, cdf, pdf, order)
                    for p, q in polynomials)
    b1, b2, b3, b4 = [tuple(m[k] / factorial(k) for k in range(order + 1))
                      for m in moments]
    b11 = _convolve(b1, b1, order)
    b12, b13 = _convolve(b1, b2, order), _convolve(b1, b3, order)
    b22 = _convolve(b2, b2, order)
    b111 = _convolve(b11, b1, order)
    b112 = _convolve(b11, b2, order)
    b1111 = _convolve(b11, b11, order)
    variance = tuple(factorial(k) * (b2[k] - b11[k]) for k in range(order + 1))
    third = tuple(factorial(k) * (b3[k] - 3 * b12[k] + 2 * b111[k])
                  for k in range(order + 1))
    fourth = tuple(factorial(k) * (b4[k] - 4 * b13[k] - 3 * b22[k]
                                    + 12 * b112[k] - 6 * b1111[k])
                   for k in range(order + 1))
    scale = s4 / denominator
    feature_alpha = tuple(scale * value for value in fourth)
    feature_mu = tuple(value / (s ** k) for k, value in enumerate(feature_alpha))
    return dict(
        feature21_alpha_derivatives=feature_alpha,
        feature21_mu_derivatives=feature_mu,
        normalized_raw_moment_alpha_derivatives=moments,
        normalized_post_variance_alpha_derivatives=variance,
        normalized_post_k3_alpha_derivatives=third,
        normalized_post_k4_alpha_derivatives=fourth,
        alpha=alpha,
        constants=dict(mu=mu0, rawv=v0, sigma=s, variance_floor=variance_floor,
                       u=u, r=r, t3=t3, t4=t4,
                       feature21_denominator=denominator,
                       feature21_exact_sigma4_over_denominator=scale),
        primitives=dict(precision=PRECISION, decimal_context='private directed contexts',
                        machin_terms_5=MACHIN_TERMS_5,
                        machin_terms_239=MACHIN_TERMS_239,
                        cdf_series_terms=CDF_SERIES_TERMS,
                        cdf_lower_endpoint=('alternating_integral' if alpha.lo.copy_abs() <= 12
                                            else 'mills_tail'),
                        cdf_upper_endpoint=('alternating_integral' if alpha.hi.copy_abs() <= 12
                                            else 'mills_tail'),
                        exp_sqrt='correctly-rounded Decimal, widened one adjacent endpoint',
                        no_adaptive_precision=True,
                        qualification='Encloses the declared real scalar block; original '
                                      'float evaluation/anchor error is a separate comparison'))
