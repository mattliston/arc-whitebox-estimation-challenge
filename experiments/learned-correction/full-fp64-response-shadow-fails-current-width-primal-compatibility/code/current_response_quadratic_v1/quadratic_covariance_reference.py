"""Inert offline covariance-only quadratic-response reference; no CLI or I/O.

All arrays are supplied by a future, separately authorized caller.  Importing
this module evaluates no fixture/model/payload arrays.  This is a NumPy FP64
continuous-extension reference, NOT a participant implementation or bill.

For fixed incoming moments and fixed coefficients a,b, let
  q(y) = a*(y-mu) + b*((y-mu)**2 - sigma**2), psi = ReLU + q.
The normalized response table A[r] = sigma**(r-1)*E[d^r ReLU/dy^r] changes
only at dA[1]=a, dA[2]=2*b*sigma.  This file expands the SAME current pair11
functional by bilinearity, including every surviving mixed/marginal-kappa
term.  It never subtracts two large pair answers to form its delta.

The diagonal is restored from the fixed-coefficient marginal functional:
  dvar = dM2[0] + k3*dM2[3]/6 + k4*dM2[4]/24.
Its mean increment is zero.  The prospective action rejects below-floor
raw variances: centering q at sigma**2 does not repair the parent's separate
raw-v denominators.  prepare_reference/full_pair11_reference retain those
literal denominators and expose the floor flag for source qualification.
No PSD, positive density, nonnegative activation, or full joint-law claim
is made; PSD is not tested or imposed on the incoming covariance.

The output is the PRE-HEAD, unmasked covariance increment.  It is also the
increment after applying the SAME already-captured P correction/centering,
because the host mean is unchanged.  Do not recompute P on modified features.
The caller owns the frozen current mask, reference/target provenance and
every consumer.  This file changes no K21/K22, radial, star, CP or recurrent
state and does not specify or admit a full estimator.

Independent tiny certificate PLAN ONLY (nothing is executed here):
1. Authenticate this source and the helper pins.  On a fixed prospective
   synthetic fixture, compare full_pair11_reference(a=b=0) off diagonal to
   the independently AST-extracted current pair11 FP64 polynomial.
2. Compare covariance_delta off diagonal to full_pair11_reference(a,b)
   minus full_pair11_reference(0,0), using the separate dense loop below.
   Algebraically isolate Gaussian/K21/K22/K31/local-k3/local-k4 operator
   terms (not independently realizable physical cumulants), null a=b=0,
   and first-jet-only/second-jet-only/mixed coefficients.
3. Independently integrate (2*f*q+q*q)*He_r(z)*phi(z)/sigma**r for
   r=0,3,4, Y=mu+sigma*z, splitting at z=-mu/sigma.  The polynomial's
   reference mu,sigma,a,b remain fixed.  This checks the diagonal without
   differentiating a ReLU kink numerically.  A location finite difference
   must likewise shift ONLY the Gaussian integration mean, not q's center.
4. Check the Gaussian closed form, finite outputs, exact algebraic null,
   unchanged inputs, and explicit rejection of below-floor action states.
   Native FP32 parity, width1024 values/cost, targets and any scientific
   test require separate prospective authority; this plan admits none.
"""

from dataclasses import dataclass

import numpy as np

from feature_second_jet_reference import hermite_response, normal_cdf


CURRENT_SOURCE_SHA256 = "409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f"
FEATURE_HELPER_SHA256 = "dd4484fc3baaefb783ff54497b43ff801214e0583cd16d28a9d9b4c212cb218f"
MOMENT_AUDIT_SHA256 = "b5bb2ae1b683d6139f3a3b4a0e743343383c97d07294f68569367e3a35e94dc1"
# These are provenance declarations, not an import-time filesystem check.


def _finite(value, name):
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite reference arithmetic: " + name)
    return value


def _copy64(value, shape, name):
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {array.shape}")
    return _finite(array, name)


def _vector(value, n, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape not in ((), (n,)):
        raise ValueError(f"{name}: expected scalar or shape {(n,)}")
    return _finite(np.broadcast_to(array, (n,)).copy(), name)


@dataclass(frozen=True)
class Reference:
    """Copied fixed inputs/tables; caller must not mutate the contained arrays."""

    mu: np.ndarray
    raw_v: np.ndarray
    vref: np.ndarray
    sigma: np.ndarray
    pre_k3: np.ndarray
    pre_k4: np.ndarray
    rho: np.ndarray
    t21: np.ndarray
    t22: np.ndarray
    t31: np.ndarray
    t3: np.ndarray
    t4: np.ndarray
    A: tuple[np.ndarray, ...]
    H: tuple[np.ndarray, ...]
    below_sigma_floor_count: int


def prepare_reference(mu, pre_cov, k21, k4, k22):
    """Copy a current pre-write state; k22 is the explicit current override.

    Positive but below-floor raw variances may be represented for the literal
    pair-polynomial source check, but covariance_delta rejects that domain.
    Nonpositive raw variances are rejected before the current raw-v divisions.
    No covariance symmetry, PSD or correlation clipping is silently imposed.
    """
    mu = np.array(mu, dtype=np.float64, copy=True)
    if mu.ndim != 1 or not len(mu):
        raise ValueError("mu must be a nonempty vector")
    _finite(mu, "mu")
    n = len(mu)
    cov = _copy64(pre_cov, (n, n), "pre_cov")
    third = _copy64(k21, (n, n), "k21")
    fourth22 = _copy64(k22, (n, n), "k22")
    fourth = _vector(k4, n, "k4")
    raw_v = np.diag(cov).copy()
    if np.any(raw_v <= 0):
        raise ValueError("Nonpositive raw variance: no exact-functional action or denominator repair")
    below = int(np.count_nonzero(raw_v < 1e-10))
    sigma = np.sqrt(np.maximum(raw_v, 1e-10))
    vref = sigma * sigma
    alpha = mu / sigma
    density = np.exp(-.5 * alpha * alpha) / np.sqrt(2 * np.pi)
    gate = normal_cdf(alpha)
    gaussian_mean = sigma * density + mu * gate
    A = tuple(hermite_response(alpha, density, gate, gaussian_mean / sigma, order=6))
    H = (gaussian_mean, gate) + tuple(A[r] / sigma ** (r-1) for r in range(2, 7))
    for r, value in enumerate(A + H):
        _finite(value, f"Gaussian response table {r}")
    rho = cov / np.outer(sigma, sigma)
    np.fill_diagonal(rho, 0.)
    t21 = third / (raw_v[:, None] * sigma[None, :])
    np.fill_diagonal(t21, 0.)
    t22 = fourth22 / np.outer(raw_v, raw_v)
    pre_k3 = np.diag(third).copy()
    t3 = pre_k3 / sigma ** 3
    t4 = fourth / raw_v ** 2
    t31 = t4[:, None] * rho
    for name, value in (("rho", rho), ("t21", t21), ("t22", t22),
                        ("t3", t3), ("t4", t4), ("t31", t31)):
        _finite(value, name)
    return Reference(mu, raw_v, vref, sigma, pre_k3, fourth, rho, t21, t22,
                     t31, t3, t4, A, H, below)


def marginal_second_delta(reference, a, b, order):
    """E[(psi²)^(order) - (ReLU²)^(order)], coefficients/reference fixed.

    Only orders0,3,4 are consumed by covariance_delta.  Support0..4 makes the
    exact polynomial-square coefficients explicit; no negative H is read.
    """
    if type(order) is not int or not 0 <= order <= 4:
        raise ValueError("Marginal second-moment derivative order must be0..4")
    n = len(reference.mu)
    a, b = _vector(a, n, "a"), _vector(b, n, "b")
    v, H, r = reference.vref, reference.H, order
    cross = b*v*v*H[r+2] + a*v*H[r+1]
    if r:
        cross = cross + 2*r*b*v*H[r] + r*a*H[r-1]
    if r >= 2:
        cross = cross + r*(r-1)*b*H[r-2]
    if r == 0:
        square = a*a*v + 2*b*b*v*v
    elif r == 1:
        square = 4*a*b*v
    elif r == 2:
        square = 2*a*a + 8*b*b*v
    elif r == 3:
        square = 12*a*b
    else:
        square = 24*b*b
    return _finite(2*cross + square, f"marginal dM2[{r}]")


def covariance_delta(reference, a, b):
    """Expanded current pair11 delta plus the separately restored scalar face.

    This fixed local action has no below-floor same-functional interpretation;
    reject it rather than silently replacing raw-v normalizers by sigma².
    The returned covariance is neither masked nor symmetrized.  No head is
    called and no input/Reference array is modified.
    """
    if reference.below_sigma_floor_count:
        raise ValueError("Below-sigma-floor reference excluded from the covariance action")
    n = len(reference.mu)
    a, b = _vector(a, n, "a"), _vector(b, n, "b")
    s, A, rho = reference.sigma, reference.A, reference.rho
    t21, t22, t31, t3, t4 = reference.t21, reference.t22, reference.t31, reference.t3, reference.t4
    c = 2*b*s  # The ONLY other nonzero normalized response change is dA[2].
    outer = np.outer
    d11 = outer(a, A[1]) + outer(A[1], a) + outer(a, a)
    d22 = outer(c, A[2]) + outer(A[2], c) + outer(c, c)
    d21 = outer(c, A[1]) + outer(A[2], a) + outer(c, a)
    rho2half = rho*rho/2
    delta = rho*d11 + rho2half*d22
    # Mixed r=0: both K21 orientations, K22, and both K31 orientations.
    delta = delta + (t21*d21 + t21.T*d21.T)/2 + t22*d22/4
    delta = delta + (t31*outer(A[3], a) + t31.T*outer(a, A[3]))/6
    # Mixed r=1; K22 has no surviving dA at this order.
    order1 = (t21*outer(A[3], c) + t21.T*outer(c, A[3]))/2
    order1 = order1 + (t31*outer(A[4], c) + t31.T*outer(c, A[4]))/6
    order1 = order1 + (outer(t3*A[4], a) + outer(a, t3*A[4]))/6
    order1 = order1 + (outer(t4*A[5], a) + outer(a, t4*A[5]))/24
    delta = delta + rho*order1
    # At r=2 only the marginal-kappa endpoints survive; all mixed legs>=3.
    order2 = (outer(t3*A[5], c) + outer(c, t3*A[5]))/6
    order2 = order2 + (outer(t4*A[6], c) + outer(c, t4*A[6]))/24
    delta = (delta + rho2half*order2) * outer(s, s)
    raw = {r: marginal_second_delta(reference, a, b, r) for r in (0, 3, 4)}
    diagonal = raw[0] + reference.pre_k3*raw[3]/6 + reference.pre_k4*raw[4]/24
    np.fill_diagonal(delta, diagonal)
    _finite(delta, "covariance delta")
    return dict(covariance=delta, diagonal_delta=diagonal,
                marginal_raw_second_derivative_deltas=raw,
                mean_delta=np.zeros(n, dtype=np.float64),
                below_sigma_floor_count=reference.below_sigma_floor_count,
                scalar_functional_domain_qualified=True, joint_psd_certified=False,
                interpretation="Covariance-only current truncated-functional delta; not a full model or native bill.")


def full_pair11_reference(reference, a=0., b=0.):
    """Separate DENSE full current pair11 builder for a future cross-check.

    Returns the physical polynomial BEFORE scalar diagonal replacement.
    Compare new-minus-old only OFF DIAGONAL; independently check the scalar
    diagonal with fixed-polynomial one-dimensional integration as above.
    This intentionally does not reuse covariance_delta's sparse expansion.
    Below-floor output is only the literal current-normalizer polynomial,
    NOT a common Gaussian/Edgeworth-law assertion or eligible local action.
    """
    n = len(reference.mu)
    a, b = _vector(a, n, "a"), _vector(b, n, "b")
    A = list(reference.A)
    A[1] = A[1] + a
    A[2] = A[2] + 2*b*reference.sigma
    rho, t21, t22, t31 = reference.rho, reference.t21, reference.t22, reference.t31
    t3, t4 = reference.t3, reference.t4
    answer = np.zeros((n, n), dtype=np.float64)
    rr = np.ones((n, n), dtype=np.float64)
    for r in range(7):
        if r:
            answer = answer + rr*np.outer(A[r], A[r])
        if r <= 2:
            ec = (t21*np.outer(A[r+2], A[r+1]) + t21.T*np.outer(A[r+1], A[r+2]))/2
            ec = ec + t22*np.outer(A[r+2], A[r+2])/4
            ec = ec + (t31*np.outer(A[r+3], A[r+1]) + t31.T*np.outer(A[r+1], A[r+3]))/6
            if r:
                ec = ec + (np.outer(t3*A[r+3], A[r]) + np.outer(A[r], t3*A[r+3]))/6
                ec = ec + (np.outer(t4*A[r+4], A[r]) + np.outer(A[r], t4*A[r+4]))/24
            answer = answer + rr*ec
        if r < 6:
            rr = rr*rho/(r+1)
    return _finite(answer*np.outer(reference.sigma, reference.sigma), "full pair11 polynomial")
