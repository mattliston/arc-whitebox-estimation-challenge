"""Inert native-point BLOCK lift of the current409f thirty input features.

This is an offline reference, not an estimator or operation-level AD.  Callers
provide the independently extracted original native functions and the frozen
quadratic-reference ``raw_jets`` function.  This module performs no IO, loads no
payload, and runs no numerical work at import time.

``prepare(context, native_env, ideal_raw_jets)`` copies the supplied frame and
independently byte-replays its scalar/pair/features endpoints before accepting
anchors.  ``features(prepared)`` returns the original float32 feature bytes and
float64 first/second derivatives.  ``lift(prepared, shift)`` evaluates the SAME
saved-base continuous lift, not a Taylor polynomial or a re-anchored replay.
Entry i of ``shift`` changes ONLY mu_i when producing feature row i: these are
separate virtual paths, not one simultaneous multivariate perturbation.  In
particular the left covariance rows need not form a symmetric shifted matrix.

Partition, fixed before the certificate: scalar Gaussian/Edgeworth moments and
normalized A responses are ideal composite changes anchored at native values;
off-diagonal covariance is an affine LEFT-response block at native FP32 right
responses and rounded constant-only products; its diagonal is the TOTAL scalar
variance response.  Post-floor/normalization/clip/RMS blocks use native base
operands.  Positive above-floor diagonal correlation is a stationary composite
with its actual native ratio retained.  All thirty final feature casts have the
same additive anchor.  Exact moving boundaries are reported, never smoothed.
"""
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np


BANK_SHA256 = '409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f'
IDEAL_SOURCE_SHA256 = 'dd4484fc3baaefb783ff54497b43ff801214e0583cd16d28a9d9b4c212cb218f'
SCALAR_KEYS = ('alpha', 'pdf', 'gate', 'gaussian_mean', 'mean',
               'post_v', 'post_k3', 'post_k4')
AUX_KEYS = ('sigma', 'alpha', 'pdf', 'gate', 'gaussian_mean', 'pre_v',
            'pre_k3', 'pre_k4', 'post_v', 'post_k3', 'post_k4')


@dataclass(frozen=True)
class Jet:
    value: np.ndarray
    first: np.ndarray
    second: np.ndarray


@dataclass(frozen=True)
class FeatureJet:
    """``native`` is (mean, covariance, K21, radial, native_aux_mapping).

    ``covariance.value`` is the covariance block's native base.  ``aux`` has
    native-valued scalar Jet fields named by SCALAR_KEYS; sigma/pre_v/pre_cov/
    pre_k3/pre_k4 are plain immutable arrays.  Neither this result nor its
    derivatives modify the parent state or the caller's captured frame.
    """
    value: np.ndarray
    first: np.ndarray
    second: np.ndarray
    aux: object
    covariance: Jet
    native: tuple
    diagnostics: dict
    cost: dict


@dataclass(frozen=True)
class Prepared:
    frame: object
    layer: int
    n: int
    native: tuple
    native_features: np.ndarray
    native_uncast: np.ndarray
    aux: object
    ideal_base: object
    ideal_response: np.ndarray
    native_response: np.ndarray
    coefficients: np.ndarray
    postscale: np.ndarray
    postdenominator: np.ndarray
    postratio_unclipped: np.ndarray
    ideal_raw_jets: object
    replay: dict
    cost: dict


def _copy(value, dtype=None):
    out = np.array(value, dtype=dtype, copy=True, order='C')
    if not np.all(np.isfinite(out)):
        raise ValueError('Nonfinite source or intermediate')
    out.setflags(write=False)
    return out


def _anchor(base, increment):
    """Uniform additive block residual, including native signed-zero bits."""
    return np.where(increment == 0, base, base + increment)


def _same_bytes(name, actual, expected, records):
    actual, expected = np.asarray(actual), np.asarray(expected)
    ok = (actual.dtype == expected.dtype and actual.shape == expected.shape
          and actual.tobytes(order='C') == expected.tobytes(order='C'))
    records[name] = dict(byte_exact=bool(ok), dtype=str(actual.dtype),
                         shape=list(actual.shape))
    if not ok:
        raise ValueError('Native replay byte mismatch: ' + name)


def _jet(native, ideal):
    return Jet(_copy(native, np.float64), _copy(ideal.first, np.float64),
               _copy(ideal.second, np.float64))


def _responses(ideal):
    """A_r=sigma**(r-1) E_G ReLU**(r), through r=8, ideal scalar block."""
    alpha = ideal['alpha'].value
    density = ideal['pdf'].value
    table = [ideal['gaussian_mean'].value / ideal['sigma'],
             ideal['gate'].value, density, -alpha * density]
    hm2, hm1 = np.ones_like(alpha), alpha
    for degree in range(2, 7):
        h = alpha * hm1 - (degree - 1) * hm2
        table.append((-1) ** degree * h * density)
        hm2, hm1 = hm1, h
    return np.stack(table, axis=0)


def _coefficient_planes(frame, aux, native_response):
    """Pack the six nonzero LEFT A_r coefficients of the native pair11 lift.

    Right operands/constants are frozen in their original rounding order.
    In particular T31 is formed before the independent float32 casts, rr is
    updated sequentially in float32, and t3*G / t4*G are formed in float32.
    Products containing the moving LEFT response are not frozen constants.
    Every subsequent coefficient accumulation is explicit float64 arithmetic.
    """
    s, v = aux['sigma'], aux['pre_v']
    n = len(s)
    rho64 = frame['pre_cov'] / np.outer(s, s)
    np.fill_diagonal(rho64, 0)
    t21 = frame['pre_k21'] / (v[:, None] * s[None, :])
    np.fill_diagonal(t21, 0)
    t22 = frame['pre_k22'] / np.outer(v, v)
    t3 = aux['pre_k3'] / s ** 3
    t4 = np.broadcast_to(frame['pre_k4'], (n,)) / v ** 2
    t31 = t4[:, None] * rho64
    rho = rho64.astype(np.float32)
    t21, t22, t31, t3, t4 = (
        x.astype(np.float32) for x in (t21, t22, t31, t3, t4))
    right = native_response
    # Seven slots make the exact A index explicit; slot zero is never used.
    coeff = np.zeros((7, n, n), np.float64)
    rr = np.ones_like(rho)

    def add(index, field, right_vector, divisor=1., left_vector=None):
        value = field.astype(np.float64) * right_vector.astype(np.float64)[None, :]
        if left_vector is not None:
            value = value * left_vector.astype(np.float64)[:, None]
        coeff[index] += value / divisor

    for r in range(7):
        rr64 = rr.astype(np.float64)
        if r:
            add(r, rr64, right[r])
        if r <= 2:
            add(r + 2, rr64 * t21, right[r + 1], 2.)
            add(r + 1, rr64 * t21.T, right[r + 2], 2.)
            add(r + 2, rr64 * t22, right[r + 2], 4.)
            add(r + 3, rr64 * t31, right[r + 1], 6.)
            add(r + 1, rr64 * t31.T, right[r + 3], 6.)
            if r:
                add(r + 3, rr64, right[r], 6., t3)
                add(r, rr64, t3 * right[r + 3], 6.)
                add(r + 4, rr64, right[r], 24., t4)
                add(r, rr64, t4 * right[r + 4], 24.)
        # Also reproduce the final original (unused) update; no changed powers.
        rr = rr * rho / (r + 1)
        if rr.dtype != np.float32:
            raise TypeError('Native rho-power arithmetic must remain float32')
    coeff *= np.outer(s, s)[None, :, :]
    return _copy(coeff[1:])


def _action(prepared, left_vectors):
    """Six physical coefficient planes acting on six per-row A increments."""
    out = np.zeros((prepared.n, prepared.n), np.float64)
    for r in range(6):
        out += prepared.coefficients[r] * left_vectors[r + 1, :, None]
    return out


def _native_ratio(aux, covariance):
    postscale = np.sqrt(np.maximum(aux['post_v'], 1e-12))
    denominator = np.outer(postscale, postscale)
    return postscale, denominator, covariance / denominator


def _values(prepared, scalar, covariance, shift):
    """Uncast feature values for independent own-mean paths, fixed base tape."""
    s = prepared.aux['sigma']
    v = s * s
    value = prepared.native_uncast.copy()
    for column, name in ((3, 'alpha'), (5, 'gate'), (6, 'pdf')):
        value[:, column] = scalar[name]
    value[:, 7] = scalar['gaussian_mean'] / s
    value[:, 8] = scalar['mean'] / s
    value[:, 9] = (scalar['mean'] - scalar['gaussian_mean']) / s
    value[:, 11] = scalar['post_v'] / v
    value[:, 20] = scalar['post_k3'] / (v * s)
    value[:, 21] = scalar['post_k4'] / (v * v)
    pre_rho_diagonal = prepared.aux['pre_v'] / (s * s)
    value[:, 26] = _anchor(prepared.native_uncast[:, 26],
                           pre_rho_diagonal * shift / (prepared.n * s))

    floor = np.maximum(scalar['post_v'], 1e-12)
    scale = np.sqrt(floor)
    # Right endpoints stay at their native values; the diagonal is total.
    ratio = covariance / (scale[:, None] * prepared.postscale[None, :])
    base_v = prepared.aux['post_v'].value
    base_composite = base_v / np.maximum(base_v, 1e-12)
    diagonal = _anchor(np.diag(prepared.postratio_unclipped),
                       scalar['post_v'] / floor - base_composite)
    np.fill_diagonal(ratio, diagonal)
    ratio = np.clip(ratio, -1, 1)
    value[:, 27] = np.sqrt(np.mean(ratio * ratio, axis=1))
    value[:, 28] = np.mean(ratio, axis=1)
    value[:, 29] = _anchor(prepared.native_uncast[:, 29],
                           (scalar['mean'] - prepared.aux['mean'].value)
                           / (prepared.n * s))
    return value


def prepare(context, native_env, ideal_raw_jets):
    """Copy a captured frame, replay original endpoints, and freeze one tape.

    ``context`` is the original capture's flat field mapping plus ``layer``
    and the matching ``weights``.  ``native_env`` is the dictionary returned
    by native_reference.original_functions(np), with no dtype changes.  The
    caller remains responsible for source/capture/weight hash authentication.
    None of the caller's arrays are modified or retained as mutable aliases.
    """
    required = ('mu', 'pre_cov', 'pre_k21', 'pre_k4', 'pre_k22', 'weights',
                'host_mean', 'host_cov', 'host_k21',
                'pair_radial_before_omitted', 'features')
    frame = {k: _copy(context[k]) for k in required}
    frame.update({'aux_' + k: _copy(context['aux_' + k]) for k in AUX_KEYS})
    layer, n = int(context['layer']), len(frame['mu'])
    if not 1 <= layer <= 14:
        raise ValueError('This reference owns the ordinary full-pair interior map only')
    if frame['mu'].shape != (n,) or frame['features'].shape != (n, 30):
        raise ValueError('Invalid mean/feature shape')
    for name in ('pre_cov', 'pre_k21', 'pre_k22', 'weights', 'host_cov', 'host_k21'):
        if frame[name].shape != (n, n):
            raise ValueError('Invalid matrix shape: ' + name)
    for name in ('mu', 'pre_cov', 'pre_k21', 'pre_k4', 'pre_k22'):
        if frame[name].dtype != np.float64:
            raise TypeError('Expected native float64 physical source: ' + name)
    if frame['features'].dtype != np.float32 or frame['weights'].dtype != np.float32:
        raise TypeError('Expected native float32 features and weights')
    if np.any(np.diag(frame['pre_cov']) == 0):
        raise ValueError('Current raw-v denominators are undefined at zero variance')
    names = ('_nonlinear_pair_batch', 'derivatives', 'features', 'features_uncast')
    if not all(callable(native_env.get(k)) for k in names) or not callable(ideal_raw_jets):
        raise TypeError('Original native functions and ideal scalar callback are required')

    # No native callback can modify the owned source tape or caller arrays.
    native = native_env['_nonlinear_pair_batch'](
        frame['mu'].copy(), frame['pre_cov'].copy(), frame['pre_k21'].copy(),
        frame['pre_k4'].copy(), k22_override=frame['pre_k22'].copy(),
        metric='covariance')
    if not isinstance(native, tuple) or len(native) != 5:
        raise ValueError('Unexpected original pair return')
    mean, covariance, k21, radial, native_aux = native
    replay = {}
    for name, actual in (('host_mean', mean), ('host_cov', covariance),
                         ('host_k21', k21), ('pair_radial_before_omitted', radial)):
        _same_bytes(name, actual, frame[name], replay)
    _same_bytes('aux_pre_cov', native_aux['pre_cov'], frame['pre_cov'], replay)
    for k in AUX_KEYS:
        _same_bytes('aux_' + k, native_aux[k], frame['aux_' + k], replay)
    native_x = native_env['features'](layer, mean, covariance, frame['pre_k21'],
                                     native_aux, frame['weights'])
    uncast = native_env['features_uncast'](layer, mean, covariance, frame['pre_k21'],
                                         native_aux, frame['weights'])
    _same_bytes('features', native_x, frame['features'], replay)
    _same_bytes('features_uncast_cast', np.asarray(uncast).astype(np.float32),
                frame['features'], replay)
    if np.asarray(uncast).dtype != np.float64:
        raise TypeError('Expected native uncast float64 feature stack')

    native_aux = {k: _copy(v) for k, v in native_aux.items()}
    native = (_copy(mean), _copy(covariance), _copy(k21), _copy(radial),
              MappingProxyType(native_aux))
    ideal = ideal_raw_jets(frame['mu'], frame['pre_cov'], frame['pre_k21'],
                          frame['pre_k4'])
    for k in ('sigma', 'pre_v', 'pre_k3', 'pre_k4'):
        _same_bytes('ideal_fixed_' + k, ideal[k], native_aux[k], replay)
    aux = {k: _copy(v) for k, v in native_aux.items() if k not in SCALAR_KEYS}
    for k in SCALAR_KEYS:
        aux[k] = _jet(mean if k == 'mean' else native_aux[k], ideal[k])
    ideal_base = {k: _copy(ideal[k].value, np.float64) for k in SCALAR_KEYS}
    ideal_response = _copy(_responses(ideal))
    aa, _ = native_env['derivatives'](
        native_aux['alpha'], native_aux['pdf'], native_aux['gate'],
        native_aux['gaussian_mean'] / native_aux['sigma'], 6)
    native_response = _copy(np.stack(aa, axis=0).astype(np.float32))
    coefficients = _coefficient_planes(frame, native_aux, native_response)
    scale, denominator, ratio = _native_ratio(native_aux, native[1])
    cost = dict(
        scope='Static offline block inventory, NOT a measured FNP or complete-card bill',
        complete_flops=None, width=n, physical_coefficient_planes=6,
        physical_coefficient_bytes=int(coefficients.nbytes),
        native_rho_power_updates=7,
        classical_scalar_ops_native_rho_updates=14 * n * n,
        two_covariance_tangent_actions=dict(
            elementwise_multiplies=12 * n * n, accumulation_adds=12 * n * n),
        one_shifted_covariance_action=dict(
            elementwise_multiplies=6 * n * n, accumulation_adds=6 * n * n),
        uniform_anchor_rule='add + equality + where; not free, unmetered here',
        excluded=['coefficient preparation and storage/copies',
                  'original three-channel pair/scalar replay and feature replay',
                  'ideal scalar/Hermite jets and displaced evaluations',
                  'normalization, clipping, RMS, casts, and diagnostics',
                  'GNN/messages, quadratic moment action, all feedback/descendants'])
    prepared = Prepared(MappingProxyType(frame), layer, n, native,
                        _copy(native_x), _copy(uncast), MappingProxyType(aux),
                        MappingProxyType(ideal_base), ideal_response,
                        native_response, coefficients, _copy(scale),
                        _copy(denominator), _copy(ratio), ideal_raw_jets,
                        replay, cost)
    scalar = {k: aux[k].value for k in SCALAR_KEYS}
    base = _values(prepared, scalar, native[1], np.zeros(n))
    _same_bytes('own_path_uncast_base', base, prepared.native_uncast, replay)
    _same_bytes('lift_zero_base', _anchor(native_x.astype(np.float64), base - uncast),
                native_x.astype(np.float64), replay)
    return prepared


def _postratio_jets(prepared, covariance):
    """Normalization derivatives at actual native base operands, with TOTAL diag."""
    var = prepared.aux['post_v']
    scale = prepared.postscale
    active = var.value > 1e-12
    scale_first = np.where(active, var.first / (2 * scale), 0.)
    scale_second = np.where(active, var.second / (2 * scale)
                            - var.first * var.first / (4 * scale ** 3), 0.)
    q1, q2 = scale_first / scale, scale_second / scale
    raw = prepared.postratio_unclipped
    c1 = covariance.first / prepared.postdenominator
    first = c1 - raw * q1[:, None]
    second = (covariance.second / prepared.postdenominator
              - 2 * c1 * q1[:, None]
              + raw * (2 * q1 * q1 - q2)[:, None])
    # Above the floor, var/max(var,floor) is exactly stationary.  Retain the
    # actual native var/(sqrt(var)*sqrt(var)) VALUE via its frozen residual.
    np.fill_diagonal(first, np.where(active, 0., var.first / 1e-12))
    np.fill_diagonal(second, np.where(active, 0., var.second / 1e-12))
    inside = (raw > -1) & (raw < 1)
    ratio = np.clip(raw, -1, 1)
    first, second = first * inside, second * inside
    mean_square = np.mean(ratio * ratio, axis=1)
    ms1 = 2 * np.mean(ratio * first, axis=1)
    ms2 = 2 * np.mean(first * first + ratio * second, axis=1)
    rms = np.sqrt(mean_square)
    nonzero = mean_square > 0
    safe = np.where(nonzero, rms, 1.)
    rms1 = np.where(nonzero, ms1 / (2 * safe), 0.)
    rms2 = np.where(nonzero, ms2 / (2 * safe) - ms1 * ms1 / (4 * safe ** 3), 0.)
    diagonal = np.eye(prepared.n, dtype=bool)
    on_clip = np.abs(raw) == 1
    stationary_diagonal = diagonal & active[:, None]
    off_boundary = int(np.count_nonzero(on_clip & ~diagonal))
    diag_boundary = int(np.count_nonzero(on_clip & diagonal & ~stationary_diagonal))
    floor_boundary = int(np.count_nonzero(var.value == 1e-12))
    rms_moving = int(np.count_nonzero((~nonzero) &
                     (np.any(first != 0, axis=1) | np.any(second != 0, axis=1))))
    details = dict(
        fixed_pre_sigma_floor=int(np.count_nonzero(prepared.aux['pre_v'] == 1e-10)),
        below_pre_sigma_floor=int(np.count_nonzero(prepared.aux['pre_v'] < 1e-10)),
        post_variance_floor=floor_boundary,
        negative_post_variance=int(np.count_nonzero(var.value < 0)),
        offdiagonal_preclip_equality=off_boundary,
        diagonal_moving_preclip_equality=diag_boundary,
        diagonal_stationary_preclip_equality=int(np.count_nonzero(on_clip & stationary_diagonal)),
        already_strictly_clipped=int(np.count_nonzero(np.abs(raw) > 1)),
        zero_rms=int(np.count_nonzero(~nonzero)),
        nonstationary_zero_rms=rms_moving,
        moving_boundary_count=floor_boundary + off_boundary + diag_boundary + rms_moving,
        boundary_convention='Conservatively reject every off-diagonal exact preclip equality; '
                            'do not count stationary above-floor positive diagonals')
    return Jet(ratio, first, second), Jet(rms, rms1, rms2), details


def features(prepared):
    """Return native float32 x plus diagonal first/second block derivatives."""
    if not isinstance(prepared, Prepared):
        raise TypeError('A fixed Prepared tape is required')
    aux, s, n = prepared.aux, prepared.aux['sigma'], prepared.n
    dA = prepared.ideal_response[1:8] / s[None, :]
    d2A = prepared.ideal_response[2:9] / (s * s)[None, :]
    c1, c2 = _action(prepared, dA), _action(prepared, d2A)
    np.fill_diagonal(c1, aux['post_v'].first)
    np.fill_diagonal(c2, aux['post_v'].second)
    covariance = Jet(prepared.native[1], _copy(c1), _copy(c2))
    ratio, rms, boundaries = _postratio_jets(prepared, covariance)
    first, second = np.zeros((n, 30)), np.zeros((n, 30))
    for order, output in (('first', first), ('second', second)):
        for column, name in ((3, 'alpha'), (5, 'gate'), (6, 'pdf')):
            output[:, column] = getattr(aux[name], order)
        output[:, 7] = getattr(aux['gaussian_mean'], order) / s
        output[:, 8] = getattr(aux['mean'], order) / s
        output[:, 9] = (getattr(aux['mean'], order)
                        - getattr(aux['gaussian_mean'], order)) / s
        output[:, 11] = getattr(aux['post_v'], order) / (s * s)
        output[:, 20] = getattr(aux['post_k3'], order) / (s * s * s)
        output[:, 21] = getattr(aux['post_k4'], order) / ((s * s) * (s * s))
        output[:, 27] = getattr(rms, order)
        output[:, 28] = np.mean(getattr(ratio, order), axis=1)
        output[:, 29] = getattr(aux['mean'], order) / (n * s)
    first[:, 26] = (aux['pre_v'] / (s * s)) / (n * s)
    diagnostics = dict(boundaries, native_replay=prepared.replay,
                       all_native_replays_byte_exact=all(v['byte_exact'] for v in prepared.replay.values()),
                       all_finite=bool(np.all(np.isfinite(first)) and np.all(np.isfinite(second))
                                       and np.all(np.isfinite(ratio.value))
                                       and np.all(np.isfinite(rms.value))),
                       derivative_convention='saved-base native BLOCK lift, not derivative of FP32 program',
                       virtual_path='separate row i derivative with respect to mu_i only',
                       below_floor_covariance_action_excluded=boundaries['below_pre_sigma_floor'] > 0)
    return FeatureJet(prepared.native_features, _copy(first), _copy(second),
                      aux, covariance, prepared.native, diagnostics, prepared.cost)


def lift(prepared, shift):
    """Saved-base continuous feature function for finite differences.

    This evaluates ideal composite blocks at displaced means and subtracts
    their SAVED ideal base values.  No captured/native endpoint is recomputed
    or re-anchored.  A length-n vector denotes n separate row-wise own-mean
    paths.  It is not a Taylor polynomial, joint mean shift, or model replay.
    """
    if not isinstance(prepared, Prepared):
        raise TypeError('A fixed Prepared tape is required')
    shift = np.asarray(shift, np.float64)
    if shift.shape != (prepared.n,) or not np.all(np.isfinite(shift)):
        raise ValueError('shift must be a finite length-n vector')
    frame = prepared.frame
    ideal = prepared.ideal_raw_jets(frame['mu'] + shift, frame['pre_cov'],
                                    frame['pre_k21'], frame['pre_k4'])
    scalar = {k: _anchor(prepared.aux[k].value,
                         ideal[k].value - prepared.ideal_base[k]) for k in SCALAR_KEYS}
    response_increment = _responses(ideal) - prepared.ideal_response
    covariance = _anchor(prepared.native[1], _action(prepared, response_increment))
    np.fill_diagonal(covariance, scalar['post_v'])
    uncast = _values(prepared, scalar, covariance, shift)
    value = _anchor(prepared.native_features.astype(np.float64),
                    uncast - prepared.native_uncast)
    if not np.all(np.isfinite(value)):
        raise ValueError('Nonfinite displaced feature lift')
    return value
