"""Inert FNP fragment: entry-frozen signed-pair metric, no imports or arrays.

Only the future exact-source composer installs these helpers in the current
participant namespace. All array work below belongs inside paid prediction.
The physical coefficient bank always remains in its original x/y coordinates.
"""


def _pair256_finite(value, name, dtype='float32'):
    if str(value.dtype) != dtype or not bool(fnp.all(fnp.isfinite(value))):
        raise RuntimeError('Pair256 nonfinite or wrong dtype: ' + name)
    return value


def _pair256_coordinates(matrix, check_finite=True):
    """A separate metric-only [midpoint,difference] copy, never physical F.

    This is explicitly FP32 arithmetic. In real arithmetic its inverse is
    x=midpoint+difference, y=midpoint-difference. We do NOT apply that inverse
    to stored physical factors or claim a bit-exact FP32 round trip.
    """
    if len(matrix.shape) != 2 or str(matrix.dtype) != 'float32':
        raise RuntimeError('Pair256 coordinate matrix contract')
    columns = int(matrix.shape[1])
    if columns <= 0 or columns % 2:
        raise RuntimeError('Pair256 requires a complete nonempty paired block')
    pairs = columns // 2
    half = fnp.asarray((0.5,), dtype=fnp.float32)
    midpoint = (matrix[:, :pairs] + matrix[:, pairs:]) * half
    difference = (matrix[:, :pairs] - matrix[:, pairs:]) * half
    result = fnp.concatenate([midpoint, difference], axis=1)
    del midpoint, difference, half
    # Entry norms below certify every matrix entry through finite sums of
    # nonnegative FP64 squares. Do not charge a redundant entry-wide scan;
    # projected metric factors have no such norm reduction and keep the scan.
    if check_finite:
        _pair256_finite(result, 'metric-only midpoint/difference copy')
    return result


def _pair256_psd(k00, k01, k11, tolerance, name):
    """Fixed rounding-error guard, never a PSD projection or eigenvalue floor.

    The exact polynomial is PSD. The multiplication comparisons below admit
    only the declared relative floating-point envelope; no coefficient is
    changed in response. All constants are typed one-element arrays.
    """
    if any(str(value.dtype) != 'float64' for value in (k00, k01, k11, tolerance)):
        raise RuntimeError('Pair256 PSD guard requires FP64 arrays')
    zero = fnp.zeros((1,), dtype=fnp.float64)
    one = fnp.ones((1,), dtype=fnp.float64)
    if not (bool(fnp.all(k00 >= zero)) and bool(fnp.all(k11 >= zero))):
        raise RuntimeError('Pair256 negative metric diagonal: ' + name)
    scale = fnp.maximum(k00, k11)
    denominator = fnp.where(scale > zero, scale, one)
    scaled00 = k00 / denominator
    scaled01 = k01 / denominator
    scaled11 = k11 / denominator
    for value in (scaled00, scaled01, scaled11):
        _pair256_finite(value, name + ' normalized PSD guard', 'float64')
    off_square = scaled01 * scaled01
    diagonal_product = scaled00 * scaled11
    if not bool(fnp.all(off_square <= diagonal_product * tolerance)):
        raise RuntimeError('Pair256 metric exceeds fixed PSD rounding envelope: ' + name)


def _pair256_metric(joiner, coefficient):
    """Return a metric-only FP32 basis and its three FP32 paired-block rows.

    joiner=[x,y] and coefficient=[a,-a] are the ACTUAL dense entry arrays.
    Norms use the rounded FP32 midpoint/difference, widened to FP64. The
    small block is evaluated in FP64 and cast once; the matrix products
    remain FP32. These weights are frozen at this cohort's sole tier entry.
    """
    if (len(joiner.shape) != 2 or len(coefficient.shape) != 1
            or str(joiner.dtype) != 'float32'
            or str(coefficient.dtype) != 'float32'
            or int(joiner.shape[1]) != int(coefficient.shape[0])):
        raise RuntimeError('Pair256 entry shape/dtype contract')
    columns = int(joiner.shape[1])
    if columns <= 0 or columns % 2:
        raise RuntimeError('Pair256 entry is not a complete signed cohort')
    pairs = columns // 2
    _pair256_finite(coefficient, 'actual signed coefficients')
    if not bool(fnp.all(coefficient[:pairs] == -coefficient[pairs:])):
        raise RuntimeError('Pair256 actual coefficient opposition failed')

    metric_basis = _pair256_coordinates(joiner, check_finite=False)
    widened = metric_basis.astype(fnp.float64)
    middle = widened[:, :pairs]
    difference = widened[:, pairs:]
    uu = fnp.sum(middle * middle, axis=0)
    vv = fnp.sum(difference * difference, axis=0)
    uv = fnp.sum(middle * difference, axis=0)
    del widened, middle, difference

    # Exact typed constants, including the prospective 64-ULP relative
    # envelope factors. No integer indexing returns a RemoteScalar here.
    constants64 = fnp.asarray((0.0, 2.0, 4.0, 1.0000000000000142),
                              dtype=fnp.float64)
    constants32 = fnp.asarray((0.0, 1.0000076293945312), dtype=fnp.float32)
    zero64 = constants64[0:1]
    two64 = constants64[1:2]
    four64 = constants64[2:3]
    tolerance64 = constants64[3:4]
    zero32 = constants32[0:1]
    tolerance32 = constants32[1:2]
    for value in (uu, vv, uv):
        _pair256_finite(value, 'entry FP64 norm/cross', 'float64')
    if not (bool(fnp.all(uu >= zero64)) and bool(fnp.all(vv >= zero64))):
        raise RuntimeError('Pair256 negative squared norm')

    norm_product = uu * vv
    cross_square = uv * uv
    if not bool(fnp.all(cross_square <= norm_product * tolerance64)):
        raise RuntimeError('Pair256 entry norm Cauchy-Schwarz guard')
    aa = coefficient[:pairs].astype(fnp.float64)
    common = four64 * (aa * aa)
    k00_64 = common * (two64 * (norm_product + cross_square))
    k01_64 = common * (two64 * uv * (uu + vv))
    k11_64 = common * (uu * uu + vv * vv + two64 * cross_square)
    del aa, common, uu, vv, uv, norm_product, cross_square
    original = (k00_64, k01_64, k11_64)
    for value in original:
        _pair256_finite(value, 'entry FP64 block', 'float64')
    _pair256_psd(*original, tolerance64, 'FP64 entry block')

    kernel = tuple(value.astype(fnp.float32) for value in original)
    for before, after in zip(original, kernel):
        _pair256_finite(after, 'entry FP32 block')
        # There is no floor, rescue rescaling or deletion of a nonzero
        # coefficient which underflows during the prescribed cast.
        if not bool(fnp.all((before == zero64) | (after != zero32))):
            raise RuntimeError('Pair256 metric coefficient underflow')
    # Evaluate the PSD guard on the ACTUAL rounded coefficients, widened;
    # squaring tiny/large FP32 coefficients directly can under/overflow.
    rounded64 = tuple(value.astype(fnp.float64) for value in kernel)
    _pair256_psd(*rounded64, tolerance32.astype(fnp.float64), 'rounded FP32 entry block')
    return metric_basis, kernel


def _pair256_apply_kernel(values, kernel):
    """Apply the metric block to an RHS; original physical factors untouched."""
    if len(values.shape) != 2 or str(values.dtype) != 'float32' or len(kernel) != 3:
        raise RuntimeError('Pair256 metric RHS contract')
    rows = int(values.shape[0])
    if rows <= 0 or rows % 2:
        raise RuntimeError('Pair256 metric RHS paired rows')
    pairs = rows // 2
    for value in kernel:
        if tuple(value.shape) != (pairs,) or str(value.dtype) != 'float32':
            raise RuntimeError('Pair256 metric block/RHS shape mismatch')
    k00, k01, k11 = kernel
    upper, lower = values[:pairs, :], values[pairs:, :]
    first = k00[:, None] * upper + k01[:, None] * lower
    second = k01[:, None] * upper + k11[:, None] * lower
    result = fnp.concatenate([first, second], axis=0)
    del first, second, upper, lower
    return _pair256_finite(result, 'metric block applied RHS')


def _pair256_nonzero_range(value):
    _pair256_finite(value, 'pre-QR metric range')
    zero = fnp.zeros((1,), dtype=fnp.float32)
    if not bool(fnp.any(value != zero)):
        raise RuntimeError('Pair256 identically zero metric range')
