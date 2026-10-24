"""Inert, public-FNP-only birth-gauge size diagnostic; never changes the card.

The caller supplies borrowed current-native arrays. Only this module's own
temporary matrices may have their diagonals overwritten and restored.
There are no numerical imports or import-time arrays.
"""
import weakref

WIDTH = 1024
TEMP_LIMIT = 96 * 1024**2
VECTOR_NAMES = (
    'birth', 'beta', 'u', 'gap', 'prevariance', 'innovation', 'safe_birth', 'disc',
    'a64_original', 't64_original', 's64_original', 'a32_widened', 't32_widened',
    's32_widened', 'w_stored', 'z_stored', 'k_stored', 'alive', 'birth_mask',
    'birth_u_minus_beta', 'w_minus_birth_mask', 'z_minus_birth_u_mask',
    'diagonal_defect', 'u_unclipped', 'u_was_clipped', 'safe_birth_was_floored',
    'disc_was_floored', 'gap_was_capped', 'u_absolute_bound', 'disc_floor',
    'gap_cap', 'native_born_diagonal')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


class Live:
    """Weak ownership ledger for every explicit allocated diagnostic result.

    Transpose/basic-index views are nonallocating. Internal public-FNP/backend
    work remains in the actual context bill and independently watched RSS.
    """
    def __init__(self):
        self.refs = []
        self.peak = 0

    def own(self, value):
        require(hasattr(value, 'dtype') and hasattr(value, 'shape'), 'Public numerical result')
        size = int(value.nbytes)
        try:
            self.refs.append((weakref.ref(value), size))
        except TypeError:
            require(tuple(value.shape) == () and size <= 8, 'Only immediate scalar lacks weakref')
        live = sum(size for ref, size in self.refs if ref() is not None)
        self.peak = max(self.peak, live)
        require(live <= TEMP_LIMIT, 'At most96MiB explicit live diagnostic arrays')
        return value


def packet(value):
    return dict(dtype=str(value.dtype), shape=tuple(value.shape), data=value.tobytes())


def metric(fnp, live, value):
    """Meter full finite scan and diagonal/off-diagonal size, restoring OWN data."""
    own = live.own
    diagonal = own(fnp.diag(value).copy())
    fnp.fill_diagonal(value, 0.)
    try:
        finite = own(fnp.isfinite(value))
        all_finite = packet(own(fnp.all(finite)))
        del finite
        square = own(value * value)
        off2 = own(fnp.sum(square))
        del square
        absolute = own(fnp.abs(value))
        offmax = own(fnp.max(absolute))
        del absolute
        square = own(diagonal * diagonal)
        diag2 = own(fnp.sum(square))
        del square
        absolute = own(fnp.abs(diagonal))
        diagmax = own(fnp.max(absolute))
        del absolute
        total2 = own(off2 + diag2)
        totalmax = own(fnp.maximum(offmax, diagmax))
        result = dict(finite=all_finite, offdiag_sum_squares=packet(off2),
            diagonal_sum_squares=packet(diag2), total_sum_squares=packet(total2),
            offdiag_max_abs=packet(offmax), diagonal_max_abs=packet(diagmax),
            total_max_abs=packet(totalmax))
    finally:
        fnp.fill_diagonal(value, diagonal)
    return result


def _capture(fnp, values, live):
    own = live.own
    B, O = values['B'], values['O']
    mask, alive, S = values['live'], values['alive'], values['births']
    born, delta = values['born'], values['bd_delta']
    birth, beta, u, gap = (values[k] for k in ('birth', 'beta', 'u', 'gap'))
    a32, t32, s32 = (values[k] for k in ('a32', 't32', 's32'))
    w, z = values['w'], values['z']
    for matrix in (B, O, mask, born, delta):
        require(tuple(matrix.shape) == (WIDTH, WIDTH), 'Exact1024-square birth state')
    require(str(B.dtype) == str(O.dtype) == str(born.dtype) == str(delta.dtype) == 'float64',
            'Exact native FP64 physical-plane ownership')
    require(str(values['new32'].dtype) == 'float32', 'Actual returned stored FP32 common base')

    # Exact stored scalar order. w and z are borrowed from the actual BD return.
    tt = own(t32 * t32); ss = own(s32 * s32)
    ttt = own(tt * t32); sss = own(ss * s32)
    cubediff = own(ttt - sss); ka = own(a32 * cubediff); k = own(ka * S)
    del tt, ss, ttt, sss, cubediff, ka
    original_diagonal = own(fnp.diag(born).copy())
    diagonal_defect = own(k - original_diagonal)
    diagonal_defect = own(diagonal_defect * alive)

    # Old stored pair versus the actually assembled native born, before BD repair.
    square = own(O * O); left = own(square * w[None, :]); del square
    right = own(z[:, None] * O.T); plane = own(left + right); del left, right
    diff = own(plane - born); del plane
    E = own(diff * mask); del diff
    fnp.fill_diagonal(E, diagonal_defect)
    stats = dict(old_stored_defect=metric(fnp, live, E))

    # This is floor/clipping discrepancy, NOT a claim that changing gap alone
    # causes a diagonal defect. Exact real arithmetic has zero diagonal here.
    birth_u = own(birth * u); clip_defect = own(birth_u - beta)
    clip_weight = own(clip_defect * S)
    part = own(clip_weight[:, None] * B.T); clipping = own(part * mask); del part
    stats['clipping_floor_component'] = metric(fnp, live, clipping)
    remainder = own(E - clipping); del clipping

    # Scalar-storage comparison on the original B. Its residual below includes
    # base storage AND diagnostic/native FP64 reassociation, not just one cause.
    birth_mask = own(birth * S); birth_u_mask = own(birth_u * S)
    dw = own(w - birth_mask); dz = own(z - birth_u_mask)
    square = own(B * B); left = own(square * dw[None, :]); del square
    right = own(dz[:, None] * B.T); part = own(left + right); del left, right
    scalar = own(part * mask); del part
    fnp.fill_diagonal(scalar, diagonal_defect)
    stats['scalar_storage_component'] = metric(fnp, live, scalar)
    rest = own(remainder - scalar); del remainder, scalar
    stats['base_storage_and_reassociation_residual'] = metric(fnp, live, rest)
    del rest

    # Read-only duplicate of the upcoming born+delta addition. The actual card
    # still executes that original addition later; its original bill is retained.
    N = own(values['new32'].astype(fnp.float64))
    square = own(N * N); left = own(square * w[None, :]); del square
    right = own(z[:, None] * N.T); plane = own(left + right); del left, right, N
    assembled = own(born + delta)
    diff = own(plane - assembled); del plane, assembled
    post = own(diff * mask); del diff
    fnp.fill_diagonal(post, diagonal_defect)
    stats['post_BD_stored_defect'] = metric(fnp, live, post)
    assembly = own(post - E); del post, E
    stats['additional_BD_assembly_residual'] = metric(fnp, live, assembly)
    del assembly

    # Borrowed born is never mutated, even temporarily, for normalization.
    reference = own(born * mask)
    square = own(reference * reference)
    norm2 = packet(own(fnp.sum(square))); del square
    absolute = own(fnp.abs(reference))
    maximum = packet(own(fnp.max(absolute))); del absolute, reference
    stats['masked_native_born_reference'] = dict(total_sum_squares=norm2, total_max_abs=maximum)

    vv = values['vv']
    unclipped = own(beta / values['safe_birth'])
    u_clipped = own(unclipped != u)
    birth_floored = own(birth < 1e-12)
    floor_v = own(.01 * vv); disc_floor = own(floor_v * vv); del floor_v
    disc_floored = own(values['disc'] < disc_floor)
    capped_argument = own(fnp.maximum(values['disc'], disc_floor))
    uncapped_gap = own(fnp.sqrt(capped_argument)); del capped_argument
    gap_cap = own(16 * vv); gap_capped = own(uncapped_gap > gap_cap); del uncapped_gap
    u_bound = own(8 * vv)
    vector_values = (
        birth, beta, u, gap, vv, values['innovation'], values['safe_birth'], values['disc'],
        values['a'], values['t'], values['s'], a32, t32, s32, w, z, k, alive, S,
        clip_defect, dw, dz, diagonal_defect, unclipped, u_clipped, birth_floored,
        disc_floored, gap_capped, u_bound, disc_floor, gap_cap, original_diagonal)
    require(len(vector_values) == len(VECTOR_NAMES) == 32, 'Exactly32 prospectively named vectors')
    vectors, vector_finite = {}, {}
    for name, value in zip(VECTOR_NAMES, vector_values):
        require(tuple(value.shape) == (WIDTH,), 'Only per-hub vectors leave the callback')
        # This charged conversion always makes an owned FP64 copy, even when
        # the source was already FP64. Host byte export is separately declared.
        exported = own(value.astype(fnp.float64))
        vectors[name] = exported.tobytes()
        finite = own(fnp.isfinite(exported))
        vector_finite[name] = packet(own(fnp.all(finite)))
        del finite
        require(len(vectors[name]) == WIDTH * 8, 'Exact vector payload size')
        del exported
    return dict(statistics=stats, vectors=vectors, vector_finite=vector_finite)


def capture(fnp, values):
    live = Live()
    result = _capture(fnp, values, live)
    require(all(ref() is None for ref, _ in live.refs), 'No derived diagnostic array retained')
    result.update(explicit_temporary_peak_bytes=live.peak,
                  all_derived_arrays_released=True, no_borrowed_array_mutation=True,
                  vector_bytes=sum(len(x) for x in result['vectors'].values()))
    return result
