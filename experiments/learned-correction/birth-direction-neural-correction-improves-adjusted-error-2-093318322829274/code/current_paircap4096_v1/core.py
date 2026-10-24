"""Inert current-bank pair retirement helpers; injected into the parent globals.

No imports, arrays, model, or estimator is created at module definition time.
The builder adds one derived _predict_binary with three exact-count edits.
Numerical globals fnp, _mm and _cp_diagonal_cube are the frozen parent's.
"""

_PAIR_CAP = 4096  # SIGNED pairs: at most8192 logical CP columns after retirement.


def _paircap_sorted(indices):
    """Stable numeric order; preserves original physical/canonical ordering."""
    return indices[fnp.argsort(indices, kind='stable')]


def _paircap_materialize(p, newborn):
    """Temporary physical post cubes, not a permanently stored next bank.

    p holds the inherited numeric prefix. Each newborn's base has zero at
    its own selected coordinate; the FP32 t/s values replace that entry.
    This describes physical cubes in real arithmetic. The original shared
    push and a literal FP32 dense push need not be byte-identical.
    """
    if newborn is None:
        return p
    old_columns, selected, base, t, s = newborn
    n, count = int(p.shape[0]), len(selected)
    if int(p.shape[1]) != old_columns or tuple(base.shape) != (n, count):
        raise ValueError('Pair-cap newborn prefix/base dimensions')
    if tuple(t.shape) != (count,) or tuple(s.shape) != (count,):
        raise ValueError('Pair-cap newborn diagonal dimensions')
    first, second = base.copy(), base.copy()
    positions = selected * count + fnp.arange(count, dtype=fnp.int64)
    fnp.put(first, positions, t)
    fnp.put(second, positions, s)
    return fnp.concatenate([p, first, second], axis=1)


def _paircap_energy(physical, lam, positive, negative):
    """The old whole-pair FP64 self-energy formula, including cancellation.

    Negative rounded norm-squared values are clamped exactly as in the old
    selection law. This is a numerical selector, not an interval certificate.
    """
    first = physical[:, positive].astype(fnp.float64)
    second = physical[:, negative].astype(fnp.float64)
    aa = fnp.sum(first * first, axis=0)
    bb = fnp.sum(second * second, axis=0)
    ab = fnp.sum(first * second, axis=0)
    weight = lam[positive].astype(fnp.float64)
    energy = weight ** 2 * fnp.maximum(aa ** 3 + bb ** 3 - 2 * ab ** 3, 0)
    if not fnp.all(fnp.isfinite(energy)):
        raise FloatingPointError('Nonfinite pair retirement energy')
    return energy


def _paircap_dropped_plane(physical, lam, drop_columns, live_indices):
    """Actual dropped cubes: current FP32 live-plane read, FP64 all-unit diag.

    Off-diagonal arithmetic deliberately uses the current _mm, not an
    unmetered NumPy GEMM or a full FP64 shadow. Diagonal cubes use the
    parent's same _cp_diagonal_cube and FP64 lambda convention.
    """
    n = int(physical.shape[0])
    dropped = physical[:, drop_columns]
    weights = lam[drop_columns]
    sub = dropped[live_indices]
    values = _mm(sub * sub, (sub * weights).T).astype(fnp.float64)
    out = fnp.zeros((n, n), dtype=fnp.float64)
    fnp.put(out, live_indices[:, None] * n + live_indices[None, :], values)
    diagonal = fnp.sum(_cp_diagonal_cube(dropped.astype(fnp.float64)) *
                       weights.astype(fnp.float64), axis=1)
    fnp.fill_diagonal(out, diagonal)
    return out


def _paircap_pad_right_columns(left, right):
    """Paid geometric padding only; never round the selected pair count.

    Arbitrary retained old/new partitions would otherwise trigger _mm's
    ordinary-GEMM fallback. Padding changes no exact-real column product;
    floating kernel parity remains a separate, unexecuted qualification.
    """
    width = int(right.shape[1])
    if width == 0 or width % 32 == 0:
        return _mm(left, right)
    padded_width = (width + 31) // 32 * 32
    padded = fnp.zeros((int(right.shape[0]), padded_width), dtype=fnp.float32)
    fnp.copyto(padded[:, :width], right)
    return _mm(left, padded)[:, :width]


def _paircap_push_bank_shared_newborn(w, p, previous, newborn):
    """The parent's two-product shared-newborn path, with paid column padding."""
    wp = w[previous, :].T
    if newborn is None:
        return _paircap_pad_right_columns(wp, p[previous, :])
    old_columns, selected, base, t, s = newborn
    inherited = _paircap_pad_right_columns(wp, p[previous, :old_columns])
    common = _paircap_pad_right_columns(wp, base[previous, :])
    directions = w[selected, :].T
    return fnp.concatenate([inherited, common + directions * t[None, :],
                            common + directions * s[None, :]], axis=1)


def _paircap_retire(p, lam, newborn, positive, negative, cp_post,
                    live_indices, cap=_PAIR_CAP):
    """Append canonical pair identities, then retire whole pairs if needed.

    Inputs are AFTER the unchanged parent gate, birth and lambda writes,
    BEFORE its unchanged rest=target-cp_post-trace_post star reset. positive
    and negative describe only the incoming logical pairs, now in p.

    Return (p, lam, newborn, positive, negative, cp_post). Retained pair
    identities stay in original chronological/within-cohort order, with
    matching positive/negative column maps. No numeric tail persists when
    it is still owned by the newest tuple. cap is exposed for a tiny test;
    the derived estimator passes the one fixed literal _PAIR_CAP=4096.
    """
    if not isinstance(cap, int) or cap < 1:
        raise ValueError('Positive fixed integer pair cap required')
    n, old_columns = int(p.shape[0]), int(p.shape[1])
    old_pairs = len(positive)
    if len(negative) != old_pairs or old_columns != 2 * old_pairs:
        raise ValueError('Incoming complete-pair column count')
    if newborn is None:
        new_count = 0
    else:
        declared_old, selected, base, t, s = newborn
        new_count = len(selected)
        if declared_old != old_columns or tuple(base.shape) != (n, new_count):
            raise ValueError('Newborn does not follow inherited prefix')
    if len(lam) != old_columns + 2 * new_count:
        raise ValueError('Logical lambda/newborn column count')
    offset = fnp.arange(new_count, dtype=fnp.int64)
    positive = fnp.concatenate([positive, old_columns + offset])
    negative = fnp.concatenate([negative, old_columns + new_count + offset])
    total_pairs = old_pairs + new_count
    if total_pairs <= cap:
        # No numerical state rewrite or energy/repack work below the cap.
        return p, lam, newborn, positive, negative, cp_post

    all_columns = _paircap_sorted(fnp.concatenate([positive, negative]))
    if not fnp.all(all_columns == fnp.arange(len(lam), dtype=fnp.int64)):
        raise ValueError('Pair identities must partition the logical bank')
    if not fnp.all(lam[positive] == -lam[negative]):
        raise ValueError('Pair lambdas must remain opposite')

    physical = _paircap_materialize(p, newborn)
    energy = _paircap_energy(physical, lam, positive, negative)
    order = fnp.argsort(-energy, kind='stable')
    chosen = _paircap_sorted(order[:cap])
    dropped_pairs = order[cap:]
    drop_columns = _paircap_sorted(fnp.concatenate([
        positive[dropped_pairs], negative[dropped_pairs]]))
    removed = _paircap_dropped_plane(physical, lam, drop_columns, live_indices)
    adjusted_cp_post = cp_post - removed

    # Selecting by energy does not reorder retained pairs by their energy.
    # Original physical order is retained within the OLD numeric prefix;
    # the NEW retained pair partition remains the two shared virtual tails.
    keep_old = chosen[chosen < old_pairs]
    keep_new = chosen[chosen >= old_pairs] - old_pairs
    old_keep_columns = _paircap_sorted(fnp.concatenate([
        positive[keep_old], negative[keep_old]]))
    kept_p = p[:, old_keep_columns]
    kept_old_columns = len(old_keep_columns)
    inverse = fnp.zeros(old_columns, dtype=fnp.int64)
    fnp.put(inverse, old_keep_columns,
            fnp.arange(kept_old_columns, dtype=fnp.int64))
    next_positive = inverse[positive[keep_old]]
    next_negative = inverse[negative[keep_old]]
    old_lam = lam[old_keep_columns]
    kept_new_count = len(keep_new)
    if kept_new_count:
        selected_new = selected[keep_new]
        next_newborn = (kept_old_columns, selected_new, base[:, keep_new],
                        t[keep_new], s[keep_new])
        new_positive = lam[old_columns + keep_new]
        new_negative = lam[old_columns + new_count + keep_new]
        next_lam = fnp.concatenate([old_lam, new_positive, new_negative])
        offsets = fnp.arange(kept_new_count, dtype=fnp.int64)
        next_positive = fnp.concatenate([next_positive, kept_old_columns + offsets])
        next_negative = fnp.concatenate([
            next_negative, kept_old_columns + kept_new_count + offsets])
    else:
        next_newborn, next_lam = None, old_lam
    if len(next_lam) != 2 * cap or len(next_positive) != cap or len(next_negative) != cap:
        raise ValueError('Retired bank has the wrong fixed logical capacity')
    return kept_p, next_lam, next_newborn, next_positive, next_negative, adjusted_cp_post
