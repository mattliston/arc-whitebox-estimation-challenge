"""Complete signed-pair energy selection; no change to the Gram metric."""
_O21_PAIR_JSK_INDEX = {}
_O21_PAIR_JSK_STATS = dict(calls=0, pairs=0, selected_pairs=0)

def _o21_pair_jsk_scores(J, norm2, weight, widths):
    # In exact arithmetic, ||lambda(p^3-q^3)||_F^2 is
    # lambda^2 * [2d(3u^2-3ud+d^2)+6u v^2], with
    # u=(||p||^2+||q||^2)/2, v=(||p||^2-||q||^2)/2,
    # d=||p-q||^2/2. This avoids subtracting nearly equal cubes.
    hold = _E2Hold()
    norm = hold(norm2.astype(fnp.float64))
    scores, at = [], 0
    for width in widths:
        p = hold(J[:, at:at + width])
        q = hold(J[:, at + width:at + 2 * width])
        delta = hold(p - q)
        d = hold(hold(fnp.sum(hold(delta * delta), axis=0, dtype=fnp.float64)) * .5)
        a, b = hold(norm[at:at + width]), hold(norm[at + width:at + 2 * width])
        u = hold(hold(a + b) * .5)
        v = hold(hold(a - b) * .5)
        energy = hold(hold(hold(2.0 * d) * hold(hold(hold(3.0 * u) * hold(u - d)) + hold(d * d))) + hold(hold(6.0 * u) * hold(v * v)))
        scores.append(hold(hold(weight[at:at + width].astype(fnp.float64)) * energy))
        at += 2 * width
    return scores[0] if len(scores) == 1 else fnp.concatenate(scores)

def _o21_pair_jsk_columns(J, norm2, weight, cohort_widths, keep):
    widths = tuple(int(x) // 2 for x in cohort_widths)
    if any(int(x) <= 0 or int(x) % 2 for x in cohort_widths):
        raise RuntimeError('Pair-energy JSK requires complete positive two-leg cohorts')
    columns = int(J.shape[1])
    if 2 * sum(widths) != columns or tuple(norm2.shape) != (columns,) or tuple(weight.shape) != (columns,) or int(keep) % 2:
        raise RuntimeError('Pair-energy JSK column roster')
    hold = _E2Hold()
    pair_indices = _O21_PAIR_JSK_INDEX.get(widths)
    if pair_indices is None:
        left, right, at = [], [], 0
        for width in widths:
            left.extend(range(at, at + width))
            right.extend(range(at + width, at + 2 * width))
            at += 2 * width
        pair_indices = fnp.asarray([left, right], dtype=fnp.int64)
        _O21_PAIR_JSK_INDEX[widths] = pair_indices
    score = hold(_o21_pair_jsk_scores(J, norm2, weight, widths))
    selected = hold(fnp.argsort(hold(-score), kind='stable')[:min(int(keep) // 2, sum(widths))])
    result = hold(pair_indices[:, selected]).reshape((-1,))
    _O21_PAIR_JSK_STATS['calls'] += 1
    _O21_PAIR_JSK_STATS['pairs'] += sum(widths)
    _O21_PAIR_JSK_STATS['selected_pairs'] += int(selected.shape[0])
    return result
