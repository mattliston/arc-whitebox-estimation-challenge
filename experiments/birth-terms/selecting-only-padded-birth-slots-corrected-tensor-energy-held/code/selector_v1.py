"""Choose only below-floor padding births using the actual corrected BD base."""

def _o21_birth_prepare(score, floor):
    # Same public operations/order as select(), retaining its already-computed
    # threshold and stable full order until this layer's BD base is available.
    keep = []
    ge = score >= floor
    nz = fnp.nonzero(ge)
    keep.append((ge, nz))
    required_count = int(nz[0].shape[0])
    count = min(len(score), (required_count + 31) // 32 * 32)
    neg = -score
    order = fnp.argsort(neg, kind='stable')
    keep.append((neg, order))
    indices = order[:count]
    mask = fnp.zeros(len(score), dtype=fnp.bool_)
    fnp.put(mask, indices, True)
    _E2_LAYER['o21_birth_padding'] = dict(
        required=ge, required_count=required_count, order=order,
        original_indices=indices, original_mask=mask, result=(indices, mask))
    return indices, mask


def _o21_birth_distinct_scores(base, a, t, s):
    # For a[(q+t e)^3-(q+s e)^3], q_h=0, the all-distinct tensor
    # energy is 3[a(t-s)]^2[(sum q^2)^2-sum q^4]. The common 3 does
    # not affect ranking. Products retain the current f32 base arithmetic;
    # reductions and scalar coefficient arithmetic use f64.
    q2 = base * base
    s2 = fnp.sum(q2, axis=0, dtype=fnp.float64)
    s4 = fnp.sum(q2 * q2, axis=0, dtype=fnp.float64)
    coefficient = a * (t - s)
    return (coefficient * coefficient) * fnp.maximum(s2 * s2 - s4, 0.0)


def _o21_birth_refine(base, a, t, s, births):
    context = _E2_LAYER.get('o21_birth_padding')
    if context is None or context['original_mask'] is not births:
        return births  # Public BD calls outside the active core are unchanged.
    count = int(context['original_indices'].shape[0])
    required_count = context['required_count']
    padding = count - required_count
    if padding == 0 or count == int(base.shape[1]):
        return births
    energy = _o21_birth_distinct_scores(base, a, t, s)
    order = context['order']
    # Stable ties inherit the original gate order, including zero-energy units.
    ranked = fnp.where(context['required'][order], -1.0, energy[order])
    positions = fnp.argsort(-ranked, kind='stable')[:padding]
    # Required units form the original prefix. Sort at most31 padding positions
    # to preserve the original global gate order of the resulting roster.
    positions = fnp.sort(positions)
    filler = order[positions]
    indices = fnp.concatenate((context['original_indices'][:required_count], filler))
    mask = context['required'].copy()
    fnp.put(mask, filler, True)
    context['result'] = (indices, mask)
    return mask


def _o21_birth_finish(indices, births):
    context = _E2_LAYER.pop('o21_birth_padding', None)
    if context is None or context['original_mask'] is not births:
        return indices, births
    return context['result']
