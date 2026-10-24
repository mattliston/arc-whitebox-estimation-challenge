# Appended to the unchanged bank prefix. All numerical operations use FNP.
_POLAR_ENABLED = True
_POLAR_PARENT_INNER = _slr_inner
_POLAR_PARENT_READ = _shared_old384_read

def _polar_pair_product(left, right, lam, cohorts, start, stop, product):
    width = int(stop) - int(start)
    if (width <= 0 or len(left.shape) != 2 or len(right.shape) != 2
            or int(left.shape[1]) != width or int(right.shape[1]) != width
            or tuple(lam.shape) != (width,)
            or str(left.dtype) != 'float32' or str(right.dtype) != 'float32'
            or str(lam.dtype) != 'float32'):
        raise RuntimeError('Polarized read dimensions and FP32 contract')
    packed_left = fnp.empty(tuple(left.shape), dtype=fnp.float32)
    packed_right = fnp.empty(tuple(right.shape), dtype=fnp.float32)
    cursor = int(start)
    for cohort_start, cohort_width in cohorts:
        cohort_start, cohort_width = int(cohort_start), int(cohort_width)
        end = cohort_start + cohort_width
        if end <= start or cohort_start >= stop:
            continue
        if cohort_start != cursor or cohort_width <= 0 or cohort_width % 2 or end > stop:
            raise RuntimeError('Polarized read intact cohort boundary')
        pairs = cohort_width // 2
        lo = cohort_start - int(start)
        mid, hi = lo + pairs, lo + cohort_width
        a = lam[lo:mid]
        if not bool(fnp.all(a == -lam[mid:hi])):
            raise RuntimeError('Polarized read signed pair opposition')
        h = (left[:,lo:mid] + left[:,mid:hi]) * 0.5
        d = left[:,lo:mid] - left[:,mid:hi]
        H = (right[:,lo:mid] + right[:,mid:hi]) * 0.5
        D = right[:,lo:mid] - right[:,mid:hi]
        fnp.copyto(packed_left[:,lo:mid], h*h + (d*d)*0.25)
        fnp.copyto(packed_left[:,mid:hi], (h*d)*2.0)
        fnp.copyto(packed_right[:,lo:mid], D*a[None,:])
        fnp.copyto(packed_right[:,mid:hi], H*a[None,:])
        del h,d,H,D,a
        cursor = end
    if cursor != int(stop):
        raise RuntimeError('Polarized read complete ordered cohort coverage')
    return product(packed_left, packed_right.T)

def _slr_inner(old_left, lam_old, state):
    if not _POLAR_ENABLED:
        return _POLAR_PARENT_INNER(old_left, lam_old, state)
    columns = int(state['columns'])
    _slr_layout(state, columns)
    _slr_require(len(old_left.shape) == 2 and int(old_left.shape[1]) == columns
                 and tuple(lam_old.shape) == (columns,)
                 and str(old_left.dtype) == str(lam_old.dtype) == 'float32',
                 'complete original weighted repeated-index read')
    group = state['lazy']
    if group is None:
        return _polar_pair_product(old_left,state['F'],lam_old,state['cohorts'],0,columns,_shared_old384_mm)
    split = int(group['prefix'])
    left0, left1 = old_left[:,:split], old_left[:,split:]
    inner0 = _polar_pair_product(left0,group['base'],lam_old[:split],state['cohorts'],0,split,_shared_old384_mm)
    inner1 = _polar_pair_product(left1,group['join'],lam_old[split:],state['cohorts'],split,columns,_shared_old384_mm)
    mapped = _shared_old384_mm(inner0,group['map'].T)
    _shared_old384_finite(mapped,'lazy mapped response inner')
    inner = mapped + inner1
    del inner0,inner1,mapped
    return inner

def _shared_old384_read(left, right, lam, live_indices, state):
    if not _POLAR_ENABLED:
        return _POLAR_PARENT_READ(left,right,lam,live_indices,state)
    old_columns = int(state['columns'])
    columns = int(lam.shape[0])
    if not old_columns:
        return _polar_pair_product(left,right,lam,state['cohorts'],0,columns,_mm)
    old_left = left[:,:old_columns]
    inner = _slr_inner(old_left,lam[:old_columns],state)
    old_plane = _shared_old384_mm(inner,state['pre_basis'][live_indices,:].T)
    young_left, young_right = left[:,old_columns:], right[:,old_columns:]
    young_plane = _polar_pair_product(young_left,young_right,lam[old_columns:],state['cohorts'],old_columns,columns,_mm)
    return young_plane + old_plane
