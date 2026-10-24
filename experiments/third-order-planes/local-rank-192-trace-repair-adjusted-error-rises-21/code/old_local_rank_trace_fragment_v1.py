"""Local rank-192 reads with trace repair; full propagated factors stay intact.

S31's tail selection and tail trace are retained. Kept rows use a shortened
local reconstruction. Their squared response-plane left operand is approximate.
Lazy canonical factors are explicitly materialized once and paid for.
"""
_LRR_RANK = 192
_LRR_PREVIOUS_FINISH = _slr_finish_factors

def _slr_finish_factors(state, factors):
    _LRR_PREVIOUS_FINISH(state, factors)
    state['local_read_factors'] = None

def _lrr_factors(state):
    if state['lazy'] is None:
        return state['F']
    value = state.get('local_read_factors')
    if value is None:
        group = state['lazy']
        prefix = _shared_old384_mm(group['map'], group['base'])
        value = fnp.concatenate([prefix, group['join']], axis=1)
        _shared_old384_finite(value, 'paid canonical local-read factors')
        state['local_read_factors'] = value
    return value

def _lrr_trace_block(U, F, lam, rows, full_rows, sign, start=0, rho=None):
    U64, F64 = U.astype(fnp.float64), F.astype(fnp.float64)
    norm2 = fnp.sum(F64 * F64, axis=0)
    if rho is None:
        rho = fnp.sum(U64 * U64) / (int(U.shape[0]) * int(U.shape[1]))
    selected = U64[rows, :]
    _OTT_DATA.setdefault('local_rank_blocks', []).append(
        (selected, F64, norm2, full_rows, sign, start))
    vector = F64 @ (lam.astype(fnp.float64) * norm2)
    return (selected @ vector) * (sign * 3 * rho)

def _slr_reconstruct(U, state):
    ctx = _OTT_CONTEXT[0]
    if ctx is None:
        return _OTT_ORIGINAL_RECON(U, state)
    m, columns = int(U.shape[0]), int(state['columns'])
    rank = min(_LRR_RANK, int(U.shape[1]))
    a = ctx['a']
    score = _OTT_Z - a
    negative_dead = (a < -_OTT_NEG_Z) & (~ctx['live'])
    score = fnp.where(negative_dead, a + _OTT_NEG_Z, score)
    kept = min(m, (int(fnp.nonzero(score >= 0.0)[0].shape[0]) + 63) // 64 * 64)
    ordered = fnp.argsort(-score, kind='stable')
    rows, tail = ordered[:kept], ordered[kept:]
    full_rows = rows if ctx['rows'] is None else ctx['rows'][rows]
    full_tail = tail if ctx['rows'] is None else ctx['rows'][tail]
    all_rows = fnp.arange(m, dtype=fnp.int64)
    all_full = all_rows if ctx['rows'] is None else ctx['rows'][all_rows]
    lam = ctx['lam'][:columns]
    F = _lrr_factors(state)
    short_F, short_U = F[:rank, :], U[:, :rank]
    old = _shared_old384_mm(short_U[rows, :], short_F)
    _shared_old384_finite(old, 'local shortened old reconstruction')
    U64 = U.astype(fnp.float64)
    rho = fnp.sum(U64 * U64) / (m * int(U.shape[1]))
    repair = fnp.zeros(ctx['n'], dtype=fnp.float64)
    group = state['lazy']
    if group is None:
        whole = _lrr_trace_block(U, F, lam, all_rows, all_full, 1, rho=rho)
        fnp.put(repair, all_full, whole)
    else:
        kept_trace = _lrr_trace_block(U, F, lam, rows, full_rows, 1, rho=rho)
        fnp.put(repair, full_rows, kept_trace)
        if kept < m:
            mapped = _shared_old384_mm(U, group['map'])
            _shared_old384_finite(mapped, 'local-rank preserved S31 tail mapping')
            split = int(group['prefix'])
            tail_trace = (_lrr_trace_block(mapped, group['base'], lam[:split], tail, full_tail, 1)
                          + _lrr_trace_block(U, group['join'], lam[split:], tail, full_tail, 1, start=split))
            fnp.put(repair, full_tail, tail_trace)
    short_trace = _lrr_trace_block(short_U, short_F, lam, rows, full_rows, -1, rho=rho)
    fnp.put(repair, full_rows, repair[full_rows] + short_trace)
    _OTT_DATA['repair'] = repair
    _OTT_DATA['local_rank_columns'] = columns
    _OTT_STATS['calls'] += 1
    _OTT_STATS['rows_omitted'] += m-kept
    inverse = fnp.full(m, kept, dtype=fnp.int64)
    fnp.put(inverse, rows, fnp.arange(kept, dtype=fnp.int64))
    return fnp.concatenate([old, fnp.zeros((1, columns), dtype=old.dtype)], axis=0)[inverse]

def _lrr_radial_delta(a, beta):
    columns = _OTT_DATA['local_rank_columns']
    energy = fnp.zeros(columns, dtype=fnp.float64)
    linear = fnp.zeros(columns, dtype=fnp.float64)
    hold = []
    for U, F, norm2, full_rows, sign, start in _OTT_DATA['local_rank_blocks']:
        width = int(F.shape[1])
        row_variance = fnp.sum(U * U, axis=1) / int(U.shape[1])
        scale = fnp.sum(a[full_rows] * row_variance)
        de = norm2 * (sign * scale)
        ds = ((U.T @ beta[full_rows]) @ F) * sign
        hold.append(fnp.add(energy[start:start+width], de, out=energy[start:start+width]))
        hold.append(fnp.add(linear[start:start+width], ds, out=linear[start:start+width]))
    return energy, linear
