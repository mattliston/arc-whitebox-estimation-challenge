"""Unqualified research: replace every old local row by its first-chaos trace.

Only local reconstruction changes. The full propagated Q/F tensor and every
young column remain. An endpoint TRAIN screen is required; algebra is not an
accuracy or cost claim. This fragment is not part of the S31 qualification.
"""

def _slr_reconstruct(U, state):
    ctx = _OTT_CONTEXT[0]
    if ctx is None:
        return _OTT_ORIGINAL_RECON(U, state)
    m = int(U.shape[0])
    columns = int(state['columns'])
    all_rows = fnp.arange(m, dtype=fnp.int64)
    lam = ctx['lam'][:columns]
    group = state['lazy']
    if group is None:
        delta = _ott_trace(U, state['F'], lam, all_rows)
    else:
        mapped = _shared_old384_mm(U, group['map'])
        _shared_old384_finite(mapped, 'all-old trace mapped basis')
        split = int(group['prefix'])
        delta = (_ott_trace(mapped, group['base'], lam[:split], all_rows)
                 + _ott_trace(U, group['join'], lam[split:], all_rows))
    full_rows = all_rows if ctx['rows'] is None else ctx['rows'][all_rows]
    repair = fnp.zeros(ctx['n'], dtype=fnp.float64)
    fnp.put(repair, full_rows, delta)
    _OTT_DATA['repair'] = repair
    _OTT_DATA['full_tail'] = full_rows
    _OTT_STATS['calls'] += 1
    _OTT_STATS['rows_omitted'] += m
    return fnp.zeros((m, columns), dtype=U.dtype)
