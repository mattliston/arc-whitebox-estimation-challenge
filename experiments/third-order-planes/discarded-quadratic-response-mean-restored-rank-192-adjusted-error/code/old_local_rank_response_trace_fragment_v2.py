"""Add the Gaussian mean of the discarded squared local response operand."""
_LRR_V1_RECON = _slr_reconstruct
_LRR_V1_INNER = _cp_slr_inner_sq

def _slr_reconstruct(U, state):
    answer = _LRR_V1_RECON(U, state)
    ctx = _OTT_CONTEXT[0]
    if ctx is not None and _OTT_DATA.get('local_rank_blocks'):
        blocks = _OTT_DATA['local_rank_blocks']
        full_F, full_norm2 = blocks[0][1:3]
        short_norm2 = blocks[-1][2]
        lam = ctx['lam'][:int(state['columns'])].astype(fnp.float64)
        U64 = U.astype(fnp.float64)
        rho = fnp.sum(U64 * U64) / (int(U.shape[0]) * int(U.shape[1]))
        trace = (full_F @ (lam * (full_norm2-short_norm2))) * rho
        _OTT_DATA['local_inner_trace'] = trace.astype(fnp.float32)
        _OTT_DATA['local_inner_owner'] = id(state)
    return answer

def _cp_slr_inner_sq(old_left, lam_old, state):
    inner = _LRR_V1_INNER(old_left, lam_old, state)
    if _OTT_DATA.get('local_inner_owner') == id(state):
        # Zero old rows are the unchanged S31 tail mask. Restore only rows
        # whose shortened local reconstruction is actually present.
        present = (fnp.sum(old_left, axis=1) > 0).astype(fnp.float32)
        inner = inner + present[:, None] * _OTT_DATA['local_inner_trace'][None, :]
    return inner
