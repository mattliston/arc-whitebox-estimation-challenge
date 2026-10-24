# [opus-ottband] OTT BAND stand-in (lead 15:0xZ item 1) on S2tr 46e9315d: rows with a = mu/sigma in (2.5, 3.0] (OTTP30's band:
# -0.50% xMSE for +0.51% bill under FULL reconstruction) are kept, but reconstructed against only the top-k frame coordinates,
# ranked by ||U_band[:, c]||^2 * sum_j lam_j^2 F[c, j]^2 (base and join blocks ranked separately in the lazy L13 group); the
# kept-row cutoff moves 2.5 -> 3.0 (rounded to 64 as before), band rows leave the tail (no trace repair, as in OTTP30).
# Both live OTT paths are replaced by one copy of their logic: _astra_bcc_recon_impl (LCR layers 10-14) and _LCR_PARENT_RECON
# (layers 7-9); everything else (u/v split, lazy group, trace repair of the remaining tail, radial bookkeeping) is the original.
import hashlib, sys
O = '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1'
H = O + '/lead/runs/opus_ladder2h_v1'
src = open(O + '/lead/runs/opus_s2tr_v1/candidate_S2tr_v1.py').read()
assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
FRAG = '''_L2H_OTTB_Z = 3.0
_L2H_OTTB_K = %d
_L2H_OTTB_STATS = dict(calls=0, band_rows=0)
_L2H_OTTB_PLAIN_PARENT = _LCR_PARENT_RECON


def _l2h_topk(Ub, F, lam):
    k = _L2H_OTTB_K
    if k >= int(F.shape[0]):
        return _shared_old384_mm(Ub, F)
    wF = fnp.sum((F * lam[None, :]) ** 2, axis=1)
    eU = fnp.sum(Ub * Ub, axis=0)
    K = fnp.argsort(-(eU * wF), kind='stable')[:k]
    return _shared_old384_mm(Ub[:, K], F[K, :])


def _l2h_ottb(U, state, lcr):
    ctx = _OTT_CONTEXT[0]
    if ctx is None:
        return _OTT_ORIGINAL_RECON(U, state)
    a = ctx['a']
    m = int(U.shape[0])
    score = _L2H_OTTB_Z - a
    negative_dead = (a < -_OTT_NEG_Z) & (~ctx['live'])
    score = fnp.where(negative_dead, a + _OTT_NEG_Z, score)
    kept = min(m, (int(fnp.nonzero(score >= 0.0)[0].shape[0]) + 63) // 64 * 64)
    if kept == m:
        return _OTT_ORIGINAL_RECON(U, state)
    ordered = fnp.argsort(-score, kind='stable')
    rows, tail = ordered[:kept], ordered[kept:]
    n25 = int(fnp.nonzero(score >= _L2H_OTTB_Z - _OTT_Z)[0].shape[0])
    c64 = min(kept, (n25 + 63) // 64 * 64)
    core, band = rows[:c64], rows[c64:]
    nb = int(band.shape[0])
    columns = int(state['columns'])
    lam = ctx['lam'][:columns]
    group = state['lazy']
    if group is None:
        old = _OTT_ORIGINAL_RECON(U[core], state) if c64 else None
        ob = _l2h_topk(U[band], state['F'], lam) if nb else None
        delta = _ott_trace(U, state['F'], lam, tail)
    else:
        mapped = _shared_old384_mm(U, group['map'])
        _shared_old384_finite(mapped, 'tail-trace lazy mapped full basis')
        split = int(group['prefix'])
        if lcr:
            uv = _uvp_state(state)
            _slr_require(uv[0] == 'l', 'local common lazy split')
            old = _astra_bcc_lazy(mapped, U, core, uv, columns) if c64 else None
        else:
            old = fnp.concatenate([_shared_old384_mm(mapped[core], group['base']), _shared_old384_mm(U[core], group['join'])], axis=1) if c64 else None
        ob = fnp.concatenate([_l2h_topk(mapped[band], group['base'], lam[:split]), _l2h_topk(U[band], group['join'], lam[split:])], axis=1) if nb else None
        delta = (_ott_trace(mapped, group['base'], lam[:split], tail)
                 + _ott_trace(U, group['join'], lam[split:], tail))
    if old is None:
        old = ob
    elif ob is not None:
        old = fnp.concatenate([old, ob], axis=0)
    full_tail = tail if ctx['rows'] is None else ctx['rows'][tail]
    repair = fnp.zeros(ctx['n'], dtype=fnp.float64)
    fnp.put(repair, full_tail, delta)
    _OTT_DATA['repair'] = repair
    _OTT_DATA['full_tail'] = full_tail
    _OTT_STATS['calls'] += 1
    _OTT_STATS['rows_omitted'] += m - kept
    _L2H_OTTB_STATS['calls'] += 1
    _L2H_OTTB_STATS['band_rows'] += nb
    return _exact_expand_rows(old, rows, m)


def _l2h_ottb_plain(U, state):
    return _l2h_ottb(U, state, False)


def _astra_bcc_recon_impl(U, state):
    if _UVP_CUR[0] not in _LCR_LAYERS:
        return _LCR_PARENT_RECON(U, state)
    return _l2h_ottb(U, state, True)


_LCR_PARENT_RECON = _l2h_ottb_plain
'''
for tag, k in (('ottb64', 64), ('ottb32', 32), ('ottbfull', 384)):  # v3: core = S2tr's own 64-rounded kept block (same kernel shapes), band = the extra kept rows up to z 3.0
    out = src + '\n\n# BEGIN [opus-ottband] ARM %s (append LAST to S2tr 46e9315d)\n' % tag + FRAG % k + '# END [opus-ottband] ARM %s\n' % tag
    p = H + '/cand_s2tr_l2h_%s_v3.py' % tag
    open(p, 'w').write(out)
    print(tag, hashlib.sha256(out.encode()).hexdigest()[:8], p)
